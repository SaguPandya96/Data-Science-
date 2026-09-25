"""The rebalancing loop.

Each cycle: settle last cycle's orders, reconcile against the venues, mark the portfolio, find
sleeves outside their bands, drop the ones whose market is closed or whose quote is stale, size
trades back to the band edge, and send them through the risk gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from .audit import AuditLog, OrderRecord
from .models import CASH_INSTRUMENTS, Model, Sleeve
from .risk import RiskGate, RiskLimits, round_down
from .sessions import MarketCalendar, Session
from .venue import InstrumentInfo, Order, OrderType, Quote, Side, TimeInForce, Venue

_EPS = 1e-9


@dataclass(frozen=True)
class EngineConfig:
    min_trade: float = 25.0
    # How far past mid a limit is priced, so it crosses the spread in thin sessions.
    limit_collar: float = 0.002
    # Headroom kept on buys for fees, so fills can't push cash under the floor.
    fee_buffer: float = 0.005
    # Baseline for comparison: behave like a 9:30-16:00 only rebalancer.
    regular_hours_only: bool = False


@dataclass(frozen=True)
class SleeveMark:
    value: float
    weight: float
    stale: bool


@dataclass
class CycleReport:
    ts: datetime
    session: Session
    account_value: float
    marks: dict[str, SleeveMark]
    orders: list[OrderRecord] = field(default_factory=list)
    halted: str | None = None


@dataclass
class _Intent:
    sleeve: Sleeve
    instrument: str
    delta: float  # dollars, positive buys
    reasons: list[str]


class Engine:
    def __init__(
        self,
        model: Model,
        venues: dict[str, Venue],
        instruments: dict[str, InstrumentInfo],
        calendar: MarketCalendar,
        *,
        limits: RiskLimits | None = None,
        config: EngineConfig | None = None,
        audit: AuditLog | None = None,
    ):
        missing = [i for s in model.invested for i in s.all_instruments if i not in instruments]
        if missing:
            raise ValueError(f"no venue configured for {missing}")
        self.model = model
        self.venues = venues
        self.instruments = instruments
        self.calendar = calendar
        self.config = config or EngineConfig()
        self.audit = audit or AuditLog()
        self.gate = RiskGate(limits or RiskLimits(), calendar, instruments, self.audit)
        self.ledger: dict[str, float] = {}
        self.last_mid: dict[str, float] = {}
        self._open: dict[str, tuple[str, OrderRecord]] = {}  # client id -> (venue, record)
        self._seen_fills: set[str] = set()
        self._fills_since: dict[str, datetime] = {}
        self._off_need: dict[tuple[datetime, str], float] = {}
        self._off_done: dict[tuple[datetime, str], float] = {}
        self._off_orders: dict[str, tuple[tuple[datetime, str], float]] = {}
        self._started = False

    # --- lifecycle -------------------------------------------------------------------------

    def start(self, now: datetime) -> None:
        self.ledger = self._venue_positions()
        self._fills_since = {name: now for name in self.venues}
        self._started = True

    def record_cash_flow(self, now: datetime, amount: float) -> None:
        self.ledger["USD"] = self.ledger.get("USD", 0.0) + amount
        kind = "deposit" if amount > 0 else "withdrawal"
        self.audit.decide(now, "cash", "cash_flow", f"{kind} of ${abs(amount):,.2f}", dedupe=False)

    def snapshot(self, now: datetime) -> CycleReport:
        """Mark the portfolio without trading."""
        session = self.calendar.session_at(now).session
        marks, value = self._mark(now, self._quotes(now))
        return CycleReport(now, session, value, marks or {})

    def cycle(self, now: datetime) -> CycleReport:
        if not self._started:
            raise RuntimeError("call start() before the first cycle")
        session = self.calendar.session_at(now).session
        self._settle(now)
        quotes = self._quotes(now)
        self._reconcile(now)
        self.gate.crypto_buys_paused(now)  # raises and clears the drawdown alert as prices move
        marks, value = self._mark(now, quotes)
        report = CycleReport(now, session, value, marks)
        if marks is None:
            report.marks = {}
            return report

        halt = self.gate.halt_reason(now)
        if halt:
            report.halted = halt
            self.audit.decide(now, "*", "halt", halt)
            return report

        intents = self._plan(now, session, quotes, marks, value)
        for intent in intents:
            record = self._send(now, session, intent, quotes, value)
            if record is not None:
                report.orders.append(record)
        self._pull_fills(now)
        return report

    # --- settle and reconcile --------------------------------------------------------------

    def _settle(self, now: datetime) -> None:
        self._pull_fills(now)
        # Unfilled limits are cancelled and re-quoted from fresh prices this cycle.
        for client_id, (venue, record) in list(self._open.items()):
            if self.venues[venue].cancel(record.order_id):
                record.status = "canceled"
                self.gate.release_unfilled(client_id)
                self._release_off_hours(client_id)
                del self._open[client_id]

    def _pull_fills(self, now: datetime) -> None:
        by_order = {rec.order_id: cid for cid, (_, rec) in self._open.items()}
        for name, venue in self.venues.items():
            for fill in venue.fills(self._fills_since.get(name, now)):
                if fill.fill_id in self._seen_fills:
                    continue
                self._seen_fills.add(fill.fill_id)
                sign = 1 if fill.side is Side.BUY else -1
                self.ledger[fill.instrument] = (
                    self.ledger.get(fill.instrument, 0.0) + sign * fill.qty
                )
                self.ledger["USD"] = (
                    self.ledger.get("USD", 0.0) - sign * fill.qty * fill.price - fill.fee
                )
                cid = by_order.get(fill.order_id)
                if cid is not None:
                    _, record = self._open.pop(cid)
                    record.status, record.fill_price, record.fee = "filled", fill.price, fill.fee
                    self.gate.forget(cid)
                    self._off_orders.pop(cid, None)
            self._fills_since[name] = now

    def _reconcile(self, now: datetime) -> None:
        actual = self._venue_positions()
        gap = 0.0
        worst = ""
        for inst in set(actual) | set(self.ledger):
            diff = abs(actual.get(inst, 0.0) - self.ledger.get(inst, 0.0))
            if diff <= _EPS:
                continue
            price = 1.0 if inst in CASH_INSTRUMENTS else self.last_mid.get(inst, math.inf)
            dollars = diff * price
            gap += dollars
            worst = f"{inst} engine {self.ledger.get(inst, 0.0):,.6g} vs venue {actual.get(inst, 0.0):,.6g}"
        if gap > self.gate.limits.max_reconciliation_gap:
            self.gate.halt(now, f"reconciliation gap ${gap:,.2f} ({worst})")

    def _venue_positions(self) -> dict[str, float]:
        merged: dict[str, float] = {}
        for venue in self.venues.values():
            for inst, qty in venue.positions().items():
                merged[inst] = merged.get(inst, 0.0) + qty
        return merged

    # --- mark ------------------------------------------------------------------------------

    def _quotes(self, now: datetime) -> dict[str, Quote]:
        fresh: dict[str, Quote] = {}
        for inst, info in self.instruments.items():
            quote = self.venues[info.venue].quote(inst)
            if quote is None:
                continue
            self.last_mid[inst] = quote.mid
            if quote.age(now) <= self.gate.limits.max_quote_age:
                fresh[inst] = quote
                self.gate.observe(inst, quote.ts, quote.mid)
        return fresh

    def _mark(
        self, now: datetime, fresh: dict[str, Quote]
    ) -> tuple[dict[str, SleeveMark] | None, float]:
        values: dict[str, tuple[float, bool]] = {}
        for sleeve in self.model.sleeves:
            total, stale = 0.0, False
            for inst in sleeve.all_instruments:
                qty = self.ledger.get(inst, 0.0)
                if inst in CASH_INSTRUMENTS:
                    total += qty
                    continue
                if abs(qty) <= _EPS:
                    continue
                if inst not in self.last_mid:
                    self.audit.alert(now, "critical", f"cannot mark {inst}: no price seen yet")
                    return None, 0.0
                total += qty * self.last_mid[inst]
                # Closed markets are marked at their last print and flagged; bands never trade off them.
                stale = stale or inst not in fresh
            values[sleeve.id] = (total, stale)
        account = sum(v for v, _ in values.values())
        if account <= 0:
            return None, 0.0
        return {sid: SleeveMark(v, v / account, s) for sid, (v, s) in values.items()}, account

    # --- plan ------------------------------------------------------------------------------

    def _route(
        self, sleeve: Sleeve, session: Session, side: Side, fresh: dict[str, Quote], now: datetime
    ) -> tuple[str | None, str]:
        """Pick the instrument to trade this sleeve with right now, or say why there isn't one."""
        if self.config.regular_hours_only and session is not Session.REGULAR:
            return None, f"regular-hours-only, {session} session"
        names = sleeve.instruments_for(session.value)
        if not names:
            return None, f"frozen: no instrument for the {session} session"
        why = ""
        for inst in names:
            info = self.instruments[inst]
            if info.venue in self.gate.venue_halts:
                why = f"venue {info.venue} halted"
                continue
            status = self.venues[info.venue].session_status(inst)
            if not status.is_open:
                why = f"frozen: {inst} market closed ({status.session})"
                continue
            if inst not in fresh:
                why = f"stale quote for {inst}, waiting for a fresh print"
                continue
            if side is Side.SELL and self.ledger.get(inst, 0.0) <= _EPS:
                why = f"nothing to sell in {inst}"
                continue
            if side is Side.BUY and info.asset_class == "crypto":
                paused = self.gate.crypto_buys_paused(now)
                if paused:
                    why = f"crypto buys paused: {paused}"
                    continue
            return inst, ""
        return None, why

    def _plan(
        self,
        now: datetime,
        session: Session,
        fresh: dict[str, Quote],
        marks: dict[str, SleeveMark],
        value: float,
    ) -> list[_Intent]:
        intents: dict[str, _Intent] = {}
        weight_after = {s.id: marks[s.id].weight for s in self.model.invested}

        for sleeve in self.model.invested:
            lo, hi = sleeve.band_edges()
            w = marks[sleeve.id].weight
            if w < lo - _EPS:
                delta, why = (lo - w) * value, f"below band: {w:.2%} vs {lo:.2%}-{hi:.2%}"
            elif w > hi + _EPS:
                delta, why = (hi - w) * value, f"above band: {w:.2%} vs {lo:.2%}-{hi:.2%}"
            else:
                self.audit.decide(now, sleeve.id, "ok", "in band")
                continue
            inst, blocked = self._route(
                sleeve, session, Side.BUY if delta > 0 else Side.SELL, fresh, now
            )
            if inst is None:
                self.audit.decide(now, sleeve.id, "hold", f"{why}; {blocked}")
                continue
            intents[sleeve.id] = _Intent(sleeve, inst, delta, [why])
            weight_after[sleeve.id] = w + delta / value

        self._plan_cash_flow(now, session, fresh, marks, value, intents, weight_after)
        return self._size(now, session, fresh, marks, value, list(intents.values()))

    def _plan_cash_flow(self, now, session, fresh, marks, value, intents, weight_after) -> None:
        """Put excess cash to work in underweight sleeves, or raise cash from overweight ones,
        before anything that is merely in band gets sold."""
        cash = self.model.cash
        edges = cash.band_edges()
        if edges is None:
            lo, hi = self.model.cash_floor, math.inf
        else:
            lo, hi = max(edges[0], self.model.cash_floor), edges[1]
        cash_w = marks[cash.id].weight
        buys = sum(i.delta for i in intents.values() if i.delta > 0)
        sells = -sum(i.delta for i in intents.values() if i.delta < 0)

        if cash_w > hi + _EPS:
            spare = (cash_w - hi) * value - buys
            side, reason = Side.BUY, f"cash flow: cash {cash_w:.2%} above {hi:.2%}, deploying"
            gaps = {s.id: (s.target - weight_after[s.id]) * value for s in self.model.invested}
        elif cash_w < lo - _EPS:
            spare = (lo - cash_w) * value - sells
            side, reason = Side.SELL, f"cash flow: cash {cash_w:.2%} below {lo:.2%}, raising"
            gaps = {s.id: (weight_after[s.id] - s.target) * value for s in self.model.invested}
        else:
            return
        if spare <= _EPS:
            return

        eligible: dict[str, tuple[float, str]] = {}
        for sleeve in self.model.invested:
            gap = gaps[sleeve.id]
            if gap <= _EPS:
                continue
            existing = intents.get(sleeve.id)
            if existing is not None:
                if (existing.delta > 0) != (side is Side.BUY):
                    continue
                eligible[sleeve.id] = (gap, existing.instrument)
                continue
            inst, _ = self._route(sleeve, session, side, fresh, now)
            if inst is not None:
                eligible[sleeve.id] = (gap, inst)
        total_gap = sum(g for g, _ in eligible.values())
        if total_gap <= _EPS:
            self.audit.decide(now, "cash", "hold", f"{reason}, but no tradable sleeve to use it")
            return
        use = min(spare, total_gap)
        sign = 1 if side is Side.BUY else -1
        for sid, (gap, inst) in eligible.items():
            amount = sign * use * gap / total_gap
            if sid in intents:
                intents[sid].delta += amount
                intents[sid].reasons.append(reason)
            else:
                intents[sid] = _Intent(self.model.sleeve(sid), inst, amount, [reason])

    def _size(self, now, session, fresh, marks, value, intents: list[_Intent]) -> list[_Intent]:
        limits = self.gate.limits
        max_order = limits.max_order_pct * value
        for intent in intents:
            info = self.instruments[intent.instrument]
            if session.off_hours and info.asset_class == "equity":
                window = self.calendar.last_regular_close(now)
                key = (window, intent.sleeve.id)
                need = max(self._off_need.get(key, 0.0), abs(intent.delta))
                self._off_need[key] = need
                allowed = max(
                    intent.sleeve.max_off_hours_pct * need - self._off_done.get(key, 0.0), 0.0
                )
                if abs(intent.delta) > allowed:
                    intent.delta = math.copysign(allowed, intent.delta)
                    intent.reasons.append(
                        f"off-hours cap {intent.sleeve.max_off_hours_pct:.0%} of ${need:,.0f} needed"
                    )
            if abs(intent.delta) > max_order:
                intent.delta = math.copysign(max_order, intent.delta)
                intent.reasons.append(f"capped at {limits.max_order_pct:.0%} single-order limit")

        # Buys only spend cash that is already there; sale proceeds wait for the fill.
        cash = marks[self.model.cash.id].value
        spendable = max(cash - self.model.cash_floor * value, 0.0)
        wanted = sum(i.delta for i in intents if i.delta > 0) * (1 + self.config.fee_buffer)
        if wanted > spendable + _EPS:
            scale = spendable / wanted
            for intent in intents:
                if intent.delta > 0:
                    intent.delta *= scale
                    intent.reasons.append(
                        f"scaled to {scale:.0%} by available cash above the {self.model.cash_floor:.0%} floor"
                    )

        if session.off_hours:
            room, budget = self.gate.turnover_room(now, session, value)
            total = sum(abs(i.delta) for i in intents)
            if total > room + _EPS:
                scale = room / total
                for intent in intents:
                    intent.delta *= scale
                    intent.reasons.append(f"scaled to the remaining {budget} turnover budget")

        kept = []
        for intent in sorted(intents, key=lambda i: i.delta):  # sells first
            if abs(intent.delta) < self.config.min_trade:
                self.audit.decide(
                    now,
                    intent.sleeve.id,
                    "skip",
                    f"trade ${abs(intent.delta):,.2f} under ${self.config.min_trade:,.0f} minimum ({'; '.join(intent.reasons)})",
                )
                continue
            kept.append(intent)
        return kept

    # --- send ------------------------------------------------------------------------------

    def _send(
        self,
        now: datetime,
        session: Session,
        intent: _Intent,
        fresh: dict[str, Quote],
        value: float,
    ) -> OrderRecord | None:
        inst = intent.instrument
        info = self.instruments[inst]
        venue = self.venues[info.venue]
        quote = fresh[inst]
        side = Side.BUY if intent.delta > 0 else Side.SELL
        collar = self.config.limit_collar
        limit = quote.mid * (1 + collar) if side is Side.BUY else quote.mid * (1 - collar)
        qty = abs(intent.delta) / (limit if side is Side.BUY else quote.mid)
        if side is Side.SELL:
            qty = min(qty, self.ledger.get(inst, 0.0))
        qty = round_down(qty, info.lot_size)
        if qty <= 0 or qty * quote.mid < self.config.min_trade:
            self.audit.decide(
                now,
                intent.sleeve.id,
                "skip",
                f"{inst} trade rounds below the minimum at lot size {info.lot_size:g}",
            )
            return None

        status = venue.session_status(inst)
        tif = TimeInForce.DAY if TimeInForce.DAY in status.time_in_force else TimeInForce.GTC
        reason = "; ".join(intent.reasons)
        order = Order(
            client_id=f"{now:%Y%m%dT%H%M%S}-{intent.sleeve.id}",
            instrument=inst,
            side=side,
            qty=qty,
            limit_price=round(limit, 8 if info.asset_class == "crypto" else 2),
            order_type=OrderType.LIMIT,
            time_in_force=tif,
            sleeve=intent.sleeve.id,
            reason=reason,
        )
        result = self.gate.check(
            order,
            now=now,
            quote=quote,
            status=status,
            session=session,
            account_value=value,
            min_trade=self.config.min_trade,
        )
        record = OrderRecord(
            ts=now,
            session=session.value,
            venue=info.venue,
            order_id=None,
            client_id=order.client_id,
            sleeve=order.sleeve,
            instrument=inst,
            side=side.value,
            qty=order.qty,
            order_type=order.order_type.value,
            time_in_force=tif.value,
            limit_price=order.limit_price,
            reference_price=quote.mid,
            notional=order.qty * order.limit_price,
            reason=reason,
            status="rejected",
        )
        if not result.approved:
            record.reject_reason = f"risk gate: {result.reason}"
            self.audit.orders.append(record)
            self.audit.decide(now, intent.sleeve.id, "reject", record.reject_reason)
            return record

        order = result.order
        if result.reason:
            order.reason = f"{order.reason}; {result.reason}"
        record.qty, record.notional, record.reason = (
            order.qty,
            order.qty * order.limit_price,
            order.reason,
        )
        ack = venue.place(order)
        record.order_id = ack.order_id
        if not ack.accepted:
            record.reject_reason = f"venue: {ack.reason}"
            self.audit.orders.append(record)
            self.gate.record_venue_reject(now, info.venue, ack.reason)
            self.audit.decide(now, intent.sleeve.id, "reject", record.reject_reason, dedupe=False)
            return record

        record.status = "placed"
        self.audit.orders.append(record)
        self._open[order.client_id] = (info.venue, record)
        self.gate.record_placed(order.client_id, now, session, record.notional)
        if session.off_hours and info.asset_class == "equity":
            key = (self.calendar.last_regular_close(now), intent.sleeve.id)
            self._off_done[key] = self._off_done.get(key, 0.0) + record.notional
            self._off_orders[order.client_id] = (key, record.notional)
        self.audit.decide(
            now,
            intent.sleeve.id,
            "trade",
            f"{side} {order.qty:g} {inst} limit {order.limit_price:g} in {session}: {order.reason}",
            dedupe=False,
        )
        return record

    def _release_off_hours(self, client_id: str) -> None:
        entry = self._off_orders.pop(client_id, None)
        if entry is not None:
            key, notional = entry
            self._off_done[key] -= notional
