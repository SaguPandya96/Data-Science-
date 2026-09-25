"""Hard limits every order passes before it reaches a venue.

The gate is the last word: the engine tries to size orders inside these limits, but anything that
slips through is rejected here, and some breaches halt trading rather than just logging.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta

from .audit import AuditLog
from .sessions import ET, MarketCalendar, Session
from .venue import InstrumentInfo, Order, OrderType, Quote, SessionStatus, Side


@dataclass(frozen=True)
class RiskLimits:
    max_order_pct: float = 0.05
    max_daily_turnover_pct: float = 0.15
    max_off_hours_turnover_pct: float = 0.05
    max_limit_deviation: float = 0.02
    max_quote_age: timedelta = timedelta(seconds=60)
    crypto_drawdown: float = 0.15
    crypto_drawdown_window: timedelta = timedelta(hours=24)
    drawdown_watch: tuple[str, ...] = ("BTC-USD", "ETH-USD")
    max_reconciliation_gap: float = 10.0
    max_rejects: int = 3
    reject_window: timedelta = timedelta(minutes=10)


@dataclass(frozen=True)
class GateResult:
    order: Order | None
    reason: str = ""

    @property
    def approved(self) -> bool:
        return self.order is not None


@dataclass
class _Placed:
    day: date
    window: datetime | None
    notional: float


@dataclass
class RiskGate:
    limits: RiskLimits
    calendar: MarketCalendar
    instruments: dict[str, InstrumentInfo]
    audit: AuditLog
    halted: str | None = None
    halted_until: datetime | None = None
    venue_halts: dict[str, str] = field(default_factory=dict)
    daily_turnover: dict[date, float] = field(default_factory=dict)
    off_hours_turnover: dict[datetime, float] = field(default_factory=dict)
    _history: dict[str, deque] = field(default_factory=dict)
    _rejects: dict[str, deque] = field(default_factory=dict)
    _placed: dict[str, _Placed] = field(default_factory=dict)
    _pause_reason: str | None = None

    # --- halts -----------------------------------------------------------------------------

    def halt(self, now: datetime, reason: str) -> None:
        if self.halted is None:
            self.halted = reason
            self.audit.alert(now, "critical", f"trading halted: {reason}")

    def clear_halt(self) -> None:
        self.halted = None

    def halt_reason(self, now: datetime) -> str | None:
        if self.halted:
            return self.halted
        if self.halted_until and now < self.halted_until:
            return f"daily turnover limit reached, halted until {self.halted_until.astimezone(ET):%a %H:%M} ET"
        return None

    def halt_venue(self, now: datetime, venue: str, reason: str) -> None:
        if venue not in self.venue_halts:
            self.venue_halts[venue] = reason
            self.audit.alert(now, "critical", f"venue {venue} halted: {reason}")

    def clear_venue(self, venue: str) -> None:
        self.venue_halts.pop(venue, None)
        self._rejects.pop(venue, None)

    def record_venue_reject(self, now: datetime, venue: str, reason: str) -> None:
        recent = self._rejects.setdefault(venue, deque())
        recent.append(now)
        while recent and now - recent[0] > self.limits.reject_window:
            recent.popleft()
        if len(recent) >= self.limits.max_rejects:
            minutes = int(self.limits.reject_window.total_seconds() // 60)
            self.halt_venue(
                now, venue, f"{len(recent)} rejected orders in {minutes} min (last: {reason})"
            )

    # --- market state ----------------------------------------------------------------------

    def observe(self, instrument: str, ts: datetime, mid: float) -> None:
        if instrument not in self.limits.drawdown_watch:
            return
        hist = self._history.setdefault(instrument, deque())
        if hist and hist[-1][0] >= ts:
            return
        hist.append((ts, mid))
        while hist and ts - hist[0][0] > self.limits.crypto_drawdown_window:
            hist.popleft()

    def drawdown(self, instrument: str) -> float:
        """Fall from the highest price in the trailing window to the latest one."""
        hist = self._history.get(instrument)
        if not hist:
            return 0.0
        high = max(p for _, p in hist)
        return 1 - hist[-1][1] / high

    def crypto_buys_paused(self, now: datetime) -> str | None:
        reason = None
        for inst in self.limits.drawdown_watch:
            dd = self.drawdown(inst)
            if dd >= self.limits.crypto_drawdown:
                hours = int(self.limits.crypto_drawdown_window.total_seconds() // 3600)
                reason = f"{inst} down {dd:.1%} from its {hours}h high"
                break
        if reason and not self._pause_reason:
            self.audit.alert(now, "warning", f"crypto buys paused: {reason}")
        elif self._pause_reason and not reason:
            self.audit.alert(now, "info", "crypto buys resumed: drawdown back inside limit")
        self._pause_reason = reason
        return reason

    # --- the gate --------------------------------------------------------------------------

    def check(
        self,
        order: Order,
        *,
        now: datetime,
        quote: Quote | None,
        status: SessionStatus,
        session: Session,
        account_value: float,
        min_trade: float,
    ) -> GateResult:
        info = self.instruments[order.instrument]
        halt = self.halt_reason(now)
        if halt:
            return GateResult(None, f"halted: {halt}")
        if info.venue in self.venue_halts:
            return GateResult(None, f"venue {info.venue} halted: {self.venue_halts[info.venue]}")
        if not status.is_open:
            return GateResult(None, f"market closed ({status.session})")
        if order.order_type is OrderType.MARKET and session is not Session.REGULAR:
            return GateResult(None, f"market orders not allowed in {session} session")
        if (
            order.order_type not in status.order_types
            or order.time_in_force not in status.time_in_force
        ):
            return GateResult(
                None, f"{order.order_type}/{order.time_in_force} not allowed in {status.session}"
            )
        if quote is None:
            return GateResult(None, "no quote")
        age = quote.age(now)
        if age > self.limits.max_quote_age:
            return GateResult(None, f"stale quote ({age.total_seconds():.0f}s old)")
        if order.order_type is OrderType.LIMIT:
            deviation = abs(order.limit_price / quote.mid - 1)
            if deviation > self.limits.max_limit_deviation:
                return GateResult(
                    None,
                    f"limit {order.limit_price:.2f} is {deviation:.1%} from reference {quote.mid:.2f}",
                )
        if order.side is Side.BUY and info.asset_class == "crypto":
            paused = self.crypto_buys_paused(now)
            if paused:
                return GateResult(None, f"crypto buys paused: {paused}")

        price = order.limit_price if order.limit_price is not None else quote.mid
        notional = order.qty * price
        max_order = self.limits.max_order_pct * account_value
        if notional > max_order + 1e-6:
            return GateResult(
                None,
                f"order ${notional:,.0f} over the {self.limits.max_order_pct:.0%} single-order limit (${max_order:,.0f})",
            )

        room, budget = self.turnover_room(now, session, account_value)
        if notional <= room + 1e-6:
            return GateResult(order)
        qty = round_down(room / price, info.lot_size)
        if qty * price >= min_trade and qty > 0:
            return GateResult(replace(order, qty=qty), f"clipped to the {budget} turnover budget")
        if budget == "daily":
            self._halt_for_day(now)
            return GateResult(None, "daily turnover limit reached")
        return GateResult(None, "off-hours turnover budget used, deferred to the regular session")

    def record_placed(
        self, client_id: str, now: datetime, session: Session, notional: float
    ) -> None:
        day = now.astimezone(ET).date()
        window = self.calendar.last_regular_close(now) if session.off_hours else None
        self.daily_turnover[day] = self.daily_turnover.get(day, 0.0) + notional
        if window is not None:
            self.off_hours_turnover[window] = self.off_hours_turnover.get(window, 0.0) + notional
        self._placed[client_id] = _Placed(day, window, notional)

    def release_unfilled(self, client_id: str) -> None:
        """Give back the turnover an order used if it was cancelled without filling."""
        placed = self._placed.pop(client_id, None)
        if placed is None:
            return
        self.daily_turnover[placed.day] -= placed.notional
        if placed.window is not None:
            self.off_hours_turnover[placed.window] -= placed.notional

    def forget(self, client_id: str) -> None:
        self._placed.pop(client_id, None)

    def turnover_room(self, now: datetime, session: Session, value: float) -> tuple[float, str]:
        """Dollars still allowed today (and tonight, off-hours) and which budget is the tighter one."""
        day = now.astimezone(ET).date()
        room = self.limits.max_daily_turnover_pct * value - self.daily_turnover.get(day, 0.0)
        budget = "daily"
        if session.off_hours:
            window = self.calendar.last_regular_close(now)
            off = self.limits.max_off_hours_turnover_pct * value - self.off_hours_turnover.get(
                window, 0.0
            )
            if off < room:
                room, budget = off, "off-hours"
        return max(room, 0.0), budget

    def _halt_for_day(self, now: datetime) -> None:
        tomorrow = now.astimezone(ET).date() + timedelta(days=1)
        until = datetime.combine(tomorrow, time(0, 0), tzinfo=ET)
        if self.halted_until != until:
            self.halted_until = until
            self.audit.alert(
                now,
                "critical",
                f"daily turnover limit reached; halted until {until:%a %b %d %H:%M} ET",
            )


def round_down(qty: float, lot: float) -> float:
    lots = math.floor(qty / lot + 1e-9)
    return round(lots * lot, 10)
