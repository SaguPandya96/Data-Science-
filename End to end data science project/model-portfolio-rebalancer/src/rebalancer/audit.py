"""Audit trail: why each sleeve did what it did, every order, every alert.

Everything is kept in memory for replays and tests. Give the log a store (see store.py) and each
entry is also written there as it happens, and the engine reads its halts and restart state back
from it. If a read or write fails, the store is dropped and the error kept in `store_error`; the
engine halts on it rather than trade without a record.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Protocol

from .venue import Fill


@dataclass(frozen=True)
class Decision:
    ts: datetime
    sleeve: str
    action: str  # ok, hold, skip, trade, reject, halt, cash_flow
    detail: str


@dataclass
class OrderRecord:
    ts: datetime
    session: str
    venue: str
    order_id: str | None
    client_id: str
    sleeve: str
    instrument: str
    side: str
    qty: float
    order_type: str
    time_in_force: str
    limit_price: float | None
    reference_price: float
    notional: float
    reason: str
    status: str  # pending, placed, partially_filled, filled, canceled, rejected
    reject_reason: str = ""
    fill_price: float | None = None  # average over all fills so far
    fee: float = 0.0
    filled_qty: float = 0.0
    off_hours_window: datetime | None = None  # last regular close, for orders placed after it


@dataclass(frozen=True)
class Alert:
    ts: datetime
    level: str
    message: str


@dataclass(frozen=True)
class PriceSeen:
    instrument: str
    mid: float
    quote_ts: datetime
    fresh: bool


@dataclass(frozen=True)
class MarkSeen:
    sleeve: str
    value: float
    weight: float
    stale: bool


@dataclass(frozen=True)
class ReconciliationLine:
    instrument: str
    engine_qty: float
    broker_qty: float
    price: float | None
    gap_usd: float


@dataclass(frozen=True)
class HaltRow:
    scope: str  # global, daily or lane
    lane: str | None
    reason: str
    until: datetime | None = None


@dataclass
class RecoveredState:
    """What a previous run left in the store, for the engine to resume from."""

    last_cycle: datetime | None = None
    open_orders: list[OrderRecord] = field(default_factory=list)
    daily_turnover: dict[date, float] = field(default_factory=dict)
    off_hours_turnover: dict[datetime, float] = field(default_factory=dict)
    off_hours_done: dict[tuple[datetime, str], float] = field(default_factory=dict)
    off_hours_needs: dict[tuple[datetime, str], float] = field(default_factory=dict)
    halts: list[HaltRow] = field(default_factory=list)
    prices: list[tuple[str, datetime, float]] = field(default_factory=list)
    seen_fills: set[str] = field(default_factory=set)  # fill ids already recorded
    recent_sales: list[tuple[datetime, str, float]] = field(default_factory=list)  # ts, class, net


class AuditStore(Protocol):
    def cycle(
        self,
        ts: datetime,
        session: str,
        account_value: float | None,
        halted: str | None,
        prices: list[PriceSeen],
        marks: list[MarkSeen],
    ) -> None: ...

    def decision(self, ts: datetime, sleeve: str, action: str, detail: str) -> None: ...

    def order_created(self, record: OrderRecord) -> None: ...

    def order_updated(self, record: OrderRecord, ts: datetime, detail: str) -> None: ...

    def fill(self, fill: Fill, client_id: str | None) -> None: ...

    def alert(self, ts: datetime, level: str, message: str) -> None: ...

    def reconciliation(
        self, ts: datetime, gap_usd: float, ok: bool, lines: list[ReconciliationLine]
    ) -> None: ...

    def halt_started(
        self, ts: datetime, scope: str, lane: str | None, reason: str, until: datetime | None
    ) -> None: ...

    def halt_cleared(self, ts: datetime, scope: str, lane: str | None) -> None: ...

    def cash_flow(self, ts: datetime, amount: float) -> None: ...

    def off_hours_need(self, window: datetime, sleeve: str, need: float) -> None: ...

    def open_halts(self, now: datetime) -> list[HaltRow]: ...

    def recover(self, now: datetime) -> RecoveredState: ...


@dataclass
class AuditLog:
    store: AuditStore | None = None
    decisions: list[Decision] = field(default_factory=list)
    orders: list[OrderRecord] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    store_error: str | None = None
    _last: dict[str, tuple[str, str]] = field(default_factory=dict, repr=False)

    def cycle(self, ts, session, account_value, halted, prices, marks) -> None:
        self._call("cycle", ts, session, account_value, halted, prices, marks)

    def decide(
        self, ts: datetime, sleeve: str, action: str, detail: str, *, dedupe: bool = True
    ) -> None:
        # Holds and skips repeat every cycle while nothing changes; only keep the transitions.
        key = (action, _strip_numbers(detail)) if dedupe else None
        if dedupe and self._last.get(sleeve) == key:
            return
        self._last[sleeve] = key
        self.decisions.append(Decision(ts, sleeve, action, detail))
        self._call("decision", ts, sleeve, action, detail)

    def order_created(self, record: OrderRecord) -> None:
        self.orders.append(record)
        self._call("order_created", record)

    def order_updated(self, record: OrderRecord, ts: datetime, detail: str = "") -> None:
        self._call("order_updated", record, ts, detail)

    def fill(self, fill: Fill, client_id: str | None) -> None:
        self._call("fill", fill, client_id)

    def alert(self, ts: datetime, level: str, message: str) -> None:
        self.alerts.append(Alert(ts, level, message))
        self._call("alert", ts, level, message)

    def reconciliation(self, ts, gap_usd, ok, lines) -> None:
        self._call("reconciliation", ts, gap_usd, ok, lines)

    def halt_started(self, ts, scope, lane, reason, until=None) -> None:
        self._call("halt_started", ts, scope, lane, reason, until)

    def halt_cleared(self, ts, scope, lane=None) -> None:
        self._call("halt_cleared", ts, scope, lane)

    def cash_flow(self, ts: datetime, amount: float) -> None:
        kind = "deposit" if amount > 0 else "withdrawal"
        self.decisions.append(Decision(ts, "cash", "cash_flow", f"{kind} of ${abs(amount):,.2f}"))
        self._call("cash_flow", ts, amount)

    def off_hours_need(self, window: datetime, sleeve: str, need: float) -> None:
        self._call("off_hours_need", window, sleeve, need)

    def open_halts(self, now: datetime) -> list[HaltRow] | None:
        """Halts in force according to the store, or None when there is no store."""
        return self._call("open_halts", now)

    def recover(self, now: datetime) -> RecoveredState | None:
        return self._call("recover", now)

    def rows(self, kind: str) -> list[dict]:
        return [asdict(r) for r in getattr(self, kind)]

    def _call(self, method: str, *args):
        if self.store is None:
            return None
        try:
            return getattr(self.store, method)(*args)
        except Exception as exc:  # any failure at all means the trail is incomplete
            self.store = None
            self.store_error = f"{method} failed: {type(exc).__name__}: {exc}".strip()
            ts = args[0] if args and isinstance(args[0], datetime) else datetime.now().astimezone()
            self.alerts.append(Alert(ts, "critical", f"audit store failed, {self.store_error}"))


def _strip_numbers(text: str) -> str:
    return re.sub(r"-?\$?[\d,]*\.?\d+%?", "#", text)
