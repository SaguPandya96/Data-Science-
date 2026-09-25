"""In-memory audit trail: why each sleeve did what it did, every order, every alert."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Decision:
    ts: datetime
    sleeve: str
    action: str  # trade, hold, skip, reject, halt
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
    status: str  # placed, rejected, filled, canceled
    reject_reason: str = ""
    fill_price: float | None = None
    fee: float = 0.0


@dataclass(frozen=True)
class Alert:
    ts: datetime
    level: str
    message: str


@dataclass
class AuditLog:
    decisions: list[Decision] = field(default_factory=list)
    orders: list[OrderRecord] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    _last: dict[str, tuple[str, str]] = field(default_factory=dict, repr=False)

    def decide(
        self, ts: datetime, sleeve: str, action: str, detail: str, *, dedupe: bool = True
    ) -> None:
        # Holds and skips repeat every cycle while nothing changes; only keep the transitions.
        key = (action, _strip_numbers(detail)) if dedupe else None
        if dedupe and self._last.get(sleeve) == key:
            return
        self._last[sleeve] = key
        self.decisions.append(Decision(ts, sleeve, action, detail))

    def alert(self, ts: datetime, level: str, message: str) -> None:
        self.alerts.append(Alert(ts, level, message))

    def rows(self, kind: str) -> list[dict]:
        return [asdict(r) for r in getattr(self, kind)]


def _strip_numbers(text: str) -> str:
    return re.sub(r"-?\$?[\d,]*\.?\d+%?", "#", text)
