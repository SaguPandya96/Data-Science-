"""Unsettled sale proceeds in a cash account.

Equity sales settle one trading day after the trade date; until then the proceeds are in the
cash balance but not spendable. Buying with them is allowed in a cash account, but selling what
they bought before they settle is a good-faith violation, so the engine simply doesn't spend
them until they settle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time

from .sessions import ET, MarketCalendar

# Trading days from trade to settlement. Crypto proceeds are treated as available at once;
# check the broker's rule before relying on that.
DEFAULT_SETTLEMENT_DAYS = {"equity": 1, "crypto": 0}


@dataclass(frozen=True)
class Proceeds:
    amount: float
    settles_at: datetime


@dataclass
class SettlementBook:
    calendar: MarketCalendar
    days: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SETTLEMENT_DAYS))
    _pending: list[Proceeds] = field(default_factory=list)

    def record_sale(self, ts: datetime, asset_class: str, amount: float) -> None:
        days = self.days.get(asset_class, 0)
        if days <= 0 or amount <= 0:
            return
        trade_date = self.calendar.trade_date(ts) or ts.astimezone(ET).date()
        settles = self.calendar.settlement_date(trade_date, days)
        # Settled funds are usable from the start of the settlement date.
        self._pending.append(Proceeds(amount, datetime.combine(settles, time(0, 0), tzinfo=ET)))

    def unsettled(self, now: datetime) -> float:
        self._pending = [p for p in self._pending if p.settles_at > now]
        return sum(p.amount for p in self._pending)

    def next_settlement(self, now: datetime) -> datetime | None:
        upcoming = [p.settles_at for p in self._pending if p.settles_at > now]
        return min(upcoming) if upcoming else None
