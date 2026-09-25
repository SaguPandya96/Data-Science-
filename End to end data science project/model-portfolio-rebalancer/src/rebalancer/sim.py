"""Simulated venues for replays and tests. No network, no real accounts."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime

from .sessions import MarketCalendar, Session
from .venue import Fill, Order, OrderAck, OrderType, Quote, SessionStatus, Side, TimeInForce

EQUITY_RULES = {
    Session.REGULAR: (
        {OrderType.LIMIT, OrderType.MARKET},
        {TimeInForce.DAY, TimeInForce.GTC, TimeInForce.IOC},
    ),
    Session.PRE: ({OrderType.LIMIT}, {TimeInForce.DAY}),
    Session.POST: ({OrderType.LIMIT}, {TimeInForce.DAY}),
    Session.OVERNIGHT: ({OrderType.LIMIT}, {TimeInForce.DAY, TimeInForce.GTC}),
}
CRYPTO_RULES = (
    {OrderType.LIMIT, OrderType.MARKET},
    {TimeInForce.DAY, TimeInForce.GTC, TimeInForce.IOC},
)

# Half-spreads in basis points by equity session. Overnight books are thin; crypto widens a bit on
# weekends. Rough numbers, only there so limit orders have something to cross.
EQUITY_HALF_SPREAD_BPS = {
    Session.REGULAR: 1.0,
    Session.PRE: 5.0,
    Session.POST: 5.0,
    Session.OVERNIGHT: 10.0,
}
CRYPTO_HALF_SPREAD_BPS = {"weekday": 3.0, "weekend": 8.0}


@dataclass
class SimClock:
    now: datetime


@dataclass
class SimAccount:
    """One cash balance shared by every venue, as if equities and crypto sat with one broker."""

    cash: float
    holdings: dict[str, float] = field(default_factory=dict)


@dataclass
class _Resting:
    order: Order
    order_id: str


class SimVenue:
    def __init__(
        self,
        name: str,
        instruments: set[str],
        clock: SimClock,
        account: SimAccount,
        calendar: MarketCalendar,
        *,
        always_open: bool = False,
        fee_bps: float = 0.0,
        reports_cash: bool = False,
    ):
        self.name = name
        self.instruments = set(instruments)
        self.clock = clock
        self.account = account
        self.calendar = calendar
        self.always_open = always_open
        self.fee_bps = fee_bps
        self.reports_cash = reports_cash
        self._mids: dict[str, tuple[float, datetime]] = {}
        self._resting: dict[str, _Resting] = {}
        self._fills: list[Fill] = []
        self._ids = itertools.count(1)
        self._forced_rejects: list[str] = []

    # --- test and replay hooks -------------------------------------------------------------

    def set_price(self, instrument: str, mid: float, ts: datetime) -> None:
        self._require(instrument)
        self._mids[instrument] = (mid, ts)
        for order_id, resting in list(self._resting.items()):
            if resting.order.instrument == instrument:
                self._try_fill(order_id, resting.order)

    def reject_next(self, count: int, reason: str = "rejected by venue") -> None:
        self._forced_rejects.extend([reason] * count)

    # --- adapter interface -----------------------------------------------------------------

    def session_status(self, instrument: str) -> SessionStatus:
        self._require(instrument)
        if self.always_open:
            types, tifs = CRYPTO_RULES
            return SessionStatus(True, "continuous", frozenset(types), frozenset(tifs))
        session = self.calendar.session_at(self.clock.now).session
        if session not in EQUITY_RULES:
            return SessionStatus(False, session.value)
        types, tifs = EQUITY_RULES[session]
        return SessionStatus(True, session.value, frozenset(types), frozenset(tifs))

    def quote(self, instrument: str) -> Quote | None:
        self._require(instrument)
        if instrument not in self._mids:
            return None
        mid, ts = self._mids[instrument]
        half = self._half_spread_bps(ts) / 10_000
        return Quote(instrument, mid * (1 - half), mid * (1 + half), 1e9, 1e9, ts)

    def place(self, order: Order) -> OrderAck:
        self._require(order.instrument)
        if self._forced_rejects:
            return OrderAck(None, False, self._forced_rejects.pop(0))
        status = self.session_status(order.instrument)
        if not status.is_open:
            return OrderAck(None, False, f"market closed ({status.session})")
        if (
            order.order_type not in status.order_types
            or order.time_in_force not in status.time_in_force
        ):
            return OrderAck(
                None,
                False,
                f"{order.order_type}/{order.time_in_force} not accepted in {status.session}",
            )
        if order.qty <= 0:
            return OrderAck(None, False, "quantity must be positive")
        if order.order_type is OrderType.LIMIT and (
            order.limit_price is None or order.limit_price <= 0
        ):
            return OrderAck(None, False, "limit price required")
        quote = self.quote(order.instrument)
        if quote is None:
            return OrderAck(None, False, "no quote")
        if order.side is Side.BUY:
            price = order.limit_price if order.order_type is OrderType.LIMIT else quote.ask
            if order.qty * price * (1 + self.fee_bps / 10_000) > self.account.cash + 1e-9:
                return OrderAck(None, False, "insufficient cash")
        elif order.qty > self.account.holdings.get(order.instrument, 0.0) + 1e-12:
            return OrderAck(None, False, "insufficient position")

        order_id = f"{self.name}-{next(self._ids)}"
        order.venue_id = order_id
        if not self._try_fill(order_id, order):
            if order.time_in_force is TimeInForce.IOC:
                return OrderAck(order_id, True, "expired unfilled")
            self._resting[order_id] = _Resting(order, order_id)
        return OrderAck(order_id, True)

    def cancel(self, order_id: str) -> bool:
        return self._resting.pop(order_id, None) is not None

    def positions(self) -> dict[str, float]:
        held = {
            k: v for k, v in self.account.holdings.items() if k in self.instruments and abs(v) > 0
        }
        if self.reports_cash:
            held["USD"] = self.account.cash
        return held

    def fills(self, since: datetime) -> list[Fill]:
        return [f for f in self._fills if f.ts >= since]

    # --- internals -------------------------------------------------------------------------

    def _try_fill(self, order_id: str, order: Order) -> bool:
        quote = self.quote(order.instrument)
        if quote is None or not self.session_status(order.instrument).is_open:
            return False
        if order.side is Side.BUY:
            price = quote.ask
            if order.order_type is OrderType.LIMIT and price > order.limit_price:
                return False
        else:
            price = quote.bid
            if order.order_type is OrderType.LIMIT and price < order.limit_price:
                return False
        fee = order.qty * price * self.fee_bps / 10_000
        sign = 1 if order.side is Side.BUY else -1
        self.account.cash -= sign * order.qty * price + fee
        self.account.holdings[order.instrument] = (
            self.account.holdings.get(order.instrument, 0.0) + sign * order.qty
        )
        self._fills.append(
            Fill(
                f"{order_id}-f",
                order_id,
                order.instrument,
                order.side,
                order.qty,
                price,
                fee,
                self.clock.now,
            )
        )
        self._resting.pop(order_id, None)
        return True

    def _half_spread_bps(self, ts: datetime) -> float:
        session = self.calendar.session_at(ts).session
        if self.always_open:
            return CRYPTO_HALF_SPREAD_BPS["weekend" if session is Session.WEEKEND else "weekday"]
        return EQUITY_HALF_SPREAD_BPS.get(session, EQUITY_HALF_SPREAD_BPS[Session.OVERNIGHT])

    def _require(self, instrument: str) -> None:
        if instrument not in self.instruments:
            raise KeyError(f"{self.name} does not trade {instrument}")
