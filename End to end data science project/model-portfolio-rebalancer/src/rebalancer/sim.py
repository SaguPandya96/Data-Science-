"""A simulated broker for replays and tests. No network, no real accounts.

It stands in for Alpaca, which holds equities and crypto in one account against one cash
balance. Equities follow the session calendar; crypto trades around the clock.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from datetime import datetime

from . import broker_rules
from .sessions import MarketCalendar, Session
from .settlement import SettlementBook
from .venue import (
    Fill,
    InstrumentInfo,
    Order,
    OrderAck,
    OrderType,
    Quote,
    SessionStatus,
    Side,
    TimeInForce,
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
    """The broker account: one cash balance behind both equities and crypto. With a settlement
    book it's a cash account, and buys can only use settled cash."""

    cash: float
    holdings: dict[str, float] = field(default_factory=dict)
    settlement: SettlementBook | None = None

    def settled_cash(self, now: datetime) -> float:
        return self.cash - (self.settlement.unsettled(now) if self.settlement else 0.0)


@dataclass
class _Resting:
    order: Order
    order_id: str
    remaining: float
    fills: int = 0


class SimVenue:
    def __init__(
        self,
        name: str,
        instruments: dict[str, InstrumentInfo],
        clock: SimClock,
        account: SimAccount,
        calendar: MarketCalendar,
        *,
        fee_bps: dict[str, float] | None = None,
        id_prefix: str = "",
    ):
        self.name = name
        self.instruments = dict(instruments)
        self.clock = clock
        self.account = account
        self.calendar = calendar
        self.fee_bps = fee_bps or {}
        self._mids: dict[str, tuple[float, datetime]] = {}
        self._resting: dict[str, _Resting] = {}
        self._fills: list[Fill] = []
        self._ids = itertools.count(1)
        # Real broker ids are unique forever. A prefix keeps simulated ones from colliding when
        # several replays write to the same database.
        self.id_prefix = id_prefix
        self._by_client: dict[str, str] = {}
        # Test hooks: while fills_paused is set nothing crosses, as if the book were empty.
        # set_depth caps how much trades at the touch between price updates, for partial fills.
        self.fills_paused = False
        self._depth: dict[str, float] = {}
        self._available: dict[str, float] = {}
        self._forced_rejects: list[tuple[str, str | None]] = []

    # --- test and replay hooks -------------------------------------------------------------

    def set_price(self, instrument: str, mid: float, ts: datetime) -> None:
        self._require(instrument)
        self._mids[instrument] = (mid, ts)
        self._available[instrument] = self._depth.get(instrument, math.inf)
        for resting in list(self._resting.values()):
            if resting.order.instrument == instrument:
                self._try_fill(resting)

    def set_depth(self, instrument: str, qty: float | None) -> None:
        """Let only `qty` trade at the touch per price update; None puts no limit on it."""
        if qty is None:
            self._depth.pop(instrument, None)
        else:
            self._depth[instrument] = qty
        self._available[instrument] = math.inf if qty is None else qty

    def reject_next(
        self, count: int, reason: str = "rejected by venue", asset_class: str | None = None
    ) -> None:
        """Reject the next `count` orders, or only those for one asset class."""
        self._forced_rejects.extend([(reason, asset_class)] * count)

    # --- adapter interface -----------------------------------------------------------------

    def session_status(self, instrument: str) -> SessionStatus:
        self._require(instrument)
        asset_class = self.instruments[instrument].asset_class
        return broker_rules.session_status(self.calendar, self.clock.now, asset_class)

    def quote(self, instrument: str) -> Quote | None:
        self._require(instrument)
        if instrument not in self._mids:
            return None
        mid, ts = self._mids[instrument]
        half = self._half_spread_bps(instrument, ts) / 10_000
        return Quote(instrument, mid * (1 - half), mid * (1 + half), 1e9, 1e9, ts)

    def place(self, order: Order) -> OrderAck:
        self._require(order.instrument)
        asset_class = self.instruments[order.instrument].asset_class
        for i, (reason, only) in enumerate(self._forced_rejects):
            if only is None or only == asset_class:
                del self._forced_rejects[i]
                return OrderAck(None, False, reason)
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
            cost = order.qty * price * (1 + self._fee(order.instrument))
            if cost > self.account.cash + 1e-9:
                return OrderAck(None, False, "insufficient cash")
            if cost > self.account.settled_cash(self.clock.now) + 1e-9:
                return OrderAck(None, False, "insufficient settled cash")
        elif order.qty > self.account.holdings.get(order.instrument, 0.0) + 1e-12:
            return OrderAck(None, False, "insufficient position")

        order_id = f"{self.name}-{self.id_prefix}{next(self._ids)}"
        order.venue_id = order_id
        self._by_client[order.client_id] = order_id
        resting = _Resting(order, order_id, order.qty)
        self._try_fill(resting)
        if resting.remaining > 0:
            if order.time_in_force is TimeInForce.IOC:
                return OrderAck(order_id, True, "rest expired unfilled")
            self._resting[order_id] = resting
        return OrderAck(order_id, True)

    def find_order(self, client_id: str) -> str | None:
        return self._by_client.get(client_id)

    def cancel(self, order_id: str) -> bool:
        return self._resting.pop(order_id, None) is not None

    def positions(self) -> dict[str, float]:
        held = {
            k: v for k, v in self.account.holdings.items() if k in self.instruments and abs(v) > 0
        }
        held["USD"] = self.account.cash
        return held

    def fills(self, since: datetime) -> list[Fill]:
        return [f for f in self._fills if f.ts >= since]

    # --- internals -------------------------------------------------------------------------

    def _try_fill(self, resting: _Resting) -> bool:
        order = resting.order
        quote = self.quote(order.instrument)
        if self.fills_paused or quote is None or not self.session_status(order.instrument).is_open:
            return False
        if order.side is Side.BUY:
            price = quote.ask
            if order.order_type is OrderType.LIMIT and price > order.limit_price:
                return False
        else:
            price = quote.bid
            if order.order_type is OrderType.LIMIT and price < order.limit_price:
                return False
        available = self._available.get(order.instrument, math.inf)
        qty = min(resting.remaining, available)
        if qty <= 0:
            return False
        self._available[order.instrument] = available - qty
        resting.remaining = round(resting.remaining - qty, 12)
        resting.fills += 1
        fee = qty * price * self._fee(order.instrument)
        sign = 1 if order.side is Side.BUY else -1
        self.account.cash -= sign * qty * price + fee
        self.account.holdings[order.instrument] = (
            self.account.holdings.get(order.instrument, 0.0) + sign * qty
        )
        if order.side is Side.SELL and self.account.settlement is not None:
            asset_class = self.instruments[order.instrument].asset_class
            self.account.settlement.record_sale(self.clock.now, asset_class, qty * price - fee)
        self._fills.append(
            Fill(
                f"{resting.order_id}-f{resting.fills}",
                resting.order_id,
                order.instrument,
                order.side,
                qty,
                price,
                fee,
                self.clock.now,
            )
        )
        if resting.remaining <= 0:
            self._resting.pop(resting.order_id, None)
        return True

    def _is_crypto(self, instrument: str) -> bool:
        return self.instruments[instrument].asset_class == "crypto"

    def _fee(self, instrument: str) -> float:
        return self.fee_bps.get(self.instruments[instrument].asset_class, 0.0) / 10_000

    def _half_spread_bps(self, instrument: str, ts: datetime) -> float:
        session = self.calendar.session_at(ts).session
        if self._is_crypto(instrument):
            return CRYPTO_HALF_SPREAD_BPS["weekend" if session is Session.WEEKEND else "weekday"]
        return EQUITY_HALF_SPREAD_BPS.get(session, EQUITY_HALF_SPREAD_BPS[Session.OVERNIGHT])

    def _require(self, instrument: str) -> None:
        if instrument not in self.instruments:
            raise KeyError(f"{self.name} does not trade {instrument}")
