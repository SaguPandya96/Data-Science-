"""The contract every venue adapter implements, and the types that cross it."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal, Protocol


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    LIMIT = "limit"
    MARKET = "market"


class TimeInForce(StrEnum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"


@dataclass(frozen=True)
class InstrumentInfo:
    venue: str
    asset_class: Literal["equity", "crypto"]
    lot_size: float
    broker_symbol: str | None = None  # when the broker spells it differently, e.g. BTC/USD


@dataclass(frozen=True)
class SessionStatus:
    is_open: bool
    session: str
    order_types: frozenset[OrderType] = frozenset()
    time_in_force: frozenset[TimeInForce] = frozenset()


@dataclass(frozen=True)
class Quote:
    instrument: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    ts: datetime

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    def age(self, now: datetime) -> timedelta:
        return now - self.ts


@dataclass
class Order:
    client_id: str
    instrument: str
    side: Side
    qty: float
    limit_price: float | None
    order_type: OrderType = OrderType.LIMIT
    time_in_force: TimeInForce = TimeInForce.DAY
    sleeve: str = ""
    reason: str = ""
    venue_id: str | None = field(default=None, compare=False)

    def notional(self, reference: float) -> float:
        return self.qty * (self.limit_price if self.limit_price is not None else reference)


@dataclass(frozen=True)
class OrderAck:
    order_id: str | None
    accepted: bool
    reason: str = ""


@dataclass(frozen=True)
class Fill:
    fill_id: str
    order_id: str
    instrument: str
    side: Side
    qty: float
    price: float
    fee: float
    ts: datetime


@dataclass(frozen=True)
class OrderState:
    status: Literal["open", "filled", "canceled", "expired", "rejected"]
    filled_qty: float


class Venue(Protocol):
    name: str

    def session_status(self, instrument: str) -> SessionStatus: ...

    def quote(self, instrument: str) -> Quote | None: ...

    def place(self, order: Order) -> OrderAck: ...

    def cancel(self, order_id: str) -> bool: ...

    def positions(self) -> dict[str, float]: ...

    def fills(self, since: datetime) -> list[Fill]: ...

    def find_order(self, client_id: str) -> str | None:
        """The broker's id for an order sent with this client id, or None if it never arrived."""
        ...

    def order_state(self, order_id: str) -> OrderState | None:
        """Where an order stands at the broker, or None if the broker doesn't know it."""
        ...

    def open_orders(self) -> dict[str, str]:
        """Every order resting at the broker, as broker id to client id."""
        ...
