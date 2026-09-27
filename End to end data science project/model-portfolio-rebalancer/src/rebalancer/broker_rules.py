"""What Alpaca accepts in each session. Shared by the real adapter and the simulated one, so a
replay can't allow an order the broker would refuse."""

from __future__ import annotations

from datetime import datetime

from .sessions import MarketCalendar, Session
from .venue import OrderType, SessionStatus, TimeInForce

EQUITY_RULES = {
    Session.REGULAR: (
        {OrderType.LIMIT, OrderType.MARKET},
        {TimeInForce.DAY, TimeInForce.GTC, TimeInForce.IOC},
    ),
    Session.PRE: ({OrderType.LIMIT}, {TimeInForce.DAY}),
    Session.POST: ({OrderType.LIMIT}, {TimeInForce.DAY}),
    Session.OVERNIGHT: ({OrderType.LIMIT}, {TimeInForce.DAY, TimeInForce.GTC}),
}
# Alpaca crypto takes GTC and IOC only, no DAY orders.
CRYPTO_RULES = ({OrderType.LIMIT, OrderType.MARKET}, {TimeInForce.GTC, TimeInForce.IOC})


def session_status(calendar: MarketCalendar, now: datetime, asset_class: str) -> SessionStatus:
    if asset_class == "crypto":
        types, tifs = CRYPTO_RULES
        return SessionStatus(True, "continuous", frozenset(types), frozenset(tifs))
    session = calendar.session_at(now).session
    if session not in EQUITY_RULES:
        return SessionStatus(False, session.value)
    types, tifs = EQUITY_RULES[session]
    return SessionStatus(True, session.value, frozenset(types), frozenset(tifs))
