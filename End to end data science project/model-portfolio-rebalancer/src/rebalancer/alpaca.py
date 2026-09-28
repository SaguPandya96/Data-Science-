"""Alpaca Trading API adapter, paper account by default.

    export APCA_API_KEY_ID=... APCA_API_SECRET_KEY=...
    python -m rebalancer.alpaca check              # read-only look at the paper account
    python -m rebalancer.alpaca check --order-test # also place and cancel a far-away limit order
    python -m rebalancer.alpaca cash-only          # stop the account from ever trading on margin

Keys only ever come from the environment. The adapter refuses any trading URL other than the
paper one unless it is built with allow_live=True, which nothing in this project does.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx

from . import broker_rules
from .sessions import MarketCalendar, Session
from .venue import (
    Fill,
    InstrumentInfo,
    Order,
    OrderAck,
    OrderState,
    OrderType,
    Quote,
    SessionStatus,
    Side,
)

PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"
PAGE_SIZE = 100
# Alpaca has a dozen order statuses. These are the ones where the order is finished; everything
# else (new, accepted, pending_cancel, held and so on) may still trade.
_FINISHED = frozenset({"filled", "canceled", "expired", "rejected"})


class AlpacaError(RuntimeError):
    pass


@dataclass(frozen=True)
class AlpacaConfig:
    key_id: str
    secret_key: str
    trading_url: str = PAPER_URL
    data_url: str = DATA_URL
    # IEX is the free stock feed. It has no overnight quotes, so equities read as stale overnight
    # and the engine won't trade them then; a feed that covers the overnight session fixes that.
    stock_feed: str = "iex"
    timeout: float = 10.0

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> AlpacaConfig:
        env = os.environ if env is None else env
        missing = [k for k in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY") if not env.get(k)]
        if missing:
            raise AlpacaError(f"set {' and '.join(missing)} (paper account keys)")
        return cls(
            key_id=env["APCA_API_KEY_ID"],
            secret_key=env["APCA_API_SECRET_KEY"],
            trading_url=env.get("APCA_API_BASE_URL", PAPER_URL).rstrip("/"),
            stock_feed=env.get("ALPACA_STOCK_FEED", "iex"),
        )


class AlpacaVenue:
    name = "alpaca"

    def __init__(
        self,
        config: AlpacaConfig,
        instruments: dict[str, InstrumentInfo],
        calendar: MarketCalendar,
        *,
        client: httpx.Client | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        allow_live: bool = False,
    ):
        if config.trading_url != PAPER_URL and not allow_live:
            raise AlpacaError(f"refusing to trade against {config.trading_url}: paper only")
        self.config = config
        self.instruments = {k: v for k, v in instruments.items() if v.venue == self.name}
        self.calendar = calendar
        self.clock = clock
        self._http = client or httpx.Client(timeout=config.timeout)
        self._headers = {
            "APCA-API-KEY-ID": config.key_id,
            "APCA-API-SECRET-KEY": config.secret_key,
        }
        self._to_broker = {s: info.broker_symbol or s for s, info in self.instruments.items()}
        # Orders take BTC/USD but positions and fills report BTCUSD, so accept both.
        self._from_broker: dict[str, str] = {}
        for symbol, broker in self._to_broker.items():
            self._from_broker[broker] = symbol
            self._from_broker[broker.replace("/", "")] = symbol
        self.last_quote_error: str | None = None

    # --- adapter interface -----------------------------------------------------------------

    def session_status(self, instrument: str) -> SessionStatus:
        info = self._info(instrument)
        return broker_rules.session_status(self.calendar, self.clock(), info.asset_class)

    def quote(self, instrument: str) -> Quote | None:
        info = self._info(instrument)
        symbol = self._to_broker[instrument]
        try:
            if info.asset_class == "crypto":
                body = self._get(
                    f"{self.config.data_url}/v1beta3/crypto/us/latest/quotes",
                    params={"symbols": symbol},
                )
                raw = body.get("quotes", {}).get(symbol)
            else:
                body = self._get(
                    f"{self.config.data_url}/v2/stocks/{symbol}/quotes/latest",
                    params={"feed": self.config.stock_feed},
                )
                raw = body.get("quote")
        except (httpx.HTTPError, AlpacaError) as exc:
            # A missing quote only means the sleeve waits a cycle; it shouldn't stop the engine.
            self.last_quote_error = f"{instrument}: {exc}"
            return None
        if not raw or not raw.get("bp") or not raw.get("ap"):
            return None
        return Quote(
            instrument,
            float(raw["bp"]),
            float(raw["ap"]),
            float(raw.get("bs") or 0),
            float(raw.get("as") or 0),
            parse_time(raw["t"]),
        )

    def place(self, order: Order) -> OrderAck:
        info = self._info(order.instrument)
        body = {
            "symbol": self._to_broker[order.instrument],
            "qty": _decimal(order.qty),
            "side": order.side.value,
            "type": order.order_type.value,
            "time_in_force": order.time_in_force.value,
            "client_order_id": order.client_id,
        }
        if order.order_type is OrderType.LIMIT:
            body["limit_price"] = _price(order.limit_price, info.asset_class)
        if info.asset_class == "equity":
            session = self.calendar.session_at(self.clock()).session
            if session is not Session.REGULAR:
                # Pre-market, after-hours and the overnight session all need this flag.
                body["extended_hours"] = True
        try:
            response = self._http.post(
                f"{self.config.trading_url}/v2/orders", json=body, headers=self._headers
            )
        except httpx.TransportError as exc:
            # The order may or may not have arrived. Ask rather than guess.
            broker_id = self.find_order(order.client_id)
            if broker_id is not None:
                return OrderAck(broker_id, True, f"confirmed by lookup after {type(exc).__name__}")
            return OrderAck(None, False, f"not received ({type(exc).__name__})")
        if response.status_code in (200, 201):
            return OrderAck(response.json()["id"], True)
        if 400 <= response.status_code < 500:
            return OrderAck(None, False, f"{response.status_code}: {_message(response)}")
        raise AlpacaError(f"order {order.client_id}: {response.status_code} {_message(response)}")

    def cancel(self, order_id: str) -> bool:
        response = self._http.delete(
            f"{self.config.trading_url}/v2/orders/{order_id}", headers=self._headers
        )
        if response.status_code in (200, 204):
            return True
        # 404 or 422: already filled, cancelled or expired. The fills will say which.
        if response.status_code in (404, 422):
            return False
        raise AlpacaError(f"cancel {order_id}: {response.status_code} {_message(response)}")

    def positions(self) -> dict[str, float]:
        held: dict[str, float] = {}
        for row in self._get(f"{self.config.trading_url}/v2/positions"):
            # Anything this project didn't buy is reported under its own symbol, so
            # reconciliation flags it instead of it being silently ignored.
            symbol = self._from_broker.get(row["symbol"], row["symbol"])
            held[symbol] = held.get(symbol, 0.0) + float(row["qty"])
        held["USD"] = float(self.account()["cash"])
        return held

    def fills(self, since: datetime) -> list[Fill]:
        # A second of overlap: the engine ignores fills it has already seen.
        after = (since - timedelta(seconds=1)).astimezone(UTC).isoformat()
        out: list[Fill] = []
        token = None
        while True:
            params = {"after": after, "direction": "asc", "page_size": PAGE_SIZE}
            if token:
                params["page_token"] = token
            page = self._get(f"{self.config.trading_url}/v2/account/activities/FILL", params=params)
            for row in page:
                symbol = self._from_broker.get(row["symbol"])
                if symbol is None:
                    continue  # not an instrument this engine trades
                ts = parse_time(row["transaction_time"])
                if ts < since:
                    continue
                out.append(
                    Fill(
                        fill_id=row["id"],
                        order_id=row["order_id"],
                        instrument=symbol,
                        side=Side.BUY if row["side"] == "buy" else Side.SELL,
                        qty=float(row["qty"]),
                        price=float(row["price"]),
                        # Alpaca books crypto fees as separate CFEE activities; see the README.
                        fee=0.0,
                        ts=ts,
                    )
                )
            if len(page) < PAGE_SIZE:
                return out
            token = page[-1]["id"]

    def find_order(self, client_id: str) -> str | None:
        response = self._http.get(
            f"{self.config.trading_url}/v2/orders:by_client_order_id",
            params={"client_order_id": client_id},
            headers=self._headers,
        )
        if response.status_code == 200:
            return response.json()["id"]
        if response.status_code == 404:
            return None
        raise AlpacaError(f"lookup {client_id}: {response.status_code} {_message(response)}")

    def order_state(self, order_id: str) -> OrderState | None:
        response = self._http.get(
            f"{self.config.trading_url}/v2/orders/{order_id}", headers=self._headers
        )
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise AlpacaError(f"order {order_id}: {response.status_code} {_message(response)}")
        body = response.json()
        status = body["status"] if body["status"] in _FINISHED else "open"
        return OrderState(status, float(body.get("filled_qty") or 0))

    def open_orders(self) -> dict[str, str]:
        rows = self._get(
            f"{self.config.trading_url}/v2/orders",
            params={"status": "open", "limit": 500, "direction": "asc"},
        )
        return {row["id"]: row["client_order_id"] for row in rows}

    # --- extras for the check command ------------------------------------------------------

    def account(self) -> dict:
        return self._get(f"{self.config.trading_url}/v2/account")

    def configurations(self) -> dict:
        return self._get(f"{self.config.trading_url}/v2/account/configurations")

    def use_cash_only(self) -> dict:
        """Set the margin multiplier to 1, so buying power is the cash in the account. Paper
        accounts start at 4x; the engine never borrows, but the broker shouldn't let it either."""
        response = self._http.patch(
            f"{self.config.trading_url}/v2/account/configurations",
            json={"max_margin_multiplier": "1"},
            headers=self._headers,
        )
        if response.status_code != 200:
            raise AlpacaError(f"cash-only: {response.status_code} {_message(response)}")
        return response.json()

    def market_clock(self) -> dict:
        return self._get(f"{self.config.trading_url}/v2/clock")

    def activities(self, kind: str, limit: int = 5) -> list[dict]:
        return self._get(
            f"{self.config.trading_url}/v2/account/activities/{kind}",
            params={"direction": "desc", "page_size": limit},
        )

    def order(self, broker_id: str) -> dict:
        return self._get(f"{self.config.trading_url}/v2/orders/{broker_id}")

    # --- internals -------------------------------------------------------------------------

    def _get(self, url: str, params: dict | None = None):
        response = self._http.get(url, params=params, headers=self._headers)
        if response.status_code != 200:
            raise AlpacaError(f"GET {url}: {response.status_code} {_message(response)}")
        return response.json()

    def _info(self, instrument: str) -> InstrumentInfo:
        if instrument not in self.instruments:
            raise KeyError(f"alpaca does not trade {instrument}")
        return self.instruments[instrument]


_FRACTION = re.compile(r"\.(\d+)")


def parse_time(text: str) -> datetime:
    """Alpaca timestamps carry nanoseconds, which datetime can't hold; keep microseconds."""
    text = text.replace("Z", "+00:00")
    text = _FRACTION.sub(lambda m: "." + m.group(1)[:6].ljust(6, "0"), text, count=1)
    return datetime.fromisoformat(text).astimezone(UTC)


def _decimal(value: float, places: int = 9) -> str:
    text = f"{value:.{places}f}".rstrip("0").rstrip(".")
    return text or "0"


def _price(value: float, asset_class: str) -> str:
    # Alpaca rejects sub-penny limit prices on stocks at or above $1.
    if asset_class == "equity" and value >= 1:
        return f"{value:.2f}"
    return _decimal(value)


def _message(response: httpx.Response) -> str:
    try:
        return response.json().get("message", response.text)
    except ValueError:
        return response.text


# --- check command -----------------------------------------------------------------------------


def check(venue: AlpacaVenue, *, order_test: bool, out=print) -> None:
    now = venue.clock()
    clock = venue.market_clock()
    out(f"clock: exchange open={clock['is_open']}, next open {clock['next_open']}")
    account = venue.account()
    out(
        f"account {account.get('account_number')} ({account.get('status')}): "
        f"cash {account.get('cash')}, buying power {account.get('buying_power')}, "
        f"non-marginable buying power {account.get('non_marginable_buying_power')}"
    )
    multiplier = venue.configurations().get("max_margin_multiplier")
    note = "" if str(multiplier) == "1" else " (margin allowed; run `cash-only` to turn it off)"
    out(f"margin multiplier: {multiplier}{note}")
    out(f"positions: {venue.positions()}")
    for symbol in venue.instruments:
        status = venue.session_status(symbol)
        quote = venue.quote(symbol)
        if quote is None:
            out(
                f"{symbol}: session {status.session}, no quote ({venue.last_quote_error or 'empty'})"
            )
            continue
        age = (now - quote.ts).total_seconds()
        out(
            f"{symbol}: session {status.session}, bid {quote.bid} ask {quote.ask}, "
            f"quote {age:,.0f}s old"
        )
    for kind in ("FILL", "CFEE"):
        rows = venue.activities(kind)
        out(f"latest {kind} activities ({len(rows)}):")
        for row in rows:
            out("  " + json.dumps(row, sort_keys=True))
    if order_test:
        _order_test(venue, out)


def cash_only(venue: AlpacaVenue, out=print) -> None:
    before = venue.configurations().get("max_margin_multiplier")
    after = venue.use_cash_only().get("max_margin_multiplier")
    account = venue.account()
    out(f"margin multiplier: {before} -> {after}")
    out(f"cash {account.get('cash')}, buying power {account.get('buying_power')}")


def _order_test(venue: AlpacaVenue, out) -> None:
    symbol = "VOO"
    status = venue.session_status(symbol)
    quote = venue.quote(symbol)
    if not status.is_open or quote is None:
        out(f"order test skipped: {symbol} session {status.session}, quote {quote}")
        return
    # Half the market price: it can't fill, and it's cancelled straight away.
    order = Order(
        client_id=f"check-{uuid.uuid4().hex[:12]}",
        instrument=symbol,
        side=Side.BUY,
        qty=1,
        limit_price=round(quote.bid * 0.5, 2),
        order_type=OrderType.LIMIT,
        time_in_force=min(status.time_in_force),
    )
    ack = venue.place(order)
    out(f"order test: placed {order.client_id} -> {ack}")
    if not ack.accepted:
        return
    out(f"order test: lookup by client id -> {venue.find_order(order.client_id)}")
    out(f"order test: cancel -> {venue.cancel(ack.order_id)}")
    out(f"order test: final status -> {venue.order(ack.order_id).get('status')}")


def main(argv: list[str] | None = None) -> None:
    from .replay import DEFAULT_INSTRUMENTS

    parser = argparse.ArgumentParser(description="Look at the Alpaca paper account")
    parser.add_argument("command", choices=["check", "cash-only"])
    parser.add_argument(
        "--order-test", action="store_true", help="place and cancel one far-away limit order"
    )
    args = parser.parse_args(argv)
    try:
        venue = AlpacaVenue(AlpacaConfig.from_env(), DEFAULT_INSTRUMENTS, MarketCalendar())
        if args.command == "cash-only":
            cash_only(venue)
        else:
            check(venue, order_test=args.order_test)
    except AlpacaError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
