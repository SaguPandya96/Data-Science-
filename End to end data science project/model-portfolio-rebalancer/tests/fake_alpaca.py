"""A small in-memory stand-in for the Alpaca Trading and Market Data APIs.

It answers the endpoints the adapter uses, in the shapes Alpaca documents, and fills a limit
order as soon as it crosses the quote. It's a model of Alpaca, not Alpaca: the real check is
`python -m rebalancer.alpaca check` against a paper account.
"""

from __future__ import annotations

import itertools
import json
from datetime import UTC, datetime

import httpx

KEY, SECRET = "PKTEST", "secret"


def stamp(ts: datetime) -> str:
    # Alpaca sends nanoseconds.
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f") + "123Z"


class FakeAlpaca:
    def __init__(self, now: datetime, cash: float = 10_000.0):
        self.now = now
        self.cash = cash
        self.positions: dict[str, float] = {}  # BTCUSD style, as Alpaca reports them
        self.quotes: dict[str, tuple[float, float]] = {}  # symbol as ordered: (bid, ask)
        self.quote_time: dict[str, datetime] = {}
        self.orders: dict[str, dict] = {}
        self.by_client: dict[str, str] = {}
        self.activities: list[dict] = []
        self.requests: list[tuple[str, str, dict | None]] = []
        self.timeout_next_post: str | None = None  # "before" or "after" the order is stored
        self.reject_next_post: tuple[int, str] | None = None
        self.fill_orders = True
        self.configuration = {"max_margin_multiplier": "4", "no_shorting": False}
        self._ids = itertools.count(1)
        self._activity_ids = itertools.count(1)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handle))

    def set_quote(self, symbol: str, bid: float, ask: float, at: datetime | None = None) -> None:
        self.quotes[symbol] = (bid, ask)
        self.quote_time[symbol] = at or self.now

    # --- routing ---------------------------------------------------------------------------

    def handle(self, request: httpx.Request) -> httpx.Response:
        if (
            request.headers.get("APCA-API-KEY-ID") != KEY
            or request.headers.get("APCA-API-SECRET-KEY") != SECRET
        ):
            return self._json(403, {"message": "forbidden"})
        path, params = request.url.path, dict(request.url.params)
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, path, body))
        if request.method == "POST" and path == "/v2/orders":
            return self._post_order(request, body)
        if request.method == "DELETE" and path.startswith("/v2/orders/"):
            return self._cancel(path.rsplit("/", 1)[1])
        if request.method == "PATCH" and path == "/v2/account/configurations":
            self.configuration.update(body)
            return self._json(200, self.configuration)
        routes = {
            "/v2/clock": lambda: {"is_open": True, "next_open": "2026-11-04T09:30:00-05:00"},
            "/v2/account": lambda: {
                "account_number": "PA0001",
                "status": "ACTIVE",
                "cash": f"{self.cash:.2f}",
                "buying_power": f"{self.cash * int(self.configuration['max_margin_multiplier']):.2f}",
                "non_marginable_buying_power": f"{self.cash:.2f}",
            },
            "/v2/positions": lambda: [
                {"symbol": s, "qty": f"{q:.9f}"} for s, q in self.positions.items() if q
            ],
            "/v2/account/configurations": lambda: self.configuration,
            "/v2/account/activities/FILL": lambda: self._activities(params),
            "/v2/account/activities/CFEE": lambda: [],
            "/v1beta3/crypto/us/latest/quotes": lambda: self._crypto_quotes(params),
        }
        if path in routes:
            return self._json(200, routes[path]())
        if path == "/v2/orders:by_client_order_id":
            broker_id = self.by_client.get(params["client_order_id"])
            if broker_id is None:
                return self._json(404, {"message": "order not found"})
            return self._json(200, self.orders[broker_id])
        if path == "/v2/orders" and params.get("status") == "open":
            return self._json(200, [o for o in self.orders.values() if o["status"] == "new"])
        if path.startswith("/v2/orders/"):
            order = self.orders.get(path.rsplit("/", 1)[1])
            if order is None:
                return self._json(404, {"message": "order not found"})
            return self._json(200, order)
        if path.startswith("/v2/stocks/") and path.endswith("/quotes/latest"):
            symbol = path.split("/")[3]
            if symbol not in self.quotes:
                return self._json(404, {"message": "no quote"})
            return self._json(200, {"symbol": symbol, "quote": self._quote(symbol)})
        return self._json(404, {"message": f"no route {request.method} {path}"})

    # --- behaviour -------------------------------------------------------------------------

    def _post_order(self, request: httpx.Request, body: dict) -> httpx.Response:
        if self.timeout_next_post == "before":
            self.timeout_next_post = None
            raise httpx.ReadTimeout("timed out", request=request)
        if self.reject_next_post:
            status, message = self.reject_next_post
            self.reject_next_post = None
            return self._json(status, {"message": message})
        if body["client_order_id"] in self.by_client:
            return self._json(422, {"message": "client_order_id must be unique"})
        broker_id = f"ord-{next(self._ids)}"
        order = {**body, "id": broker_id, "status": "new", "filled_qty": "0"}
        self.orders[broker_id] = order
        self.by_client[body["client_order_id"]] = broker_id
        self._try_fill(order)
        if self.timeout_next_post == "after":
            self.timeout_next_post = None
            raise httpx.ReadTimeout("timed out", request=request)
        return self._json(200, order)

    def _try_fill(self, order: dict) -> None:
        bid, ask = self.quotes[order["symbol"]]
        qty, limit = float(order["qty"]), float(order["limit_price"])
        price = ask if order["side"] == "buy" else bid
        crosses = price <= limit if order["side"] == "buy" else price >= limit
        if not (self.fill_orders and crosses):
            return
        sign = 1 if order["side"] == "buy" else -1
        held = order["symbol"].replace("/", "")
        self.positions[held] = self.positions.get(held, 0.0) + sign * qty
        self.cash -= sign * qty * price
        order.update(status="filled", filled_qty=order["qty"], filled_avg_price=str(price))
        self.activities.append(
            {
                "id": f"{self.now:%Y%m%d%H%M%S}000::act-{next(self._activity_ids)}",
                "activity_type": "FILL",
                "type": "fill",
                "transaction_time": stamp(self.now),
                "symbol": held,
                "side": order["side"],
                "qty": order["qty"],
                "price": str(price),
                "order_id": order["id"],
                "cum_qty": order["qty"],
                "leaves_qty": "0",
            }
        )

    def _cancel(self, broker_id: str) -> httpx.Response:
        order = self.orders.get(broker_id)
        if order is None:
            return self._json(404, {"message": "order not found"})
        if order["status"] != "new":
            return self._json(422, {"message": f"order is {order['status']}"})
        order["status"] = "canceled"
        return httpx.Response(204)

    def _activities(self, params: dict) -> list[dict]:
        rows = [a for a in self.activities if a["transaction_time"] > params.get("after", "")]
        if "page_token" in params:
            ids = [a["id"] for a in rows]
            rows = rows[ids.index(params["page_token"]) + 1 :]
        return rows[: int(params.get("page_size", 100))]

    def _quote(self, symbol: str) -> dict:
        bid, ask = self.quotes[symbol]
        return {"bp": bid, "bs": 100, "ap": ask, "as": 100, "t": stamp(self.quote_time[symbol])}

    def _crypto_quotes(self, params: dict) -> dict:
        symbols = params["symbols"].split(",")
        return {"quotes": {s: self._quote(s) for s in symbols if s in self.quotes}}

    @staticmethod
    def _json(status: int, payload) -> httpx.Response:
        return httpx.Response(status, json=payload)
