"""A one-page dashboard for one account: weights against targets, resting orders, today's
trades, alerts, and a halt button.

    export DATABASE_URL=...
    python -m rebalancer.dashboard --account paper-main     # http://127.0.0.1:8050

It reads Postgres and nothing else, so it works whether or not the engine is running. Connect
as a login in the rebalancer_dashboard role: that can read everything, insert a halt and
acknowledge an alert, and nothing more. In particular it can't clear a halt, so the worst a
stray click can do is stop trading.

It listens on localhost only. To use it from a phone, reach it through an SSH tunnel or a
private network rather than opening the port.
"""

from __future__ import annotations

import argparse
import hmac
import html
import logging
import os
import secrets
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

from .sessions import ET

log = logging.getLogger("rebalancer.dashboard")

# The watchdog's timeout: past this the heartbeat is shown as late.
HEARTBEAT_LATE = timedelta(minutes=5)


@dataclass(frozen=True)
class Summary:
    account: str
    mode: str
    last_cycle: datetime | None
    session: str | None
    account_value: float | None
    halted: str | None  # what the engine reported in its last cycle


@dataclass(frozen=True)
class SleeveRow:
    sleeve: str
    target: float
    weight: float
    value: float
    band: float | None  # allowed drift either side of target; None for cash
    stale: bool

    @property
    def drift(self) -> float:
        return self.weight - self.target

    @property
    def status(self) -> str:
        if self.stale:
            return "stale"
        if self.band is None:
            return ""
        return "out" if abs(self.drift) > self.band + 1e-9 else "in"


@dataclass(frozen=True)
class OrderRow:
    created_at: datetime
    sleeve: str
    instrument: str
    side: str
    qty: float
    filled_qty: float
    limit_price: float | None
    status: str
    session: str


@dataclass(frozen=True)
class TradeRow:
    ts: datetime
    sleeve: str | None
    instrument: str
    side: str
    qty: float
    price: float
    fee: float


@dataclass(frozen=True)
class AlertRow:
    id: int
    ts: datetime
    level: str
    message: str


@dataclass(frozen=True)
class HaltRow:
    scope: str
    lane: str | None
    reason: str
    source: str
    started_at: datetime
    until: datetime | None


@dataclass(frozen=True)
class Page:
    now: datetime
    summary: Summary
    sleeves: list[SleeveRow]
    orders: list[OrderRow]
    trades: list[TradeRow]
    alerts: list[AlertRow]
    halts: list[HaltRow]


# --- Postgres ----------------------------------------------------------------------------------


class DashboardStore:
    def __init__(self, connect: Callable[[], object], account_name: str):
        self._connect = connect
        self.conn = None
        row = (
            self._db()
            .execute("select id, name, mode from accounts where name = %s", (account_name,))
            .fetchone()
        )
        if row is None:
            raise LookupError(f"no account named {account_name}: run the engine on it first")
        self.account_id, self.account_name, self.mode = row

    def _db(self):
        if self.conn is None or self.conn.closed:
            self.conn = self._connect()
            if not self.conn.autocommit:
                raise ValueError("connect with autocommit=True")
        return self.conn

    def page(self, now: datetime) -> Page:
        cycle = (
            self._db()
            .execute(
                """select id, ts, session, account_value, halted, model_version_id from cycles
               where account_id = %s order by ts desc, id desc limit 1""",
                (self.account_id,),
            )
            .fetchone()
        )
        if cycle is None:
            summary = Summary(self.account_name, self.mode, None, None, None, None)
            sleeves = []
        else:
            cycle_id, ts, session, value, halted, model_version_id = cycle
            summary = Summary(
                self.account_name,
                self.mode,
                ts,
                session,
                None if value is None else float(value),
                halted,
            )
            sleeves = self._sleeves(cycle_id, model_version_id)
        return Page(
            now,
            summary,
            sleeves,
            self._orders(),
            self._trades(now),
            self._alerts(),
            self._halts(now),
        )

    def halt(self, now: datetime, reason: str) -> None:
        self._db().execute(
            """insert into halts (account_id, scope, reason, source, started_at)
               values (%s, 'global', %s, 'dashboard', %s)""",
            (self.account_id, reason, now),
        )

    def acknowledge(self, alert_id: int, now: datetime) -> None:
        self._db().execute(
            """update alerts set acknowledged_at = %s
               where id = %s and (account_id = %s or account_id is null)
                 and acknowledged_at is null""",
            (now, alert_id, self.account_id),
        )

    # --- sections --------------------------------------------------------------------------

    def _sleeves(self, cycle_id: int, model_version_id: int) -> list[SleeveRow]:
        rows = (
            self._db()
            .execute(
                """select s.sleeve, s.target, m.weight, m.value, s.band_abs, s.band_rel, m.stale
               from model_sleeves s
               join sleeve_marks m on m.sleeve = s.sleeve and m.cycle_id = %s
               where s.model_version_id = %s
               order by s.target desc, s.sleeve""",
                (cycle_id, model_version_id),
            )
            .fetchall()
        )
        out = []
        for sleeve, target, weight, value, band_abs, band_rel, stale in rows:
            widths = [float(band_abs)] if band_abs is not None else []
            if band_rel is not None:
                widths.append(float(band_rel) * float(target))
            out.append(
                SleeveRow(
                    sleeve,
                    float(target),
                    float(weight),
                    float(value),
                    min(widths) if widths else None,
                    stale,
                )
            )
        return out

    def _orders(self) -> list[OrderRow]:
        rows = (
            self._db()
            .execute(
                """select o.created_at, o.sleeve, o.instrument, o.side::text, o.qty,
                      coalesce(f.qty, 0), o.limit_price, o.status::text, o.session
               from orders o
               left join (select order_id, sum(qty) as qty from fills group by order_id) f
                      on f.order_id = o.id
               where o.account_id = %s and o.status in ('pending', 'accepted', 'partially_filled')
               order by o.created_at, o.id""",
                (self.account_id,),
            )
            .fetchall()
        )
        return [
            OrderRow(
                created,
                sleeve,
                instrument,
                side,
                float(qty),
                float(filled),
                None if limit is None else float(limit),
                status,
                session,
            )
            for created, sleeve, instrument, side, qty, filled, limit, status, session in rows
        ]

    def _trades(self, now: datetime) -> list[TradeRow]:
        midnight = datetime.combine(now.astimezone(ET).date(), datetime.min.time(), tzinfo=ET)
        rows = (
            self._db()
            .execute(
                """select f.ts, o.sleeve, f.instrument, f.side::text, f.qty, f.price, f.fee
               from fills f left join orders o on o.id = f.order_id
               where f.account_id = %s and f.ts >= %s
               order by f.ts desc, f.id desc""",
                (self.account_id, midnight),
            )
            .fetchall()
        )
        return [
            TradeRow(ts, sleeve, inst, side, float(qty), float(price), float(fee))
            for ts, sleeve, inst, side, qty, price, fee in rows
        ]

    def _alerts(self) -> list[AlertRow]:
        rows = (
            self._db()
            .execute(
                """select id, ts, level, message from alerts
               where (account_id = %s or account_id is null) and acknowledged_at is null
               order by ts desc, id desc limit 50""",
                (self.account_id,),
            )
            .fetchall()
        )
        return [AlertRow(*row) for row in rows]

    def _halts(self, now: datetime) -> list[HaltRow]:
        rows = (
            self._db()
            .execute(
                """select scope::text, lane, reason, source, started_at, until from halts
               where (account_id = %s or account_id is null) and cleared_at is null
                 and (scope <> 'daily' or until > %s)
               order by started_at, id""",
                (self.account_id, now),
            )
            .fetchall()
        )
        return [HaltRow(*row) for row in rows]


# --- HTML --------------------------------------------------------------------------------------

STYLE = """
body { font: 15px/1.4 system-ui, sans-serif; margin: 0 auto; max-width: 960px; padding: 12px;
       color: #1d2327; background: #fff; }
h1 { font-size: 20px; margin: 4px 0 8px; }
h2 { font-size: 16px; margin: 22px 0 6px; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 4px 8px 4px 0; border-bottom: 1px solid #e3e6e8; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.wrap { overflow-x: auto; }
.muted { color: #646970; }
.bad { color: #b32d2e; font-weight: 600; }
.warn { color: #996800; }
.banner { padding: 8px 10px; border-radius: 4px; margin: 8px 0; }
.banner.bad { background: #fcf0f1; color: inherit; font-weight: normal; }
.banner.ok { background: #edfaef; color: #00450c; }
form.inline { display: inline; }
input[type=text] { width: 60%; max-width: 360px; padding: 6px; }
button { padding: 6px 12px; }
button.halt { background: #b32d2e; color: #fff; border: 0; border-radius: 4px; }
"""


def render(page: Page, token: str) -> str:
    s = page.summary
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<meta http-equiv='refresh' content='30'>",
        f"<title>Rebalancer: {_e(s.account)}</title><style>{STYLE}</style></head><body>",
        f"<h1>{_e(s.account)} <span class='muted'>({_e(s.mode)})</span></h1>",
        _status(page),
        _halt_section(page, token),
        "<h2>Weights</h2>",
        _sleeve_table(page.sleeves),
        f"<h2>Resting orders ({len(page.orders)})</h2>",
        _order_table(page.orders),
        f"<h2>Today's trades ({len(page.trades)})</h2>",
        _trade_table(page.trades),
        f"<h2>Alerts to acknowledge ({len(page.alerts)})</h2>",
        _alert_table(page.alerts, token),
        f"<p class='muted'>As of {_time(page.now)} ET. Refreshes every 30 seconds.</p>",
        "</body></html>",
    ]
    return "".join(parts)


def _status(page: Page) -> str:
    s = page.summary
    if s.last_cycle is None:
        return "<p class='warn'>The engine hasn't recorded a cycle for this account yet.</p>"
    age = page.now - s.last_cycle
    beat = f"last cycle {_time(s.last_cycle)} ET, {_ago(age)} ago"
    if age > HEARTBEAT_LATE:
        beat = f"<span class='bad'>{beat}: the engine looks stopped</span>"
    value = "-" if s.account_value is None else f"${s.account_value:,.2f}"
    return f"<p>{value} &middot; {_e(s.session or '-')} session &middot; {beat}</p>"


def _halt_section(page: Page, token: str) -> str:
    if page.halts:
        items = "".join(
            f"<li><b>{_e(h.scope if h.lane is None else h.lane)}</b>: {_e(h.reason)} "
            f"<span class='muted'>({_e(h.source)}, since {_time(h.started_at)} ET"
            + (f", until {_time(h.until)} ET" if h.until else "")
            + ")</span></li>"
            for h in page.halts
        )
        banner = f"<div class='banner bad'><b class='bad'>Halted</b><ul>{items}</ul></div>"
    else:
        banner = "<div class='banner ok'>Trading is not halted.</div>"
    form = (
        "<form method='post' action='/halt'>"
        f"<input type='hidden' name='token' value='{_e(token)}'>"
        "<input type='text' name='reason' placeholder='Reason (optional)' maxlength='200'> "
        "<button class='halt' type='submit'>Halt all trading</button></form>"
        "<p class='muted'>The engine stops trading and cancels resting orders on its next "
        "cycle. Clearing a halt is done by hand in the database.</p>"
    )
    return banner + form


def _sleeve_table(rows: list[SleeveRow]) -> str:
    if not rows:
        return "<p class='muted'>No weights yet.</p>"
    body = "".join(
        f"<tr><td>{_e(r.sleeve)}</td><td class='n'>{_pct(r.target)}</td>"
        f"<td class='n'>{_pct(r.weight)}</td><td class='n'>{_pct(r.drift, signed=True)}</td>"
        f"<td class='n'>{'-' if r.band is None else '&plusmn;' + _pct(r.band)}</td>"
        f"<td class='n'>${r.value:,.2f}</td>"
        f"<td class='{ {'out': 'bad', 'stale': 'warn'}.get(r.status, '') }'>{r.status}</td></tr>"
        for r in rows
    )
    return _table(["Sleeve", "Target", "Weight", "Drift", "Band", "Value", ""], body, 1)


def _order_table(rows: list[OrderRow]) -> str:
    if not rows:
        return "<p class='muted'>Nothing resting.</p>"
    body = "".join(
        f"<tr><td>{_time(r.created_at)}</td><td>{_e(r.sleeve)}</td><td>{_e(r.instrument)}</td>"
        f"<td>{_e(r.side)}</td><td class='n'>{r.qty:g}</td><td class='n'>{r.filled_qty:g}</td>"
        f"<td class='n'>{'market' if r.limit_price is None else f'{r.limit_price:,.2f}'}</td>"
        f"<td>{_e(r.status)}</td><td>{_e(r.session)}</td></tr>"
        for r in rows
    )
    headers = ["Placed", "Sleeve", "Instrument", "Side", "Qty", "Filled", "Limit", "Status"]
    return _table([*headers, "Session"], body, 4)


def _trade_table(rows: list[TradeRow]) -> str:
    if not rows:
        return "<p class='muted'>No trades today.</p>"
    body = "".join(
        f"<tr><td>{_clock(r.ts)}</td><td>{_e(r.sleeve or '-')}</td><td>{_e(r.instrument)}</td>"
        f"<td>{_e(r.side)}</td><td class='n'>{r.qty:g}</td><td class='n'>{r.price:,.2f}</td>"
        f"<td class='n'>${r.qty * r.price:,.2f}</td><td class='n'>${r.fee:,.2f}</td></tr>"
        for r in rows
    )
    headers = ["Time", "Sleeve", "Instrument", "Side", "Qty", "Price", "Notional", "Fee"]
    return _table(headers, body, 4)


def _alert_table(rows: list[AlertRow], token: str) -> str:
    if not rows:
        return "<p class='muted'>None.</p>"
    body = "".join(
        f"<tr><td>{_time(r.ts)}</td>"
        f"<td class='{ {'critical': 'bad', 'warning': 'warn'}.get(r.level, '') }'>{_e(r.level)}</td>"
        f"<td>{_e(r.message)}</td><td><form class='inline' method='post' action='/alerts/{r.id}/ack'>"
        f"<input type='hidden' name='token' value='{_e(token)}'>"
        "<button type='submit'>OK</button></form></td></tr>"
        for r in rows
    )
    return _table(["Time", "Level", "Message", ""], body, None)


def _table(headers: list[str], body: str, numeric_from: int | None) -> str:
    cells = "".join(
        f"<th class='n'>{h}</th>"
        if numeric_from is not None and i >= numeric_from and h
        else f"<th>{h}</th>"
        for i, h in enumerate(headers)
    )
    return f"<div class='wrap'><table><tr>{cells}</tr>{body}</table></div>"


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def _pct(value: float, *, signed: bool = False) -> str:
    shown = round(value * 100, 1) + 0.0  # adding 0.0 turns -0.0 into 0.0
    return f"{shown:+.1f}%" if signed else f"{shown:.1f}%"


def _time(ts: datetime) -> str:
    return ts.astimezone(ET).strftime("%b %d %H:%M")


def _clock(ts: datetime) -> str:
    return ts.astimezone(ET).strftime("%H:%M")


def _ago(delta: timedelta) -> str:
    seconds = max(0, int(delta.total_seconds()))
    if seconds < 120:
        return f"{seconds} s"
    if seconds < 7200:
        return f"{seconds // 60} min"
    return f"{seconds // 3600} h"


# --- HTTP --------------------------------------------------------------------------------------


def make_handler(
    store: DashboardStore,
    token: str,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> type[BaseHTTPRequestHandler]:
    """A request handler bound to one store. Every form carries a token made when the server
    starts, so another page open in the same browser can't post a halt or an acknowledgement."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/":
                self._send(404, "not found")
                return
            try:
                page = store.page(clock())
            except Exception as exc:
                log.exception("could not read the database")
                self._send(503, f"could not read the database: {type(exc).__name__}")
                return
            self._send(200, render(page, token), "text/html; charset=utf-8")

        def do_POST(self) -> None:
            length = min(int(self.headers.get("Content-Length") or 0), 10_000)
            form = parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
            if not hmac.compare_digest(form.get("token", [""])[0], token):
                self._send(403, "stale or missing form token: reload the page")
                return
            now = clock()
            try:
                if self.path == "/halt":
                    reason = form.get("reason", [""])[0].strip()[:200]
                    store.halt(now, reason or "halted from the dashboard")
                    log.warning("halt from the dashboard: %s", reason or "(no reason)")
                elif self.path.startswith("/alerts/") and self.path.endswith("/ack"):
                    alert_id = self.path.split("/")[2]
                    if not alert_id.isdigit():
                        self._send(404, "not found")
                        return
                    store.acknowledge(int(alert_id), now)
                else:
                    self._send(404, "not found")
                    return
            except Exception as exc:
                log.exception("could not write to the database")
                self._send(503, f"could not write to the database: {type(exc).__name__}")
                return
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

        def _send(self, status: int, body: str, content_type: str = "text/plain") -> None:
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args) -> None:
            log.info("%s %s", self.address_string(), format % args)

    return Handler


def main(argv: list[str] | None = None) -> None:
    import psycopg

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--account", required=True, help="the account name the runner uses")
    parser.add_argument("--dsn", default=os.environ.get("DATABASE_URL"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args(argv)
    if not args.dsn:
        sys.exit("set DATABASE_URL or pass --dsn")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        store = DashboardStore(
            lambda: psycopg.connect(args.dsn, autocommit=True, connect_timeout=10), args.account
        )
    except LookupError as exc:
        sys.exit(str(exc))
    server = HTTPServer((args.host, args.port), make_handler(store, secrets.token_urlsafe(24)))
    log.info("dashboard for %s on http://%s:%s", args.account, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
