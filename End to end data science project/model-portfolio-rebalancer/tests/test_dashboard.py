import http.client
import threading
from datetime import timedelta
from http.server import HTTPServer
from urllib.parse import urlencode

import pytest
from conftest import et
from test_runner import alpaca_runner, paper_account

from rebalancer.dashboard import (
    AlertRow,
    HaltRow,
    OrderRow,
    Page,
    SleeveRow,
    Summary,
    TradeRow,
    make_handler,
    render,
)

T10 = et(2026, 11, 3, 10, 0)
TOKEN = "form-token"


def page(**changes):
    base = dict(
        now=T10 + timedelta(seconds=40),
        summary=Summary("paper-main", "paper", T10, "regular", 100_000.0, None),
        sleeves=[
            SleeveRow("us_large_cap", 0.45, 0.42, 42_000.0, 0.03, False),
            SleeveRow("btc", 0.20, 0.26, 26_000.0, 0.03, False),
            SleeveRow("intl_equity", 0.15, 0.15, 15_000.0, 0.02, True),
            SleeveRow("cash", 0.10, 0.07, 7_000.0, None, False),
        ],
        orders=[
            OrderRow(
                T10, "btc", "BTC-USD", "sell", 0.03, 0.01, 99_800.0, "partially_filled", "regular"
            )
        ],
        trades=[TradeRow(T10, "btc", "BTC-USD", "sell", 0.01, 99_800.0, 2.5)],
        alerts=[AlertRow(7, T10, "critical", "trading halted: <script>alert(1)</script>")],
        halts=[],
    )
    base.update(changes)
    return Page(**base)


# --- the page ----------------------------------------------------------------------------------


def test_sleeves_are_marked_in_out_or_stale():
    rows = {r.sleeve: r for r in page().sleeves}
    assert rows["us_large_cap"].status == "in"  # 3 points under, band is 3
    assert rows["btc"].status == "out"
    assert rows["intl_equity"].status == "stale"
    assert rows["cash"].status == ""  # no band
    html = render(page(), TOKEN)
    assert "<td class='bad'>out</td>" in html
    assert "+6.0%" in html and "&plusmn;3.0%" in html
    tiny = render(page(sleeves=[SleeveRow("eth", 0.10, 0.09999, 9_999.0, 0.02, False)]), TOKEN)
    assert "+0.0%" in tiny and "-0.0%" not in tiny


def test_everything_from_the_database_is_escaped():
    halts = [HaltRow("global", None, "<b>fat finger</b>", "dashboard", T10, None)]
    html = render(page(halts=halts), TOKEN)
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<b>fat finger</b>" not in html and "&lt;b&gt;fat finger" in html


def test_halts_and_a_late_heartbeat_stand_out():
    html = render(page(), TOKEN)
    assert "Trading is not halted." in html
    assert "looks stopped" not in html

    halts = [HaltRow("lane", "alpaca/crypto", "3 rejects in 10 minutes", "engine", T10, None)]
    late = render(page(halts=halts, now=T10 + timedelta(minutes=6)), TOKEN)
    assert "Halted" in late and "alpaca/crypto" in late
    assert "last cycle Nov 03 10:00 ET, 6 min ago: the engine looks stopped" in late


def test_an_account_with_no_cycles_yet():
    summary = Summary("paper-main", "paper", None, None, None, None)
    html = render(page(summary=summary, sleeves=[], orders=[], trades=[], alerts=[]), TOKEN)
    assert "hasn't recorded a cycle" in html
    assert "Nothing resting." in html and "No trades today." in html


# --- the server --------------------------------------------------------------------------------


class FakeStore:
    def __init__(self):
        self.halts, self.acks = [], []
        self.fail = False

    def page(self, now):
        if self.fail:
            raise ConnectionError("database is gone")
        return page(now=now)

    def halt(self, now, reason):
        self.halts.append(reason)

    def acknowledge(self, alert_id, now):
        self.acks.append(alert_id)


@pytest.fixture
def server():
    store = FakeStore()
    httpd = HTTPServer(("127.0.0.1", 0), make_handler(store, TOKEN, clock=lambda: T10))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield store, httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def request(port, method, path, form=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    body = urlencode(form) if form is not None else None
    headers = {"Content-Type": "application/x-www-form-urlencoded"} if body else {}
    conn.request(method, path, body=body, headers=headers)
    response = conn.getresponse()
    return response.status, response.read().decode(), response.getheader("Location")


def test_the_page_and_its_forms_carry_the_token(server):
    _, port = server
    status, body, _ = request(port, "GET", "/")
    assert status == 200
    assert body.count(f"name='token' value='{TOKEN}'") == 2  # the halt form and one alert
    assert request(port, "GET", "/other")[0] == 404


def test_halting_needs_the_token(server):
    store, port = server
    assert request(port, "POST", "/halt", {"reason": "x"})[0] == 403
    assert request(port, "POST", "/halt", {"reason": "x", "token": "guess"})[0] == 403
    assert store.halts == []

    status, _, location = request(port, "POST", "/halt", {"token": TOKEN, "reason": " news "})
    assert (status, location) == (303, "/")
    request(port, "POST", "/halt", {"token": TOKEN})
    assert store.halts == ["news", "halted from the dashboard"]


def test_acknowledging_an_alert(server):
    store, port = server
    assert request(port, "POST", "/alerts/7/ack", {"token": TOKEN})[0] == 303
    assert request(port, "POST", "/alerts/x/ack", {"token": TOKEN})[0] == 404
    assert request(port, "POST", "/alerts/8/ack", {"token": "guess"})[0] == 403
    assert store.acks == [7]


def test_a_database_failure_is_a_503_not_a_crash(server):
    store, port = server
    store.fail = True
    status, body, _ = request(port, "GET", "/")
    assert status == 503 and "ConnectionError" in body
    store.fail = False
    assert request(port, "GET", "/")[0] == 200


# --- against Postgres as the dashboard role ----------------------------------------------------


@pytest.fixture
def fake():
    return paper_account()


def dashboard_store(dsn):
    import psycopg

    from rebalancer.dashboard import DashboardStore

    def connect():
        return psycopg.connect(dsn, autocommit=True, options="-c role=rebalancer_dashboard")

    return DashboardStore(connect, "paper-main")


def refresh_quotes(fake):
    for symbol, (bid, ask) in list(fake.quotes.items()):
        fake.set_quote(symbol, bid, ask)


def test_the_dashboard_reads_the_engine_and_halts_it(schema, dsn, fake, calendar):
    import psycopg

    runner = alpaca_runner(dsn, fake, calendar)
    runner.engine.start(fake.now)
    fake.fill_orders = False
    runner.tick(fake.now)
    store = dashboard_store(dsn)

    first = store.page(fake.now)
    assert first.summary.account == "paper-main" and first.summary.session == "regular"
    sleeves = {r.sleeve: r for r in first.sleeves}
    assert sleeves["btc"].target == pytest.approx(0.20)
    assert sleeves["btc"].band == pytest.approx(0.03)
    assert sleeves["us_large_cap"].band == pytest.approx(0.03)  # the narrower of 3% and 20% of 45%
    assert sleeves["btc"].status == "out"
    assert [(o.instrument, o.side, o.status) for o in first.orders] == [
        ("BTC-USD", "sell", "accepted")
    ]
    assert first.trades == []

    # Next cycle the engine re-quotes and the sell fills.
    fake.fill_orders = True
    fake.now += timedelta(minutes=1)
    refresh_quotes(fake)
    runner.tick(fake.now)
    account = schema.execute("select id from accounts").fetchone()[0]
    schema.execute(
        """insert into fills (broker_fill_id, account_id, instrument, side, qty, price, ts)
           values ('yesterday', %s, 'VOO', 'buy', 1, 555, %s)""",
        (account, fake.now - timedelta(days=1)),
    )
    second = store.page(fake.now)
    assert second.orders == []
    assert [(t.sleeve, t.instrument, t.side) for t in second.trades] == [("btc", "BTC-USD", "sell")]

    store.halt(fake.now, "going on holiday")
    fake.now += timedelta(minutes=1)
    refresh_quotes(fake)
    assert runner.tick(fake.now).halted == "going on holiday"
    third = store.page(fake.now)
    assert [(h.reason, h.source) for h in third.halts] == [("going on holiday", "dashboard")]
    alert = next(a for a in third.alerts if "going on holiday" in a.message)
    store.acknowledge(alert.id, fake.now)
    assert alert.id not in {a.id for a in store.page(fake.now).alerts}

    # The dashboard role can stop trading but not restart it.
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        store.conn.execute("update halts set cleared_at = now()")
