from datetime import timedelta

import pytest
from conftest import MODELS, PRICES, et
from test_recovery import BTC_HEAVY
from test_runner import alpaca_runner, paper_account

from rebalancer.models import load_model
from rebalancer.replay import build_world
from rebalancer.venue import OrderState
from rebalancer.watchdog import AlertRow, ReconciliationRow, Watchdog

T10 = et(2026, 11, 3, 10, 0)


def minutes(n):
    return T10 + timedelta(minutes=n)


class MemoryStore:
    """What the watchdog reads and writes, without a database. `down` makes every call fail."""

    def __init__(self):
        self.beat = None
        self.reconciliation = None
        self.halts = []
        self.alerts = []
        self.delivered = {}
        self.down = False

    def _check(self):
        if self.down:
            raise ConnectionError("database is gone")

    def last_heartbeat(self):
        self._check()
        return self.beat

    def last_reconciliation(self):
        self._check()
        return self.reconciliation

    def global_halt_open(self):
        self._check()
        return bool(self.halts)

    def halt(self, ts, reason):
        self._check()
        self.halts.append(reason)

    def alert(self, ts, level, message):
        self._check()
        self.alerts.append(AlertRow(len(self.alerts) + 1, ts, level, message))

    def undelivered_alerts(self, since):
        self._check()
        return [a for a in self.alerts if a.id not in self.delivered and a.ts >= since]

    def mark_delivered(self, alert_id, ts):
        self._check()
        self.delivered[alert_id] = ts


class Phone:
    def __init__(self):
        self.received = []
        self.failures = 0  # how many sends to fail before they start going through

    def send(self, level, message):
        if self.failures:
            self.failures -= 1
            raise ConnectionError("no signal")
        self.received.append((level, message))


def messages(store, level=None):
    return [a.message for a in store.alerts if level is None or a.level == level]


@pytest.fixture
def growth():
    return load_model(MODELS / "growth-247.yaml")


@pytest.fixture
def resting(growth):
    """An engine that has just placed a BTC sell which is resting at the broker."""
    world = build_world(growth, T10, dict(PRICES), weights=BTC_HEAVY)
    world.venues["alpaca"].fills_paused = True
    order = next(o for o in world.step(T10, PRICES).orders if o.instrument == "BTC-USD")
    assert order.status == "placed"
    return world, order


def watchdog_for(world, store=None, phone=None):
    store = store or MemoryStore()
    store.beat = T10
    return Watchdog(store, world.venues["alpaca"], phone or Phone()), store


# --- heartbeat ---------------------------------------------------------------------------------


def test_a_silent_engine_has_its_orders_cancelled_after_five_minutes(resting):
    world, order = resting
    venue = world.venues["alpaca"]
    watchdog, store = watchdog_for(world)

    watchdog.check(minutes(5))
    assert venue.open_orders() == {order.order_id: order.client_id}
    assert store.alerts == []

    watchdog.check(minutes(5) + timedelta(seconds=30))
    assert venue.open_orders() == {}
    assert messages(store, "critical") == [
        "no engine heartbeat since 2026-11-03T10:00:00-05:00 (6 min): "
        "cancelled 1 order resting at the broker"
    ]
    watchdog.check(minutes(6))
    assert len(store.alerts) == 1  # once per outage, not every check

    store.beat = minutes(7)
    watchdog.check(minutes(7))
    assert messages(store, "info") == ["engine heartbeat is back"]


def test_the_engine_never_having_run_is_not_an_outage(resting):
    world, _ = resting
    watchdog, store = watchdog_for(world)
    store.beat = None
    watchdog.check(minutes(60))
    assert world.venues["alpaca"].open_orders() != {}
    assert store.alerts == []


def test_orders_that_appear_during_an_outage_are_cancelled_too(resting, growth):
    world, order = resting
    venue = world.venues["alpaca"]
    watchdog, store = watchdog_for(world)
    watchdog.check(minutes(6))
    # Something still places an order, say a hung engine that wakes up for one cycle.
    late = next(o for o in world.step(minutes(7), PRICES).orders if o.instrument == "BTC-USD")
    watchdog.check(minutes(8))
    assert venue.open_orders() == {}
    assert messages(store, "warning") == ["engine still silent: cancelled 1 order more"]
    assert late.order_id != order.order_id


def test_a_broker_failure_is_reported_once_and_retried(resting):
    world, _ = resting
    venue = world.venues["alpaca"]
    watchdog, store = watchdog_for(world)
    real = venue.open_orders
    venue.open_orders = lambda: (_ for _ in ()).throw(TimeoutError("broker timed out"))

    watchdog.check(minutes(6))
    watchdog.check(minutes(7))
    assert messages(store) == [
        "no engine heartbeat for 6 min and cancelling orders at the broker failed: "
        "TimeoutError: broker timed out"
    ]
    venue.open_orders = real
    watchdog.check(minutes(8))
    assert real() == {}
    assert messages(store)[-1].endswith("cancelled 1 order resting at the broker")


def test_losing_the_database_still_cancels_and_alerts_directly(resting):
    world, _ = resting
    phone = Phone()
    watchdog, store = watchdog_for(world, phone=phone)
    watchdog.check(minutes(1))  # sees the heartbeat
    store.down = True

    watchdog.check(minutes(2))
    watchdog.check(minutes(3))
    assert phone.received == [
        ("critical", "watchdog cannot use the database: ConnectionError: database is gone")
    ]
    # Without the database it can't tell whether the engine is alive, so it counts from the
    # last heartbeat it saw.
    watchdog.check(minutes(6))
    assert world.venues["alpaca"].open_orders() == {}
    assert phone.received[-1][0] == "critical"
    assert "cancelled 1 order" in phone.received[-1][1]


# --- reconciliation ----------------------------------------------------------------------------


def test_a_failed_reconciliation_nobody_halted_for_halts_trading(resting):
    world, _ = resting
    watchdog, store = watchdog_for(world)
    store.reconciliation = ReconciliationRow(7, minutes(1), False, 55.0)

    watchdog.check(minutes(1))
    assert store.halts == [
        "reconciliation at 2026-11-03T10:01:00-05:00 was off by $55.00 "
        "and the engine had not halted"
    ]
    assert messages(store, "critical") == [f"watchdog halted trading: {store.halts[0]}"]

    # Clearing the halt by hand is respected until a new reconciliation fails.
    store.halts.clear()
    watchdog.check(minutes(2))
    assert store.halts == []
    store.reconciliation = ReconciliationRow(8, minutes(3), False, None)
    watchdog.check(minutes(3))
    assert "off by an unpriced amount" in store.halts[0]


def test_no_halt_when_reconciliation_is_clean_or_already_halted(resting):
    world, _ = resting
    watchdog, store = watchdog_for(world)
    store.reconciliation = ReconciliationRow(1, minutes(1), True, 0.0)
    watchdog.check(minutes(1))
    store.reconciliation = ReconciliationRow(2, minutes(2), False, 55.0)
    store.halts.append("reconciliation gap $55.00")  # the engine's own halt
    watchdog.check(minutes(2))
    assert store.halts == ["reconciliation gap $55.00"]
    assert store.alerts == []


# --- delivery ----------------------------------------------------------------------------------


def test_alerts_go_out_once_and_are_retried_after_a_failure(resting):
    world, _ = resting
    phone = Phone()
    watchdog, store = watchdog_for(world, phone=phone)
    store.alert(minutes(0) - timedelta(days=2), "critical", "old news")
    store.alert(minutes(0), "critical", "trading halted: reconciliation gap $40.00")
    store.alert(minutes(1), "info", "trading resumed: halt cleared")

    phone.failures = 1
    watchdog.check(minutes(1))
    assert store.delivered == {}  # the second waits for the first, to keep them in order

    watchdog.check(minutes(2))
    watchdog.check(minutes(3))
    assert phone.received == [
        ("critical", "trading halted: reconciliation gap $40.00"),
        ("info", "trading resumed: halt cleared"),
    ]
    assert set(store.delivered) == {2, 3}


def test_the_webhook_posts_plain_text_with_a_priority():
    httpx = pytest.importorskip("httpx")
    from rebalancer.watchdog import WebhookNotifier

    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200 if len(seen) == 1 else 500)

    hook = WebhookNotifier(
        "https://ntfy.example/topic", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    hook.send("critical", "trading halted: x")
    assert seen[0].content == b"trading halted: x"
    assert seen[0].headers["Title"] == "rebalancer critical"
    assert seen[0].headers["Priority"] == "urgent"
    with pytest.raises(httpx.HTTPStatusError):
        hook.send("info", "second")


# --- the engine after the watchdog has cancelled its orders -----------------------------------


def test_the_engine_closes_an_order_cancelled_behind_its_back(resting):
    world, first = resting
    world.venues["alpaca"].cancel(first.order_id)
    second = next(o for o in world.step(minutes(1), PRICES).orders if o.instrument == "BTC-USD")
    assert first.status == "canceled"
    assert second.status == "placed"
    # The first order's turnover was given back; only the replacement counts.
    assert world.engine.gate.daily_turnover[T10.date()] == pytest.approx(second.notional)


def test_a_part_filled_order_cancelled_elsewhere_keeps_its_filled_share(growth):
    world = build_world(growth, T10, dict(PRICES), weights=BTC_HEAVY)
    venue = world.venues["alpaca"]
    venue.set_depth("BTC-USD", 0.01)
    first = next(o for o in world.step(T10, PRICES).orders if o.instrument == "BTC-USD")
    assert first.status == "partially_filled"

    venue.cancel(first.order_id)
    venue.set_depth("BTC-USD", 0.0)
    second = next(o for o in world.step(minutes(1), PRICES).orders if o.instrument == "BTC-USD")
    assert first.status == "canceled" and first.filled_qty == pytest.approx(0.01)
    kept = first.notional * first.filled_qty / first.qty
    assert world.engine.gate.daily_turnover[T10.date()] == pytest.approx(kept + second.notional)


def test_an_order_waits_for_fills_the_engine_has_not_seen_yet(resting):
    world, first = resting
    venue = world.venues["alpaca"]
    venue.cancel(first.order_id)
    # The broker says it traded half before the cancel, but that fill hasn't come through.
    venue.order_state = lambda order_id: OrderState("canceled", first.qty / 2)
    world.step(minutes(1), PRICES)
    assert first.status == "placed"


# --- against Postgres as the watchdog role, and the fake Alpaca --------------------------------


@pytest.fixture
def fake():
    return paper_account()


def watchdog_store(dsn):
    import psycopg

    from rebalancer.watchdog import PostgresWatchdogStore

    def connect():
        return psycopg.connect(dsn, autocommit=True, options="-c role=rebalancer_watchdog")

    return PostgresWatchdogStore(connect, "paper-main")


def refresh_quotes(fake):
    for symbol, (bid, ask) in list(fake.quotes.items()):
        fake.set_quote(symbol, bid, ask)


@pytest.fixture
def engine_with_a_resting_order(schema, dsn, fake, calendar):
    fake.fill_orders = False
    runner = alpaca_runner(dsn, fake, calendar)
    runner.engine.start(fake.now)
    runner.tick(fake.now)
    assert [o["status"] for o in fake.orders.values()] == ["new"]
    return runner


def alpaca_venue(fake, calendar):
    from fake_alpaca import KEY, SECRET

    from rebalancer.alpaca import AlpacaConfig, AlpacaVenue
    from rebalancer.replay import DEFAULT_INSTRUMENTS

    return AlpacaVenue(
        AlpacaConfig(KEY, SECRET),
        DEFAULT_INSTRUMENTS,
        calendar,
        client=fake.client(),
        clock=lambda: fake.now,
    )


def test_a_dead_engine_through_postgres_and_alpaca(
    engine_with_a_resting_order, schema, dsn, fake, calendar
):
    runner = engine_with_a_resting_order
    phone = Phone()
    watchdog = Watchdog(watchdog_store(dsn), alpaca_venue(fake, calendar), phone)

    fake.now += timedelta(minutes=6)
    watchdog.check(fake.now)
    assert [o["status"] for o in fake.orders.values()] == ["canceled"]
    level, message = phone.received[-1]
    assert level == "critical" and "cancelled 1 order" in message
    undelivered = schema.execute("select count(*) from alerts where delivered_at is null")
    assert undelivered.fetchone()[0] == 0

    # The engine comes back, finds its order gone and closes it.
    refresh_quotes(fake)
    runner.tick(fake.now)
    status, detail = schema.execute(
        """select o.status::text, e.detail from orders o join order_events e on e.order_id = o.id
           order by o.id, e.id desc limit 1"""
    ).fetchone()
    assert (status, detail) == ("canceled", "unfilled, canceled at the broker")
    watchdog.check(fake.now)
    assert [level for level, _ in phone.received] == ["critical", "info"]  # nothing sent twice
    assert phone.received[-1] == ("info", "engine heartbeat is back")


def test_a_watchdog_halt_stops_the_engine(engine_with_a_resting_order, schema, dsn, fake, calendar):
    runner = engine_with_a_resting_order
    account = schema.execute("select id from accounts").fetchone()[0]
    schema.execute(
        "insert into reconciliations (account_id, ts, gap_usd, ok) values (%s, %s, 40, false)",
        (account, fake.now),
    )
    watchdog = Watchdog(watchdog_store(dsn), alpaca_venue(fake, calendar), Phone())
    watchdog.check(fake.now)
    source = schema.execute("select source from halts where cleared_at is null").fetchone()
    assert source == ("watchdog",)

    fake.now += timedelta(minutes=1)
    refresh_quotes(fake)
    report = runner.tick(fake.now)
    assert "the engine had not halted" in report.halted


def test_the_watchdog_role_is_limited_and_reconnects(schema, dsn, growth, calendar):
    import psycopg

    from rebalancer.store import Account, PostgresStore

    PostgresStore.open(schema, growth, {}, account=Account("paper-main", "PA0001"))
    store = watchdog_store(dsn)
    assert store.last_heartbeat() is None
    store.conn.close()
    assert store.last_reconciliation() is None  # a fresh connection
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        store.conn.execute("update alerts set message = 'edited'")
    with pytest.raises(LookupError, match="no account named nobody"):
        type(store)(store._connect, "nobody")
