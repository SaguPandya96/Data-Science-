"""Restart recovery: kill the engine at awkward moments, start a new one on the same broker
account from what the store holds, and check it carries on as if nothing happened."""

from datetime import timedelta

import pytest
from conftest import ADMIN_DSN, PRICES, et, make_steps, ramp

from rebalancer.audit import AuditLog
from rebalancer.engine import Engine
from rebalancer.replay import DEFAULT_INSTRUMENTS, build_world

try:
    import psycopg
    from psycopg.conninfo import make_conninfo

    from rebalancer.store import Account, PostgresStore
except ImportError:  # the simulator runs without the db extra installed
    pass

T10 = et(2026, 11, 3, 10, 0)
BTC_HEAVY = {"us_large_cap": 0.42, "intl_equity": 0.15, "btc": 0.26, "eth": 0.10, "cash": 0.07}


@pytest.fixture
def dsn(schema):
    return make_conninfo(ADMIN_DSN, dbname=schema.info.dbname)


def engine_store(dsn, model):
    conn = psycopg.connect(dsn, autocommit=True)
    conn.execute("set role rebalancer_engine")
    return PostgresStore.open(
        conn, model, DEFAULT_INSTRUMENTS, account=Account("main", "PA-main"), engine_build="test"
    )


def start_world(dsn, model, when, weights=None):
    audit = AuditLog(store=engine_store(dsn, model))
    return build_world(model, when, dict(PRICES), weights=weights, audit=audit)


def restart(world, dsn, now):
    """Drop the engine as if its process died and start a fresh one on the same account."""
    old = world.engine
    old.audit.store.conn.close()
    world.clock.now = now
    world.engine = Engine(
        world.model,
        world.venues,
        world.instruments,
        old.calendar,
        limits=old.gate.limits,
        config=old.config,
        audit=AuditLog(store=engine_store(dsn, world.model)),
    )
    world.engine.start(now)


def events(db, client_id):
    return [
        r[0]
        for r in db.execute(
            """select e.status::text from order_events e join orders o on o.id = e.order_id
           where o.client_order_id = %s order by e.id""",
            (client_id,),
        )
    ]


def one(db, query, *params):
    return db.execute(query, params or None).fetchone()[0]


def test_a_resting_order_is_picked_up_and_requoted(schema, dsn, growth):
    w = start_world(dsn, growth, T10, BTC_HEAVY)
    w.venues["alpaca"].fills_paused = True
    first = w.step(T10, PRICES).orders[0]
    assert first.status == "placed"
    turnover = dict(w.engine.gate.daily_turnover)

    restart(w, dsn, T10 + timedelta(minutes=1))
    assert w.engine.gate.daily_turnover == pytest.approx(turnover)

    second = w.step(T10 + timedelta(minutes=1), PRICES).orders
    assert len(second) == 1 and second[0].client_id != first.client_id
    assert events(schema, first.client_id) == ["pending", "accepted", "canceled"]
    assert (
        one(
            schema,
            "select count(*) from alerts where message like 'resumed after a restart%%1 open orders%%'",
        )
        == 1
    )


def test_a_fill_while_the_engine_was_down_is_recorded_once(schema, dsn, growth):
    w = start_world(dsn, growth, T10, BTC_HEAVY)
    venue = w.venues["alpaca"]
    venue.fills_paused = True
    first = w.step(T10, PRICES).orders[0]

    # The engine is down; the book comes back and the resting sell fills.
    venue.fills_paused = False
    w.clock.now = T10 + timedelta(seconds=30)
    venue.set_price("BTC-USD", PRICES["BTC-USD"], w.clock.now)

    restart(w, dsn, T10 + timedelta(minutes=1))
    assert events(schema, first.client_id) == ["pending", "accepted", "filled"]
    assert one(schema, "select count(*) from fills where order_id is not null") == 1

    report = w.step(T10 + timedelta(minutes=1), PRICES)
    assert report.halted is None  # the fill isn't counted twice, so reconciliation is clean
    assert report.orders == []
    assert w.engine.ledger["BTC-USD"] == pytest.approx(w.account.holdings["BTC-USD"])


def test_an_order_written_but_never_sent_is_closed_out(schema, dsn, growth):
    w = start_world(dsn, growth, T10, BTC_HEAVY)
    venue = w.venues["alpaca"]

    def die(order):
        raise RuntimeError("process killed")

    venue.place = die
    with pytest.raises(RuntimeError, match="process killed"):
        w.step(T10, PRICES)
    del venue.place
    client_id = one(schema, "select client_order_id from orders")
    assert events(schema, client_id) == ["pending"]

    restart(w, dsn, T10 + timedelta(minutes=1))
    assert events(schema, client_id) == ["pending", "rejected"]
    assert "never reached the broker" in one(schema, "select reject_reason from orders")
    assert sum(w.engine.gate.daily_turnover.values()) == 0

    retry = w.step(T10 + timedelta(minutes=1), PRICES).orders
    assert [o.status for o in retry] == ["filled"]


def test_an_order_sent_just_before_a_crash_is_found_at_the_broker(schema, dsn, growth):
    w = start_world(dsn, growth, T10, BTC_HEAVY)
    venue = w.venues["alpaca"]
    real_place = venue.place

    def send_then_die(order):
        real_place(order)
        raise RuntimeError("process killed")

    venue.place = send_then_die
    with pytest.raises(RuntimeError):
        w.step(T10, PRICES)
    del venue.place
    client_id = one(schema, "select client_order_id from orders")

    restart(w, dsn, T10 + timedelta(minutes=1))
    assert events(schema, client_id) == ["pending", "accepted", "filled"]
    notional = float(one(schema, "select notional from orders"))
    assert sum(w.engine.gate.daily_turnover.values()) == pytest.approx(notional)

    report = w.step(T10 + timedelta(minutes=1), PRICES)
    assert report.halted is None and report.orders == []


def test_a_dashboard_halt_stops_trading_and_clearing_it_resumes(schema, dsn, growth):
    w = start_world(dsn, growth, T10, BTC_HEAVY)
    w.venues["alpaca"].fills_paused = True
    resting = w.step(T10, PRICES).orders[0]

    schema.execute("set role rebalancer_dashboard")
    schema.execute(
        "insert into halts (scope, reason, source) values ('global', 'pressed halt', 'dashboard')"
    )
    schema.execute("reset role")

    halted = w.step(T10 + timedelta(minutes=1), PRICES)
    assert halted.halted == "pressed halt" and halted.orders == []
    assert events(schema, resting.client_id)[-1] == "canceled"
    assert any(a.message == "trading halted: pressed halt" for a in w.engine.audit.alerts)

    schema.execute("update halts set cleared_at = now(), cleared_by = 'me'")
    resumed = w.step(T10 + timedelta(minutes=2), PRICES)
    assert resumed.halted is None and len(resumed.orders) == 1
    assert any(a.message == "trading resumed: halt cleared" for a in w.engine.audit.alerts)


def test_halts_survive_a_restart(schema, dsn, growth):
    w = start_world(dsn, growth, T10, BTC_HEAVY)
    w.account.holdings["BTC-USD"] += 0.001  # a $100 gap: global halt
    assert w.step(T10, PRICES).halted
    w.engine.gate.halt_lane(T10, "alpaca/equity", "3 rejected orders in 10 min")

    restart(w, dsn, T10 + timedelta(minutes=1))
    assert "alpaca/equity" in w.engine.gate.lane_halts
    report = w.step(T10 + timedelta(minutes=1), PRICES)
    assert report.halted and "reconciliation gap" in report.halted
    assert report.orders == []
    assert one(schema, "select count(*) from halts where cleared_at is null") == 2


def test_the_overnight_cap_and_budget_carry_across_a_restart(schema, dsn, growth):
    night = et(2026, 11, 3, 23, 0)
    weights = {"us_large_cap": 0.38, "intl_equity": 0.17, "btc": 0.22, "eth": 0.11, "cash": 0.12}
    w = start_world(dsn, growth, night, weights)
    first = [o for o in w.step(night, PRICES).orders if o.instrument == "VOO"]
    assert first and first[0].status == "filled"
    budget = dict(w.engine.gate.off_hours_turnover)
    need = float(one(schema, "select need from off_hours_needs where sleeve = 'us_large_cap'"))
    assert need == pytest.approx(4_000, rel=0.01)

    restart(w, dsn, night + timedelta(minutes=30))
    assert w.engine.gate.off_hours_turnover == pytest.approx(budget)
    window = et(2026, 11, 3, 16, 0)
    assert w.engine._off_need[(window, "us_large_cap")] == pytest.approx(need)
    assert w.engine._off_done[(window, "us_large_cap")] == pytest.approx(first[0].notional)
    later = []
    ts = night + timedelta(minutes=30)
    while ts < et(2026, 11, 4, 4, 0):
        later += [o for o in w.step(ts, PRICES).orders if o.instrument == "VOO"]
        ts += timedelta(minutes=30)
    # What was bought before the restart still counts against tonight's 25%.
    assert later == []


def test_the_crypto_drawdown_pause_carries_across_a_restart(schema, dsn, crypto_tilt, calendar):
    start = et(2026, 11, 14, 0, 0)
    steps = make_steps(
        calendar,
        start,
        et(2026, 11, 14, 9, 0),
        {
            "VOO": lambda ts: 560.0,
            "VXUS": lambda ts: 68.0,
            "BTC-USD": ramp(100_000, 80_000, et(2026, 11, 14, 2, 0), et(2026, 11, 14, 8, 0)),
            "ETH-USD": ramp(3_500, 2_800, et(2026, 11, 14, 2, 0), et(2026, 11, 14, 8, 0)),
        },
    )
    w = start_world(dsn, crypto_tilt, start)
    for ts, prices in steps[:-1]:
        w.step(ts, prices)
    assert w.engine.gate._pause_reason

    last_ts, last_prices = steps[-1]
    restart(w, dsn, last_ts)
    assert w.engine.gate.crypto_buys_paused(last_ts)
    report = w.step(last_ts, last_prices)
    assert not any(o.side == "buy" and o.instrument.endswith("-USD") for o in report.orders)
    assert one(schema, "select count(*) from alerts where message like 'crypto buys paused%%'") == 1


def test_a_fresh_account_starts_clean(schema, dsn, growth):
    w = start_world(dsn, growth, T10, BTC_HEAVY)
    assert one(schema, "select count(*) from alerts") == 0
    assert w.step(T10, PRICES).orders[0].status == "filled"
