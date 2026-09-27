"""The engine's audit trail in Postgres. Uses the throwaway databases from conftest, and writes
as the rebalancer_engine role, so the grants in the schema are exercised too."""

import math
from datetime import timedelta

import pytest
from conftest import ADMIN_DSN, MODELS, PRICES, et, make_steps, ramp

from rebalancer.audit import AuditLog, ReconciliationLine
from rebalancer.engine import EngineConfig
from rebalancer.models import load_model
from rebalancer.replay import DEFAULT_INSTRUMENTS, build_world, main, run_replay

try:
    from psycopg.conninfo import make_conninfo

    from rebalancer.store import Account, ModelChangedError, PostgresStore
except ImportError:  # the simulator runs without the db extra installed
    pass

TUESDAY_10AM = et(2026, 11, 3, 10, 0)
BTC_HEAVY = {"us_large_cap": 0.42, "intl_equity": 0.15, "btc": 0.26, "eth": 0.10, "cash": 0.07}


@pytest.fixture
def engine_conn(schema):
    schema.execute("set role rebalancer_engine")
    return schema


def open_store(conn, model, name="main"):
    return PostgresStore.open(
        conn, model, DEFAULT_INSTRUMENTS, account=Account(name, f"PA-{name}"), engine_build="test"
    )


def recording_world(conn, model, weights=BTC_HEAVY, when=TUESDAY_10AM):
    audit = AuditLog(store=open_store(conn, model))
    return build_world(model, when, dict(PRICES), weights=weights, audit=audit)


def one(conn, query, *params):
    return conn.execute(query, params or None).fetchone()[0]


def test_a_cycle_with_a_trade_is_recorded_end_to_end(engine_conn, growth):
    conn = engine_conn
    w = recording_world(conn, growth)
    report = w.step(TUESDAY_10AM, PRICES)
    assert [o.status for o in report.orders] == ["filled"]

    assert one(conn, "select count(*) from cycles") == 1
    value, session, build = conn.execute(
        "select account_value, session, engine_build from cycles"
    ).fetchone()
    assert float(value) == pytest.approx(report.account_value, abs=0.01)
    assert (session, build) == ("regular", "test")
    assert one(conn, "select count(*) from cycle_prices where fresh") == 4
    assert one(conn, "select count(*) from sleeve_marks") == 5

    order = conn.execute(
        """select o.id, o.status, o.broker_order_id, d.action, d.detail
           from orders o join decisions d on d.id = o.decision_id"""
    ).fetchone()
    order_id, status, broker_id, action, detail = order
    assert (status, action) == ("filled", "trade")
    assert broker_id == report.orders[0].order_id
    assert detail.startswith("sell")

    events = [
        r[0]
        for r in conn.execute(
            "select status from order_events where order_id = %s order by id", (order_id,)
        )
    ]
    assert events == ["pending", "accepted", "filled"]

    fill_qty, fill_order = conn.execute("select qty, order_id from fills").fetchone()
    assert fill_order == order_id
    assert float(fill_qty) == pytest.approx(report.orders[0].qty)

    assert conn.execute("select gap_usd, ok from reconciliations").fetchone() == (0, True)
    assert one(conn, "select broker_symbol from instruments where symbol = 'BTC-USD'") == "BTC/USD"
    assert one(conn, "select yaml from model_versions") == (MODELS / "growth-247.yaml").read_text()
    assert one(conn, "select count(*) from account_models") == 1


def test_replay_trail_in_postgres_matches_memory(engine_conn, crypto_tilt, calendar):
    conn = engine_conn
    steps = make_steps(
        calendar,
        et(2026, 11, 13, 10, 0),
        et(2026, 11, 15, 12, 0),
        {
            "VOO": lambda ts: 560.0,
            "VXUS": lambda ts: 68.0,
            "BTC-USD": ramp(100_000, 80_000, et(2026, 11, 14, 2, 0), et(2026, 11, 14, 8, 0)),
            "ETH-USD": ramp(3_500, 2_800, et(2026, 11, 14, 2, 0), et(2026, 11, 14, 8, 0)),
        },
    )
    audit = AuditLog(store=open_store(conn, crypto_tilt))
    result = run_replay(crypto_tilt, steps, calendar=calendar, audit=audit)
    assert audit.store_error is None

    assert one(conn, "select count(*) from cycles") == len(result.snapshots)
    assert one(conn, "select count(*) from reconciliations where ok") == len(result.snapshots)
    assert one(conn, "select count(*) from decisions") == len(result.decisions)
    assert one(conn, "select count(*) from alerts") == len(result.alerts) > 0
    stored = dict(conn.execute("select client_order_id, status::text from orders").fetchall())
    in_memory = dict(zip(result.orders["client_id"], result.orders["status"], strict=True))
    assert stored == {k: {"placed": "accepted"}.get(v, v) for k, v in in_memory.items()}
    filled = (result.orders["status"] == "filled").sum()
    assert one(conn, "select count(*) from fills where order_id is not null") == filled > 0


def test_a_lost_database_halts_the_engine(engine_conn, growth):
    w = recording_world(engine_conn, growth)
    engine_conn.close()

    report = w.step(TUESDAY_10AM, PRICES)
    assert w.engine.audit.store_error
    assert report.halted and "audit store unavailable" in report.halted
    assert report.orders == []
    assert w.venues["alpaca"].fills(TUESDAY_10AM - timedelta(days=1)) == []
    assert any("audit store failed" in a.message for a in w.engine.audit.alerts)

    later = w.step(TUESDAY_10AM + timedelta(minutes=1), PRICES)
    assert later.halted and later.orders == []


def test_an_order_that_cannot_be_written_is_never_sent(engine_conn, growth, monkeypatch):
    w = recording_world(engine_conn, growth)

    def fail(record):
        raise RuntimeError("disk full")

    monkeypatch.setattr(w.engine.audit.store, "order_created", fail)
    report = w.step(TUESDAY_10AM, PRICES)
    assert [o.status for o in report.orders] == ["rejected"]
    assert "audit store unavailable" in report.orders[0].reject_reason
    assert w.venues["alpaca"].fills(TUESDAY_10AM - timedelta(days=1)) == []
    assert w.venues["alpaca"].find_order(report.orders[0].client_id) is None
    assert w.engine.gate.halted and "disk full" in w.engine.gate.halted


def test_a_model_edited_without_a_version_bump_is_refused(engine_conn, growth, tmp_path):
    open_store(engine_conn, growth)
    edited = tmp_path / "growth-247.yaml"
    edited.write_text(
        (MODELS / "growth-247.yaml")
        .read_text()
        .replace("0.45", "0.44", 1)
        .replace("0.15\n", "0.16\n", 1)
    )
    with pytest.raises(ModelChangedError, match="bump the version"):
        open_store(engine_conn, load_model(edited))


def test_a_models_sleeves_are_recorded_with_it(engine_conn, growth):
    open_store(engine_conn, growth)
    open_store(engine_conn, growth)  # a restart loads the same version again
    rows = engine_conn.execute(
        "select sleeve, target, band_abs, band_rel, instruments from model_sleeves"
    ).fetchall()
    assert {(r[0], float(r[1])) for r in rows} == {(s.id, s.target) for s in growth.sleeves}
    btc = next(r for r in rows if r[0] == "btc")
    assert (float(btc[2]), float(btc[3])) == (0.03, 0.15)
    assert btc[4] == {"any": ["BTC-USD"]}

    # A version recorded before sleeves were stored gets them when it's next loaded.
    engine_conn.execute("reset role")
    engine_conn.execute("delete from model_sleeves")
    engine_conn.execute("set role rebalancer_engine")
    open_store(engine_conn, growth)
    assert one(engine_conn, "select count(*) from model_sleeves") == len(growth.sleeves)


def test_a_new_model_version_is_recorded_as_a_change_for_the_account(engine_conn, growth, tmp_path):
    open_store(engine_conn, growth)
    bumped = tmp_path / "growth-247.yaml"
    bumped.write_text((MODELS / "growth-247.yaml").read_text().replace("version: 1", "version: 2"))
    open_store(engine_conn, load_model(bumped))
    versions = [
        r[0]
        for r in engine_conn.execute(
            """select m.version from account_models a join model_versions m on m.id = a.model_version_id
           order by a.effective_from"""
        )
    ]
    assert versions == [1, 2]


def test_reconciliation_gap_and_halt_are_recorded(engine_conn, growth):
    conn = engine_conn
    w = recording_world(conn, growth)
    w.account.holdings["BTC-USD"] += 0.001  # $100 the engine doesn't know about
    report = w.step(TUESDAY_10AM, PRICES)
    assert report.halted

    gap, ok = conn.execute("select gap_usd, ok from reconciliations").fetchone()
    assert not ok and float(gap) == pytest.approx(100, abs=0.5)
    line = conn.execute(
        "select instrument, engine_qty, broker_qty from reconciliation_lines"
    ).fetchone()
    assert line[0] == "BTC-USD" and float(line[2] - line[1]) == pytest.approx(0.001)
    scope, reason = conn.execute(
        "select scope::text, reason from halts where cleared_at is null"
    ).fetchone()
    assert scope == "global" and "reconciliation gap" in reason
    assert one(conn, "select halted from cycles") == reason
    assert one(conn, "select count(*) from orders") == 0

    w.engine.gate.clear_halt(TUESDAY_10AM + timedelta(minutes=5))
    assert one(conn, "select count(*) from halts where cleared_at is not null") == 1


def test_an_unpriced_mismatch_is_stored_without_a_dollar_gap(engine_conn, growth):
    conn = engine_conn
    store = open_store(conn, growth)
    audit = AuditLog(store=store)
    build_world(growth, TUESDAY_10AM, dict(PRICES), audit=audit).step(TUESDAY_10AM, PRICES)
    audit.reconciliation(
        TUESDAY_10AM, math.inf, False, [ReconciliationLine("XYZ", 0.0, 1.0, None, math.inf)]
    )
    assert audit.store_error is None
    gap, ok = conn.execute("select gap_usd, ok from reconciliations order by id desc").fetchone()
    assert gap is None and not ok
    assert one(conn, "select gap_usd from reconciliation_lines where instrument = 'XYZ'") is None


def test_lane_halts_and_deposits_are_recorded(engine_conn, growth):
    conn = engine_conn
    weights = {"us_large_cap": 0.38, "intl_equity": 0.15, "btc": 0.20, "eth": 0.10, "cash": 0.17}
    w = recording_world(conn, growth, weights=weights)
    w.venues["alpaca"].reject_next(10, "exchange unavailable", asset_class="equity")
    ts = TUESDAY_10AM
    for _ in range(3):
        w.step(ts, PRICES)
        ts += timedelta(minutes=1)
    assert conn.execute("select scope::text, lane from halts").fetchone() == (
        "lane",
        "alpaca/equity",
    )
    rejected = conn.execute(
        "select count(*) from order_events where status = 'rejected' and detail like 'venue:%'"
    ).fetchone()[0]
    assert rejected == 3

    w.engine.gate.clear_lane("alpaca/equity", ts)
    assert one(conn, "select cleared_by from halts") == "engine"

    w.deposit(ts, 5_000.0)
    assert conn.execute("select amount, kind from cash_flows").fetchone() == (5000, "deposit")


def test_replay_command_records_to_postgres(schema, tmp_path, calendar):
    steps = make_steps(
        calendar,
        et(2026, 11, 3, 10, 0),
        et(2026, 11, 4, 10, 0),
        {
            "VOO": lambda ts: 560.0,
            "VXUS": lambda ts: 68.0,
            "BTC-USD": ramp(100_000, 120_000, et(2026, 11, 3, 12, 0), et(2026, 11, 4, 0, 0)),
            "ETH-USD": lambda ts: 3_500.0,
        },
    )
    prices = tmp_path / "prices.csv"
    prices.write_text(
        "ts,instrument,price\n"
        + "".join(f"{ts.isoformat()},{i},{p}\n" for ts, row in steps for i, p in row.items())
    )
    dsn = make_conninfo(ADMIN_DSN, dbname=schema.info.dbname)
    main(
        [
            "--model",
            str(MODELS / "growth-247.yaml"),
            "--prices",
            str(prices),
            "--out",
            str(tmp_path / "out"),
            "--dsn",
            dsn,
        ]
    )
    assert one(schema, "select count(*) from accounts where name like 'replay-growth-247-%'") == 1
    assert one(schema, "select count(*) from cycles") == len(steps)
    assert one(schema, "select count(*) from orders where status = 'filled'") > 0


def test_client_order_ids_carry_the_prefix(engine_conn, growth):
    audit = AuditLog(store=open_store(engine_conn, growth))
    w = build_world(
        growth,
        TUESDAY_10AM,
        dict(PRICES),
        weights=BTC_HEAVY,
        audit=audit,
        config=EngineConfig(order_prefix="run7-"),
    )
    w.step(TUESDAY_10AM, PRICES)
    assert one(engine_conn, "select client_order_id from orders").startswith("run7-")
