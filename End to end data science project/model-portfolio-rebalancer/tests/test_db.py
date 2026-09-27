"""Schema and migration tests against a real Postgres.

They need TEST_DATABASE_URL pointing at a server where the user can create databases and
roles; each test gets its own fresh database. Without it they skip, unless REQUIRE_DB_TESTS
is set (as in CI), where a missing database is a failure.
"""

import os
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import pytest

try:
    import psycopg
    from psycopg import errors, sql
    from psycopg.conninfo import make_conninfo
except ImportError:  # the simulator runs without the db extra installed
    psycopg = None

from rebalancer.migrate import Migration, MigrationError, available, migrate, status

ADMIN_DSN = os.environ.get("TEST_DATABASE_URL")
REQUIRED = bool(os.environ.get("REQUIRE_DB_TESTS"))

TABLES = {
    "model_versions",
    "model_sleeves",
    "accounts",
    "account_models",
    "instruments",
    "cycles",
    "cycle_prices",
    "sleeve_marks",
    "decisions",
    "orders",
    "order_events",
    "fills",
    "cash_flows",
    "reconciliations",
    "reconciliation_lines",
    "halts",
    "off_hours_needs",
    "alerts",
    "tax_lots",
    "lot_closures",
    "reference_prices",
    "corporate_actions",
    "schema_migrations",
}


@pytest.fixture
def db():
    if psycopg is None or not ADMIN_DSN:
        reason = "install the db extra and set TEST_DATABASE_URL to run database tests"
        if REQUIRED:
            pytest.fail(reason)
        pytest.skip(reason)
    name = f"rebalancer_test_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(ADMIN_DSN, autocommit=True) as admin:
        admin.execute(sql.SQL("create database {}").format(sql.Identifier(name)))
    conn = psycopg.connect(make_conninfo(ADMIN_DSN, dbname=name), autocommit=True)
    try:
        yield conn
    finally:
        conn.close()
        with psycopg.connect(ADMIN_DSN, autocommit=True) as admin:
            admin.execute(sql.SQL("drop database {} with (force)").format(sql.Identifier(name)))


@pytest.fixture
def schema(db):
    migrate(db)
    return db


@contextmanager
def as_role(conn, role):
    conn.execute(sql.SQL("set role {}").format(sql.Identifier(role)))
    try:
        yield
    finally:
        conn.execute("reset role")


def seed(conn):
    """One account, model, two instruments, a cycle, a decision and an open order."""
    conn.execute(
        """
        insert into model_versions (name, version, yaml, sha256, git_commit, cash_floor)
        values ('growth-247', 1, 'model: growth-247', 'abc', 'deadbeef', 0.02);
        insert into accounts (name, broker_account_id, mode, tax_treatment)
        values ('main', 'PA000', 'paper', 'taxable');
        insert into instruments (symbol, broker_symbol, asset_class, venue, lot_size, fractionable, refreshed_at)
        values ('VOO', 'VOO', 'equity', 'alpaca', 0.000001, true, now()),
               ('BTC-USD', 'BTC/USD', 'crypto', 'alpaca', 0.000001, true, now());
        insert into cycles (account_id, model_version_id, ts, session, account_value, engine_build)
        values (1, 1, now(), 'overnight', 100000, 'abc1234');
        insert into decisions (cycle_id, sleeve, action, detail)
        values (1, 'us_large_cap', 'trade', 'below band');
        insert into orders (client_order_id, account_id, cycle_id, decision_id, sleeve, instrument,
                            side, qty, order_type, time_in_force, limit_price, reference_price,
                            notional, session, reason, status, created_at, updated_at)
        values ('c1', 1, 1, 1, 'us_large_cap', 'VOO', 'buy', 1.5, 'limit', 'day', 561.12, 560,
                841.68, 'overnight', 'below band', 'accepted', now(), now());
        insert into alerts (ts, level, message) values (now(), 'warning', 'crypto buys paused');
        """
    )


ORDER = """
    insert into orders (client_order_id, account_id, cycle_id, sleeve, instrument, side, qty,
                        order_type, time_in_force, limit_price, reference_price, notional,
                        session, reason, status, created_at, updated_at)
    values (%s, 1, 1, 'us_large_cap', 'VOO', 'buy', %s, %s, 'day', %s, 560, 560, %s, 'x',
            'pending', now(), now())
"""


# --- migrations ----------------------------------------------------------------------------


def test_migration_files_are_numbered_from_one():
    migrations = available()
    assert [m.version for m in migrations] == list(range(1, len(migrations) + 1))
    assert migrations[0].name == "initial"


def test_migrate_creates_every_table(db):
    assert [m.version for m in migrate(db)] == [1]
    tables = {
        r[0] for r in db.execute("select tablename from pg_tables where schemaname = 'public'")
    }
    assert tables == TABLES
    assert status(db)[0][1] is not None


def test_second_run_applies_nothing(schema):
    assert migrate(schema) == []


def test_an_applied_migration_that_changed_is_refused(schema):
    first = available()[0]
    edited = Migration(first.version, first.name, first.sql + "\n-- tweak\n")
    with pytest.raises(MigrationError, match="changed after it was applied"):
        migrate(schema, [edited])


def test_a_failed_migration_leaves_nothing_behind(schema):
    broken = Migration(2, "broken", "create table half_done (id int); select * from missing_table;")
    with pytest.raises(errors.UndefinedTable):
        migrate(schema, available() + [broken])
    assert schema.execute("select to_regclass('half_done')").fetchone()[0] is None
    assert schema.execute("select max(version) from schema_migrations").fetchone()[0] == 1


def test_runner_needs_autocommit(db):
    db.autocommit = False
    with pytest.raises(MigrationError, match="autocommit"):
        migrate(db)


# --- constraints ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("statement", "params", "error"),
    [
        (ORDER, ("c2", 1, "market", None, "overnight"), "CheckViolation"),  # market order off-hours
        (ORDER, ("c3", 1, "limit", None, "regular"), "CheckViolation"),  # limit with no price
        (ORDER, ("c4", 0, "limit", 561, "regular"), "CheckViolation"),  # zero quantity
        (ORDER, ("c1", 1, "limit", 561, "regular"), "UniqueViolation"),  # reused client order id
        (
            "insert into halts (scope, reason, source) values ('lane', 'x', 'engine')",
            None,
            "CheckViolation",
        ),
        (
            "insert into halts (scope, reason, source) values ('daily', 'x', 'engine')",
            None,
            "CheckViolation",
        ),
        (
            "insert into halts (scope, lane, reason, source) values ('global', 'alpaca/equity', 'x', 'engine')",
            None,
            "CheckViolation",
        ),
        (
            "insert into instruments (symbol, broker_symbol, asset_class, venue, lot_size, fractionable, refreshed_at) values ('X', 'X', 'bond', 'alpaca', 1, true, now())",
            None,
            "InvalidTextRepresentation",
        ),
    ],
)
def test_constraints_reject_bad_rows(schema, statement, params, error):
    seed(schema)
    with pytest.raises(getattr(errors, error)):
        schema.execute(statement, params)


def test_valid_halts_are_accepted(schema):
    seed(schema)
    schema.execute(
        """
        insert into halts (account_id, scope, lane, reason, source)
        values (1, 'lane', 'alpaca/equity', '3 rejects', 'engine');
        insert into halts (scope, reason, source, until)
        values ('daily', 'turnover', 'engine', now() + interval '4 hours');
        insert into halts (scope, reason, source) values ('global', 'pressed halt', 'dashboard');
        """
    )
    assert schema.execute("select count(*) from halts where cleared_at is null").fetchone()[0] == 3


def test_tax_lot_cannot_have_more_open_than_bought(schema):
    seed(schema)
    schema.execute(
        """insert into fills (broker_fill_id, order_id, account_id, instrument, side, qty, price, ts)
           values ('f1', 1, 1, 'VOO', 'buy', 1.5, 561, now())"""
    )
    with pytest.raises(errors.CheckViolation):
        schema.execute(
            """insert into tax_lots (account_id, instrument, open_fill_id, acquired_at,
                                     holding_period_start, qty, qty_open, cost_per_unit)
               values (1, 'VOO', 1, now(), now(), 1.5, 2, 561)"""
        )


# --- roles ---------------------------------------------------------------------------------


def test_engine_cannot_rewrite_the_audit_trail(schema):
    seed(schema)
    with as_role(schema, "rebalancer_engine"):
        schema.execute("update orders set status = 'filled', updated_at = now() where id = 1")
        schema.execute(
            "insert into decisions (cycle_id, sleeve, action, detail) values (1, 'btc', 'ok', 'in band')"
        )
        for statement in (
            "update decisions set detail = 'rewritten' where id = 1",
            "delete from decisions where id = 1",
            "delete from orders where id = 1",
        ):
            with pytest.raises(errors.InsufficientPrivilege):
                schema.execute(statement)


def test_dashboard_can_only_halt_and_acknowledge(schema):
    seed(schema)
    with as_role(schema, "rebalancer_dashboard"):
        schema.execute(
            "insert into halts (scope, reason, source) values ('global', 'pressed halt', 'dashboard')"
        )
        schema.execute("update alerts set acknowledged_at = now() where id = 1")
        assert schema.execute("select count(*) from orders").fetchone()[0] == 1
        for statement in (
            "update alerts set message = 'hidden' where id = 1",
            "update orders set status = 'canceled' where id = 1",
            "update halts set cleared_at = now()",
            "insert into decisions (cycle_id, sleeve, action, detail) values (1, 'btc', 'ok', 'x')",
        ):
            with pytest.raises(errors.InsufficientPrivilege):
                schema.execute(statement)


def test_watchdog_can_raise_halts_but_not_trade(schema):
    seed(schema)
    with as_role(schema, "rebalancer_watchdog"):
        schema.execute(
            "insert into halts (scope, reason, source) values ('global', 'heartbeat lost', 'watchdog')"
        )
        schema.execute(
            "insert into alerts (ts, level, message) values (now(), 'critical', 'heartbeat lost')"
        )
        with pytest.raises(errors.InsufficientPrivilege):
            schema.execute(ORDER, ("w1", 1, "limit", 561, "regular"))


# --- retention -----------------------------------------------------------------------------


def test_old_cycle_detail_is_thinned_to_five_minutes(schema):
    seed(schema)
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    old_start = now.replace(minute=0) - timedelta(days=100)
    recent_start = now - timedelta(days=1)
    with schema.cursor() as cur:
        for start in (old_start, recent_start):
            for minute in range(20):
                ts = start + timedelta(minutes=minute)
                cycle_id = cur.execute(
                    """insert into cycles (account_id, model_version_id, ts, session, engine_build)
                       values (1, 1, %s, 'regular', 'abc') returning id""",
                    (ts,),
                ).fetchone()[0]
                cur.execute(
                    "insert into cycle_prices values (%s, 'VOO', 560, %s, true)", (cycle_id, ts)
                )
                cur.execute(
                    "insert into sleeve_marks values (%s, 'us_large_cap', 45000, 0.45, false)",
                    (cycle_id,),
                )

    with as_role(schema, "rebalancer_engine"):
        deleted = schema.execute("select prune_cycle_detail(as_of => %s)", (now,)).fetchone()[0]
    assert deleted == 2 * (20 - 4)

    counts = dict(
        schema.execute(
            """select c.ts < %s, count(*) from cycle_prices p join cycles c on c.id = p.cycle_id
               group by 1""",
            (now - timedelta(days=90),),
        ).fetchall()
    )
    assert counts == {True: 4, False: 20}
    # Cycles themselves are kept: decisions and orders still point at them.
    assert schema.execute("select count(*) from cycles").fetchone()[0] == 41


def test_only_the_engine_can_prune(schema):
    with as_role(schema, "rebalancer_dashboard"), pytest.raises(errors.InsufficientPrivilege):
        schema.execute("select prune_cycle_detail()")
