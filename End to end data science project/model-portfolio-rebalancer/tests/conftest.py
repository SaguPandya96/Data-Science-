from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import pytest

try:
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo
except ImportError:  # the simulator runs without the db extra installed
    psycopg = None

from rebalancer.migrate import migrate
from rebalancer.models import load_model
from rebalancer.sessions import ET, MarketCalendar

# Database tests need a server where this user can create databases and roles. Without one they
# skip, unless REQUIRE_DB_TESTS is set (as in CI), where a missing database is a failure.
ADMIN_DSN = os.environ.get("TEST_DATABASE_URL")
REQUIRED = bool(os.environ.get("REQUIRE_DB_TESTS"))

MODELS = Path(__file__).resolve().parents[1] / "models"
PRICES = {"VOO": 560.0, "VXUS": 68.0, "BTC-USD": 100_000.0, "ETH-USD": 3_500.0}
EQUITIES = {"VOO", "VXUS"}


def et(*args: int) -> datetime:
    return datetime(*args, tzinfo=ET)


@pytest.fixture(scope="session")
def calendar() -> MarketCalendar:
    return MarketCalendar()


@pytest.fixture(scope="session")
def growth():
    return load_model(MODELS / "growth-247.yaml")


@pytest.fixture(scope="session")
def crypto_tilt():
    return load_model(MODELS / "crypto-tilt.yaml")


def make_steps(
    calendar: MarketCalendar,
    start: datetime,
    end: datetime,
    paths: dict[str, Callable[[datetime], float | None]],
    step: timedelta = timedelta(minutes=30),
) -> list[tuple[datetime, dict[str, float]]]:
    """Price steps where equities only print while their market is open. A path can return None
    to skip a print, which is how a test leaves a session without quotes."""
    steps = []
    ts = start
    while ts <= end:
        open_now = calendar.session_at(ts).session.equities_open
        prices = {}
        for inst, path in paths.items():
            if inst in EQUITIES and not open_now:
                continue
            price = path(ts)
            if price is not None:
                prices[inst] = price
        steps.append((ts, prices))
        ts += step
    return steps


def ramp(
    before: float, after: float, start: datetime, end: datetime
) -> Callable[[datetime], float]:
    def price(ts: datetime) -> float:
        if ts <= start:
            return before
        if ts >= end:
            return after
        return before + (after - before) * (ts - start) / (end - start)

    return price


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


@pytest.fixture
def dsn(schema):
    """Connection string for the test's database, for code that opens its own connections."""
    return make_conninfo(ADMIN_DSN, dbname=schema.info.dbname)
