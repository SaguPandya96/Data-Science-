from datetime import timedelta

import pytest
from conftest import MODELS, PRICES, et

from rebalancer.models import load_model
from rebalancer.replay import build_world
from rebalancer.runner import Runner, RunnerConfig, main

T10 = et(2026, 11, 3, 10, 0)
BTC_HEAVY = {"us_large_cap": 0.42, "intl_equity": 0.15, "btc": 0.26, "eth": 0.10, "cash": 0.07}


class FakeTime:
    """Clock and wait for the runner: waiting moves time on and refreshes the simulated quotes."""

    def __init__(self, world, start):
        self.world, self.now = world, start

    def clock(self):
        return self.now

    def wait(self, seconds):
        self.now += timedelta(seconds=seconds)
        self.world.set_prices(self.now, PRICES)


def runner_for(world, start, **config):
    time = FakeTime(world, start)
    world.set_prices(start, PRICES)
    runner = Runner(world.engine, config=RunnerConfig(**config), clock=time.clock, wait=time.wait)
    return runner, time


def record_ticks(runner):
    ticks, cycle = [], runner.engine.cycle

    def recording(now):
        ticks.append(now)
        return cycle(now)

    runner.engine.cycle = recording
    return ticks


@pytest.fixture
def growth():
    return load_model(MODELS / "growth-247.yaml")


def test_cycles_land_on_the_minute(growth):
    start = T10 + timedelta(seconds=20)
    world = build_world(growth, start, dict(PRICES))
    runner, _ = runner_for(world, start)
    ticks = record_ticks(runner)
    assert runner.run(max_cycles=3) == 3
    assert ticks == [start, T10 + timedelta(minutes=1), T10 + timedelta(minutes=2)]


def test_a_slow_cycle_skips_ahead_instead_of_catching_up(growth):
    world = build_world(growth, T10, dict(PRICES))
    runner, time = runner_for(world, T10)
    ticks, cycle = [], world.engine.cycle

    def slow(now):
        ticks.append(now)
        report = cycle(now)
        time.now += timedelta(seconds=150)
        return report

    world.engine.cycle = slow
    runner.run(max_cycles=2)
    assert ticks == [T10, T10 + timedelta(minutes=3)]


def test_failed_cycles_alert_and_three_in_a_row_halt(growth):
    world = build_world(growth, T10, dict(PRICES))
    runner, _ = runner_for(world, T10)
    world.engine.start(T10)
    real = world.engine.cycle
    outcomes = iter(["fail", "fail", "ok", "fail", "fail", "fail"])

    def flaky(now):
        if next(outcomes) == "fail":
            raise RuntimeError("broker timed out")
        return real(now)

    world.engine.cycle = flaky
    for minute in range(5):
        runner.tick(T10 + timedelta(minutes=minute))
    assert world.engine.gate.halted is None  # the success in the middle reset the count
    runner.tick(T10 + timedelta(minutes=5))
    assert "3 cycles in a row failed" in world.engine.gate.halted
    alerts = [a.message for a in world.engine.audit.alerts]
    assert sum("cycle failed" in m for m in alerts) == 5


def test_stopping_cancels_whatever_is_resting(growth):
    world = build_world(growth, T10, dict(PRICES), weights=BTC_HEAVY)
    world.venues["alpaca"].fills_paused = True
    runner, time = runner_for(world, T10)

    def wait_then_stop(seconds):
        time.wait(seconds)
        runner.stop()

    runner._wait = wait_then_stop
    assert runner.run() == 1
    order = world.engine.audit.orders[0]
    assert order.status == "canceled"
    assert world.venues["alpaca"].find_order(order.client_id) is not None


def test_the_command_needs_a_database(capsys, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(SystemExit, match="needs Postgres"):
        main(
            [
                "--model",
                str(MODELS / "growth-247.yaml"),
                "--account",
                "a",
                "--broker-account",
                "PA1",
            ]
        )


# --- against the fake Alpaca and a real Postgres -----------------------------------------------


def alpaca_runner(dsn, fake, calendar, account="paper-main"):
    from fake_alpaca import KEY, SECRET

    from rebalancer.alpaca import AlpacaConfig, AlpacaVenue
    from rebalancer.replay import DEFAULT_INSTRUMENTS
    from rebalancer.runner import build

    def venue(instruments, cal):
        return AlpacaVenue(
            AlpacaConfig(KEY, SECRET),
            instruments,
            cal,
            client=fake.client(),
            clock=lambda: fake.now,
        )

    engine = build(
        load_model(MODELS / "growth-247.yaml"),
        dsn=dsn,
        account_name=account,
        broker_account="PA0001",
        venue_factory=venue,
        instruments=DEFAULT_INSTRUMENTS,
        calendar=calendar,
    )

    def wait(seconds):
        fake.now += timedelta(seconds=seconds)
        for symbol, (bid, ask) in list(fake.quotes.items()):
            fake.set_quote(symbol, bid, ask)

    return Runner(engine, clock=lambda: fake.now, wait=wait)


def paper_account():
    """The fake Alpaca holding roughly the Growth model, BTC a little heavy."""
    pytest.importorskip("httpx")
    from fake_alpaca import FakeAlpaca

    server = FakeAlpaca(T10, cash=7_000.0)
    server.positions = {"VOO": 75.0, "VXUS": 220.588, "BTCUSD": 0.26, "ETHUSD": 2.857}
    for symbol, (bid, ask) in {
        "VOO": (559.9, 560.1),
        "VXUS": (67.99, 68.01),
        "BTC/USD": (99_990.0, 100_010.0),
        "ETH/USD": (3_499.0, 3_501.0),
    }.items():
        server.set_quote(symbol, bid, ask)
    return server


@pytest.fixture
def fake():
    return paper_account()


def test_the_runner_trades_through_alpaca_and_records_every_cycle(schema, dsn, fake, calendar):
    runner = alpaca_runner(dsn, fake, calendar)
    assert runner.run(max_cycles=3) == 3
    assert schema.execute("select count(*) from cycles").fetchone()[0] == 3
    status, symbol = schema.execute("select status::text, instrument from orders").fetchone()
    assert (status, symbol) == ("filled", "BTC-USD")
    assert schema.execute("select count(*) from reconciliations where not ok").fetchone()[0] == 0


def test_a_second_runner_on_the_same_account_is_refused(schema, dsn, fake, calendar):
    first = alpaca_runner(dsn, fake, calendar)
    with pytest.raises(SystemExit, match="already being run"):
        alpaca_runner(dsn, fake, calendar)
    # Once the first one's connection closes, the account is free again.
    first.engine.audit.store.conn.close()
    alpaca_runner(dsn, fake, calendar)
