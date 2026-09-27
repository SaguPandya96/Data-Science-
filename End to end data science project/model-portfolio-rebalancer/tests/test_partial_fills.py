"""Orders that fill in pieces: in memory, in the store, and across a restart."""

from datetime import timedelta

import pytest
from conftest import PRICES, et
from test_recovery import BTC_HEAVY, T10, events, one, restart, start_world

from rebalancer.replay import build_world

NEXT = T10 + timedelta(minutes=1)


def btc_order(report):
    return next(o for o in report.orders if o.instrument == "BTC-USD")


def test_a_partial_fill_updates_positions_and_the_rest_fills_later(growth):
    w = build_world(growth, T10, dict(PRICES), weights=BTC_HEAVY)
    venue = w.venues["alpaca"]
    venue.set_depth("BTC-USD", 0.01)

    order = btc_order(w.step(T10, PRICES))
    assert order.status == "partially_filled"
    assert order.filled_qty == pytest.approx(0.01) and order.qty > 0.02
    assert w.engine.ledger["BTC-USD"] == pytest.approx(w.account.holdings["BTC-USD"])

    venue.set_depth("BTC-USD", None)
    report = w.step(NEXT, PRICES)
    assert report.halted is None and report.orders == []
    assert order.status == "filled"
    assert order.filled_qty == pytest.approx(order.qty)
    assert w.engine.gate.daily_turnover[T10.date()] == pytest.approx(order.notional)


def test_cancelling_a_part_filled_order_gives_back_only_the_unfilled_turnover(growth):
    w = build_world(growth, T10, dict(PRICES), weights=BTC_HEAVY)
    venue = w.venues["alpaca"]
    venue.set_depth("BTC-USD", 0.01)
    first = btc_order(w.step(T10, PRICES))

    venue.set_depth("BTC-USD", 0.0)
    report = w.step(NEXT, PRICES)
    assert first.status == "canceled" and first.filled_qty == pytest.approx(0.01)
    second = btc_order(report)
    # The replacement only sells what the first one didn't.
    assert second.qty == pytest.approx(first.qty - 0.01, rel=0.02)
    kept = first.notional * first.filled_qty / first.qty
    assert w.engine.gate.daily_turnover[T10.date()] == pytest.approx(kept + second.notional)


def test_a_part_filled_overnight_order_keeps_its_filled_share_against_the_cap(growth):
    night = et(2026, 11, 3, 23, 0)
    weights = {"us_large_cap": 0.38, "intl_equity": 0.17, "btc": 0.22, "eth": 0.11, "cash": 0.12}
    w = build_world(growth, night, dict(PRICES), weights=weights)
    venue = w.venues["alpaca"]
    venue.set_depth("VOO", 0.5)
    first = next(o for o in w.step(night, PRICES).orders if o.instrument == "VOO")
    assert first.status == "partially_filled"
    cap = 0.25 * 4_000

    venue.set_depth("VOO", 0.0)
    second = next(
        o for o in w.step(night + timedelta(minutes=30), PRICES).orders if o.instrument == "VOO"
    )
    filled_first = first.notional * first.filled_qty / first.qty
    assert first.status == "canceled"
    assert filled_first + second.notional == pytest.approx(cap, abs=1.0)


# --- in the store ----------------------------------------------------------------------------


def test_partial_fills_are_recorded_as_they_happen(schema, dsn, growth):
    w = start_world(dsn, growth, T10, BTC_HEAVY)
    venue = w.venues["alpaca"]
    venue.set_depth("BTC-USD", 0.01)
    order = btc_order(w.step(T10, PRICES))
    venue.set_depth("BTC-USD", None)
    w.step(NEXT, PRICES)

    assert events(schema, order.client_id) == ["pending", "accepted", "partially_filled", "filled"]
    assert one(schema, "select status::text from orders") == "filled"
    qty, count = schema.execute(
        "select sum(qty), count(*) from fills where order_id is not null"
    ).fetchone()
    assert (float(qty), count) == (pytest.approx(order.qty), 2)


def test_a_part_filled_order_finishes_filling_while_the_engine_is_down(schema, dsn, growth):
    w = start_world(dsn, growth, T10, BTC_HEAVY)
    venue = w.venues["alpaca"]
    venue.set_depth("BTC-USD", 0.01)
    order = btc_order(w.step(T10, PRICES))

    # Down: the book refills and the rest of the order trades.
    venue.set_depth("BTC-USD", None)
    w.clock.now = T10 + timedelta(seconds=30)
    venue.set_price("BTC-USD", PRICES["BTC-USD"], w.clock.now)

    restart(w, dsn, NEXT)
    # The first fill was already recorded before the crash and must not be applied again.
    assert events(schema, order.client_id) == ["pending", "accepted", "partially_filled", "filled"]
    assert sum(w.engine.gate.daily_turnover.values()) == pytest.approx(order.notional)

    report = w.step(NEXT, PRICES)
    assert report.halted is None and report.orders == []


def test_turnover_of_a_cancelled_part_fill_survives_a_restart(schema, dsn, growth):
    w = start_world(dsn, growth, T10, BTC_HEAVY)
    venue = w.venues["alpaca"]
    venue.set_depth("BTC-USD", 0.01)
    w.step(T10, PRICES)
    venue.set_depth("BTC-USD", 0.0)
    w.step(NEXT, PRICES)  # cancels the rest and re-quotes it, which then rests
    turnover = sum(w.engine.gate.daily_turnover.values())

    restart(w, dsn, NEXT + timedelta(minutes=1))
    assert sum(w.engine.gate.daily_turnover.values()) == pytest.approx(turnover)
