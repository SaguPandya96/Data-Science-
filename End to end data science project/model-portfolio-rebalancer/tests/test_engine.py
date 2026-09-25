from datetime import timedelta

import pytest
from conftest import PRICES, et

from rebalancer.replay import build_world

TUESDAY_10AM = et(2026, 11, 3, 10, 0)
TUESDAY_11PM = et(2026, 11, 3, 23, 0)
SATURDAY_NOON = et(2026, 11, 7, 12, 0)


def weights(**w):
    base = {"us_large_cap": 0.45, "intl_equity": 0.15, "btc": 0.20, "eth": 0.10, "cash": 0.10}
    base.update(w)
    assert sum(base.values()) == pytest.approx(1.0)
    return base


def world(model, when, w, prices=PRICES, **kw):
    return build_world(model, when, dict(prices), weights=w, **kw)


def placed(report):
    return [o for o in report.orders if o.status != "rejected"]


def test_no_orders_when_everything_is_in_band(growth):
    w = world(growth, TUESDAY_10AM, weights())
    assert w.step(TUESDAY_10AM, PRICES).orders == []


def test_trades_back_to_the_band_edge_not_the_target(growth):
    w = world(
        growth,
        TUESDAY_10AM,
        weights(btc=0.26, us_large_cap=0.42, intl_equity=0.15, eth=0.10, cash=0.07),
    )
    report = w.step(TUESDAY_10AM, PRICES)
    orders = placed(report)
    assert [(o.instrument, o.side, o.status) for o in orders] == [("BTC-USD", "sell", "filled")]
    after = w.engine.snapshot(TUESDAY_10AM).marks["btc"].weight
    assert after == pytest.approx(0.23, abs=0.001)


def test_equity_sleeve_is_frozen_on_the_weekend(growth):
    w = world(
        growth,
        SATURDAY_NOON,
        weights(us_large_cap=0.38, cash=0.17),
        price_times={"VOO": et(2026, 11, 6, 19, 59), "VXUS": et(2026, 11, 6, 19, 59)},
    )
    report = w.step(SATURDAY_NOON, {"BTC-USD": 100_000.0, "ETH-USD": 3_500.0})
    assert not any(o.instrument in {"VOO", "VXUS"} for o in report.orders)
    assert report.marks["us_large_cap"].stale
    holds = [
        d for d in w.engine.audit.decisions if d.sleeve == "us_large_cap" and d.action == "hold"
    ]
    assert holds and "frozen" in holds[0].detail


def test_stale_quote_skips_the_sleeve(growth):
    w = world(
        growth,
        TUESDAY_10AM,
        weights(us_large_cap=0.38, cash=0.17),
        price_times={"VOO": TUESDAY_10AM - timedelta(minutes=5)},
    )
    report = w.step(TUESDAY_10AM, {"VXUS": 68.0, "BTC-USD": 100_000.0, "ETH-USD": 3_500.0})
    assert not any(o.instrument == "VOO" for o in report.orders)
    holds = [
        d for d in w.engine.audit.decisions if d.sleeve == "us_large_cap" and d.action == "hold"
    ]
    assert "stale quote" in holds[0].detail


def test_trades_under_25_dollars_are_skipped(growth):
    # 0.02 points below the band on $100k is $20.
    w = world(growth, TUESDAY_10AM, weights(eth=0.0798, cash=0.1202))
    report = w.step(TUESDAY_10AM, PRICES)
    assert report.orders == []
    skips = [d for d in w.engine.audit.decisions if d.sleeve == "eth" and d.action == "skip"]
    assert skips and "under $25" in skips[0].detail


def test_off_hours_equity_trade_is_capped_at_a_quarter_of_the_need(growth):
    w = world(
        growth,
        TUESDAY_11PM,
        weights(us_large_cap=0.38, intl_equity=0.17, btc=0.22, eth=0.11, cash=0.12),
    )
    need = (0.42 - 0.38) * 100_000
    bought = 0.0
    ts = TUESDAY_11PM
    while ts < et(2026, 11, 4, 4, 0):
        for o in placed(w.step(ts, PRICES)):
            if o.instrument == "VOO":
                assert o.session == "overnight"
                bought += o.notional
        ts += timedelta(minutes=30)
    assert 0 < bought <= 0.25 * need + 1e-6

    # The rest waits for the regular session, where it goes at full size.
    regular = placed(w.step(et(2026, 11, 4, 9, 30), PRICES))
    voo = [o for o in regular if o.instrument == "VOO"]
    assert voo and voo[0].notional > 0.25 * need


def test_deposit_is_spent_on_underweights_before_anything_is_sold(growth):
    w = world(growth, TUESDAY_10AM, weights())
    w.deposit(TUESDAY_10AM, 20_000.0)
    report = w.step(TUESDAY_10AM, PRICES)
    sides = {o.side for o in placed(report)}
    assert sides == {"buy"}
    assert report.marks["cash"].weight > 0.02


def test_withdrawal_below_the_floor_sells_overweights(growth):
    w = world(
        growth,
        TUESDAY_10AM,
        weights(us_large_cap=0.48, intl_equity=0.17, btc=0.23, eth=0.11, cash=0.01),
    )
    report = w.step(TUESDAY_10AM, PRICES)
    orders = placed(report)
    assert orders and {o.side for o in orders} == {"sell"}
    raised = sum(o.notional for o in orders)
    assert 4_500 < raised <= (0.07 - 0.01) * 100_000


def test_buys_only_spend_cash_above_the_floor(growth):
    w = world(
        growth,
        TUESDAY_10AM,
        weights(us_large_cap=0.38, intl_equity=0.20, btc=0.27, eth=0.12, cash=0.03),
    )
    before = w.engine.snapshot(TUESDAY_10AM)
    spendable = before.marks["cash"].value - 0.02 * before.account_value
    report = w.step(TUESDAY_10AM, PRICES)
    buys = [o for o in placed(report) if o.side == "buy"]
    assert buys and sum(o.notional for o in buys) <= spendable
    assert w.engine.snapshot(TUESDAY_10AM).marks["cash"].weight >= 0.02


def test_single_order_is_capped_at_five_percent(growth):
    w = world(growth, TUESDAY_10AM, weights(us_large_cap=0.30, cash=0.25))
    report = w.step(TUESDAY_10AM, PRICES)
    for o in placed(report):
        assert o.notional <= 0.05 * 100_000 + 1e-6


def test_orders_are_limits_priced_near_the_quote(growth):
    w = world(
        growth,
        TUESDAY_11PM,
        weights(us_large_cap=0.38, intl_equity=0.17, btc=0.26, eth=0.07, cash=0.12),
    )
    report = w.step(TUESDAY_11PM, PRICES)
    assert report.orders
    for o in report.orders:
        assert o.order_type == "limit"
        assert abs(o.limit_price / o.reference_price - 1) <= 0.02
