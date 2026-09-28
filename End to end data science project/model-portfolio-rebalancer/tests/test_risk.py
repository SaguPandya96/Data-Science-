from datetime import timedelta

import pytest
from conftest import PRICES, et

from rebalancer.audit import AuditLog
from rebalancer.replay import DEFAULT_INSTRUMENTS, build_world
from rebalancer.risk import RiskGate, RiskLimits
from rebalancer.sessions import Session
from rebalancer.venue import Order, OrderType, Quote, SessionStatus, Side, TimeInForce

NOW = et(2026, 11, 3, 10, 0)
NIGHT = et(2026, 11, 3, 23, 0)
VALUE = 100_000.0
OPEN = SessionStatus(True, "regular", frozenset(OrderType), frozenset(TimeInForce))
OVERNIGHT = SessionStatus(
    True, "overnight", frozenset({OrderType.LIMIT}), frozenset({TimeInForce.DAY, TimeInForce.GTC})
)


@pytest.fixture
def gate(calendar):
    return RiskGate(RiskLimits(), calendar, DEFAULT_INSTRUMENTS, AuditLog())


def quote(inst="VOO", mid=100.0, ts=NOW):
    return Quote(inst, mid * 0.9999, mid * 1.0001, 1e6, 1e6, ts)


def order(qty=10.0, limit=100.0, inst="VOO", side=Side.BUY, kind=OrderType.LIMIT):
    return Order(
        "c1", inst, side, qty, limit if kind is OrderType.LIMIT else None, kind, TimeInForce.DAY
    )


def check(gate, o, *, now=NOW, q=None, status=OPEN, session=Session.REGULAR):
    return gate.check(
        o,
        now=now,
        quote=q or quote(o.instrument, ts=now),
        status=status,
        session=session,
        account_value=VALUE,
        min_trade=25.0,
    )


def test_accepts_a_normal_order(gate):
    assert check(gate, order()).approved


def test_rejects_an_order_over_five_percent(gate):
    result = check(gate, order(qty=51))
    assert not result.approved and "single-order limit" in result.reason


def test_rejects_a_limit_more_than_two_percent_from_reference(gate):
    assert check(gate, order(limit=101.9)).approved
    result = check(gate, order(limit=102.1))
    assert not result.approved and "from reference" in result.reason


def test_rejects_a_stale_quote(gate):
    result = check(gate, order(), q=quote(ts=NOW - timedelta(seconds=61)))
    assert not result.approved and "stale quote" in result.reason


def test_crypto_quotes_may_be_older_than_equity_quotes(gate):
    btc = order(inst="BTC-USD", qty=0.01, limit=100.0)
    four_minutes = NOW - timedelta(minutes=4)
    assert check(gate, btc, q=quote("BTC-USD", ts=four_minutes)).approved
    assert not check(gate, order(), q=quote(ts=four_minutes)).approved
    result = check(gate, btc, q=quote("BTC-USD", ts=NOW - timedelta(minutes=5, seconds=1)))
    assert not result.approved and "stale quote (301s old)" in result.reason


def test_no_market_orders_outside_the_regular_session(gate):
    assert check(gate, order(kind=OrderType.MARKET)).approved
    all_types = SessionStatus(True, "overnight", frozenset(OrderType), frozenset(TimeInForce))
    result = check(
        gate, order(kind=OrderType.MARKET), now=NIGHT, status=all_types, session=Session.OVERNIGHT
    )
    assert not result.approved and "market orders not allowed" in result.reason


def test_daily_turnover_clips_then_halts_until_tomorrow(gate):
    gate.record_placed("a", NOW, Session.REGULAR, 14_900.0)
    clipped = check(gate, order(qty=40))
    assert clipped.approved and clipped.order.qty == 1 and "daily" in clipped.reason
    gate.record_placed("b", NOW, Session.REGULAR, 100.0)
    result = check(gate, order(qty=10))
    assert not result.approved
    assert "halted" in gate.halt_reason(NOW + timedelta(hours=1))
    assert gate.halt_reason(et(2026, 11, 4, 0, 1)) is None


def test_off_hours_budget_is_five_percent_per_night(gate):
    gate.record_placed("a", NIGHT, Session.OVERNIGHT, 4_900.0)
    clipped = check(gate, order(qty=10), now=NIGHT, status=OVERNIGHT, session=Session.OVERNIGHT)
    assert clipped.approved and clipped.order.qty == 1
    gate.record_placed("b", NIGHT, Session.OVERNIGHT, 100.0)
    deferred = check(
        gate,
        order(qty=10),
        now=NIGHT + timedelta(hours=2),
        status=OVERNIGHT,
        session=Session.OVERNIGHT,
    )
    assert not deferred.approved and "deferred to the regular session" in deferred.reason
    # Not a halt: the regular session and the next night have their own room.
    assert gate.halt_reason(NIGHT) is None
    assert check(gate, order(qty=10), now=et(2026, 11, 4, 10, 0)).approved
    next_night = et(2026, 11, 4, 23, 0)
    assert check(
        gate, order(qty=10), now=next_night, status=OVERNIGHT, session=Session.OVERNIGHT
    ).approved


def test_cancelled_orders_give_their_turnover_back(gate):
    gate.record_placed("a", NIGHT, Session.OVERNIGHT, 5_000.0)
    gate.release_unfilled("a")
    assert check(
        gate, order(qty=10), now=NIGHT, status=OVERNIGHT, session=Session.OVERNIGHT
    ).approved


def test_crypto_drawdown_pauses_buys_but_not_sells(gate):
    start = et(2026, 11, 7, 0, 0)
    for hour, price in enumerate([100_000, 98_000, 92_000, 86_000, 84_000]):
        gate.observe("BTC-USD", start + timedelta(hours=hour), price)
    now = start + timedelta(hours=4)
    buy = order(qty=0.01, limit=84_000, inst="BTC-USD")
    result = check(gate, buy, now=now, q=quote("BTC-USD", 84_000, now))
    assert not result.approved and "crypto buys paused" in result.reason
    sell = order(qty=0.01, limit=84_000, inst="BTC-USD", side=Side.SELL)
    assert check(gate, sell, now=now, q=quote("BTC-USD", 84_000, now)).approved
    assert any("crypto buys paused" in a.message for a in gate.audit.alerts)

    # Once the high rolls out of the 24h window the pause lifts on its own.
    later = start + timedelta(hours=29)
    gate.observe("BTC-USD", later, 84_500)
    assert check(gate, buy, now=later, q=quote("BTC-USD", 84_500, later)).approved


def test_three_rejects_in_ten_minutes_halt_that_asset_class(gate):
    for minutes in (0, 12, 24):
        gate.record_reject(NOW + timedelta(minutes=minutes), "VOO", "x")
    assert gate.lane_halts == {}
    gate.record_reject(NOW + timedelta(minutes=25), "VXUS", "x")
    gate.record_reject(NOW + timedelta(minutes=26), "VOO", "x")
    assert set(gate.lane_halts) == {"alpaca/equity"}

    later = NOW + timedelta(minutes=27)
    result = check(gate, order(), now=later, q=quote(ts=later))
    assert not result.approved and "alpaca/equity halted" in result.reason
    btc = order(qty=0.01, limit=100_000, inst="BTC-USD")
    assert check(gate, btc, now=later, q=quote("BTC-USD", 100_000, later)).approved

    gate.clear_lane("alpaca/equity")
    assert check(gate, order(), now=later, q=quote(ts=later)).approved


def test_crypto_rejects_do_not_count_towards_the_equity_limit(gate):
    gate.record_reject(NOW, "VOO", "x")
    gate.record_reject(NOW, "BTC-USD", "x")
    gate.record_reject(NOW, "ETH-USD", "x")
    gate.record_reject(NOW, "VXUS", "x")
    assert gate.lane_halts == {}


def test_equity_rejects_halt_equities_but_crypto_keeps_trading(growth):
    w = build_world(
        growth,
        NOW,
        dict(PRICES),
        weights={"us_large_cap": 0.38, "intl_equity": 0.15, "btc": 0.20, "eth": 0.10, "cash": 0.17},
    )
    w.venues["alpaca"].reject_next(10, "exchange unavailable", asset_class="equity")
    ts = NOW
    for _ in range(3):
        w.step(ts, PRICES)
        ts += timedelta(minutes=1)
    assert set(w.engine.gate.lane_halts) == {"alpaca/equity"}
    assert w.engine.gate.halt_reason(ts) is None

    # BTC rallies: the crypto sleeve still rebalances while equities stay halted.
    report = w.step(ts, {**PRICES, "BTC-USD": 125_000.0})
    placed = {o.instrument: o.status for o in report.orders}
    assert placed.get("BTC-USD") == "filled"
    assert "VOO" not in placed


def test_reconciliation_gap_over_ten_dollars_halts_everything(growth):
    w = build_world(
        growth,
        NOW,
        dict(PRICES),
        weights={"us_large_cap": 0.45, "intl_equity": 0.15, "btc": 0.26, "eth": 0.10, "cash": 0.04},
    )
    w.account.holdings["BTC-USD"] += 0.00005  # $5: tolerated
    assert w.step(NOW, PRICES).halted is None
    w.account.holdings["BTC-USD"] += 0.001  # another $100
    report = w.step(NOW + timedelta(minutes=1), PRICES)
    assert report.halted and "reconciliation gap" in report.halted
    assert report.orders == []
    assert any(a.level == "critical" for a in w.engine.audit.alerts)


def test_the_halt_names_the_biggest_gap_not_the_last_one(growth):
    w = build_world(growth, NOW, dict(PRICES))
    w.account.holdings["BTC-USD"] += 0.02  # $2,000
    w.account.holdings["VOO"] += 0.01  # $5.60, and it sorts after BTC-USD
    report = w.step(NOW, PRICES)
    assert report.halted.startswith("reconciliation gap $2,005.60 (BTC-USD engine ")
    assert report.halted.endswith(", and 1 smaller)")
