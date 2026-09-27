"""Cash account settlement: equity sale proceeds can't be spent until they settle."""

from datetime import date, datetime, time, timedelta

import pytest
from conftest import PRICES, et, make_steps
from test_recovery import restart, start_world

from rebalancer.engine import EngineConfig
from rebalancer.replay import build_world, run_replay
from rebalancer.sessions import ET
from rebalancer.settlement import SettlementBook
from rebalancer.venue import Order, OrderType, Side, TimeInForce

TUESDAY = et(2026, 11, 3, 10, 0)
# US large cap well over its band and BTC well under it, with little cash: the BTC buy needs the
# proceeds of the VOO sale.
WEIGHTS = {"us_large_cap": 0.56, "intl_equity": 0.16, "btc": 0.12, "eth": 0.12, "cash": 0.04}


@pytest.mark.parametrize(
    ("when", "trade_day", "settles"),
    [
        (et(2026, 11, 3, 10, 0), date(2026, 11, 3), date(2026, 11, 4)),  # Tue regular -> Wed
        (et(2026, 11, 3, 18, 0), date(2026, 11, 3), date(2026, 11, 4)),  # Tue after-hours
        (et(2026, 11, 3, 23, 0), date(2026, 11, 4), date(2026, 11, 5)),  # Tue night trades Wed
        (et(2026, 11, 6, 15, 0), date(2026, 11, 6), date(2026, 11, 9)),  # Fri -> Mon
        (et(2026, 11, 25, 15, 0), date(2026, 11, 25), date(2026, 11, 27)),  # over Thanksgiving
        (et(2026, 11, 8, 21, 0), date(2026, 11, 9), date(2026, 11, 10)),  # Sun night is Monday
    ],
)
def test_trade_and_settlement_dates(calendar, when, trade_day, settles):
    assert calendar.trade_date(when) == trade_day
    assert calendar.settlement_date(trade_day, 1) == settles


def test_no_trade_date_while_equities_are_closed(calendar):
    assert calendar.trade_date(et(2026, 11, 7, 12, 0)) is None


def test_proceeds_are_unsettled_until_the_settlement_date(calendar):
    book = SettlementBook(calendar)
    book.record_sale(et(2026, 11, 6, 15, 0), "equity", 1_000.0)
    book.record_sale(et(2026, 11, 6, 15, 0), "crypto", 500.0)
    assert book.unsettled(et(2026, 11, 8, 23, 59)) == 1_000.0
    assert book.next_settlement(et(2026, 11, 7, 0, 0)) == datetime.combine(
        date(2026, 11, 9), time(0, 0), tzinfo=ET
    )
    assert book.unsettled(et(2026, 11, 9, 0, 0)) == 0.0


def test_the_broker_rejects_a_buy_paid_with_unsettled_proceeds(growth):
    w = build_world(growth, TUESDAY, dict(PRICES), weights=WEIGHTS)
    venue = w.venues["alpaca"]
    sold = Order("s1", "VOO", Side.SELL, 10, 500.0)
    assert venue.place(sold).accepted
    settled = w.account.settled_cash(TUESDAY)
    qty = round((settled + 1_000) / 100_300, 6)
    buy = Order("b1", "BTC-USD", Side.BUY, qty, 100_300.0, OrderType.LIMIT, TimeInForce.GTC)
    ack = venue.place(buy)
    assert not ack.accepted and ack.reason == "insufficient settled cash"


def run_tuesday_to_wednesday(growth, calendar, **config):
    steps = make_steps(
        calendar,
        TUESDAY,
        et(2026, 11, 4, 2, 0),
        {inst: (lambda p: lambda ts: p)(price) for inst, price in PRICES.items()},
    )
    return run_replay(
        growth, steps, calendar=calendar, weights=WEIGHTS, config=EngineConfig(**config)
    )


def test_btc_waits_for_the_voo_proceeds_to_settle(growth, calendar):
    result = run_tuesday_to_wednesday(growth, calendar)
    orders = result.orders
    assert (orders["status"] != "rejected").all()  # never trips the broker's rule

    # The 5% single-order cap splits the VOO sale over two cycles on Tuesday morning.
    voo_sales = orders[(orders["instrument"] == "VOO") & (orders["side"] == "sell")]
    assert voo_sales["ts"].min() == TUESDAY and voo_sales["ts"].max() < et(2026, 11, 3, 12, 0)
    btc_buys = orders[(orders["instrument"] == "BTC-USD") & (orders["side"] == "buy")]
    tuesday = btc_buys[btc_buys["ts"] < et(2026, 11, 4, 0, 0)]
    wednesday = btc_buys[btc_buys["ts"] >= et(2026, 11, 4, 0, 0)]
    assert not tuesday.empty and not wednesday.empty
    assert result.decisions["detail"].str.contains("not settled yet").any()

    snaps = result.snapshots.set_index("ts")
    assert snaps.loc[et(2026, 11, 3, 23, 30), "unsettled"] > 4_000
    assert snaps.loc[et(2026, 11, 4, 0, 30), "unsettled"] == 0
    # Buys go to the band edge, so they land a hair under it after the spread and fees.
    lo = 0.17 - 0.001
    assert snaps.loc[et(2026, 11, 3, 23, 30), "w_btc"] < lo
    assert snaps.loc[et(2026, 11, 4, 1, 0), "w_btc"] >= lo


def test_without_settlement_the_proceeds_go_straight_back_out(growth, calendar):
    result = run_tuesday_to_wednesday(growth, calendar, settlement_days={"equity": 0, "crypto": 0})
    snaps = result.snapshots.set_index("ts")
    assert snaps.loc[et(2026, 11, 3, 11, 0), "w_btc"] >= 0.17 - 0.001


def test_unsettled_proceeds_survive_a_restart(schema, dsn, growth):
    w = start_world(dsn, growth, TUESDAY, WEIGHTS)
    w.step(TUESDAY, PRICES)
    before = w.engine.settlement.unsettled(TUESDAY)
    assert before > 4_000

    restart(w, dsn, TUESDAY + timedelta(minutes=30))
    assert w.engine.settlement.unsettled(TUESDAY + timedelta(minutes=30)) == pytest.approx(before)
    report = w.step(TUESDAY + timedelta(minutes=30), PRICES)
    assert all(o.status != "rejected" for o in report.orders)


def test_a_sale_that_fills_while_the_engine_is_down_is_still_unsettled(schema, dsn, growth):
    w = start_world(dsn, growth, TUESDAY, WEIGHTS)
    venue = w.venues["alpaca"]
    venue.fills_paused = True
    w.step(TUESDAY, PRICES)
    assert w.engine.settlement.unsettled(TUESDAY) == 0

    venue.fills_paused = False
    w.clock.now = TUESDAY + timedelta(minutes=5)
    venue.set_price("VOO", PRICES["VOO"], w.clock.now)

    restart(w, dsn, TUESDAY + timedelta(minutes=30))
    assert w.engine.settlement.unsettled(TUESDAY + timedelta(minutes=30)) > 4_000
    report = w.step(TUESDAY + timedelta(minutes=30), PRICES)
    assert all(o.status != "rejected" for o in report.orders)
