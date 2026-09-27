from datetime import timedelta

import pytest
from conftest import MODELS, et

httpx = pytest.importorskip("httpx")

from fake_alpaca import KEY, SECRET, FakeAlpaca  # noqa: E402

from rebalancer import alpaca  # noqa: E402
from rebalancer.alpaca import (  # noqa: E402
    AlpacaConfig,
    AlpacaError,
    AlpacaVenue,
    check,
    parse_time,
)
from rebalancer.engine import Engine  # noqa: E402
from rebalancer.models import load_model  # noqa: E402
from rebalancer.replay import DEFAULT_INSTRUMENTS  # noqa: E402
from rebalancer.venue import Order, OrderType, Side, TimeInForce  # noqa: E402

TUESDAY = et(2026, 11, 3, 10, 0)
NIGHT = et(2026, 11, 3, 23, 0)
CONFIG = AlpacaConfig(KEY, SECRET)


@pytest.fixture
def fake():
    server = FakeAlpaca(TUESDAY)
    server.set_quote("VOO", 559.9, 560.1)
    server.set_quote("VXUS", 67.99, 68.01)
    server.set_quote("BTC/USD", 99_990.0, 100_010.0)
    server.set_quote("ETH/USD", 3_499.0, 3_501.0)
    return server


def venue_for(fake, calendar, **kw):
    return AlpacaVenue(
        CONFIG, DEFAULT_INSTRUMENTS, calendar, client=fake.client(), clock=lambda: fake.now, **kw
    )


def last_order_body(fake):
    return next(body for method, path, body in reversed(fake.requests) if method == "POST")


# --- configuration -----------------------------------------------------------------------------


def test_refuses_anything_but_the_paper_url(calendar):
    live = AlpacaConfig(KEY, SECRET, trading_url="https://api.alpaca.markets")
    with pytest.raises(AlpacaError, match="paper only"):
        AlpacaVenue(live, DEFAULT_INSTRUMENTS, calendar)


def test_keys_come_from_the_environment():
    with pytest.raises(AlpacaError, match="APCA_API_SECRET_KEY"):
        AlpacaConfig.from_env({"APCA_API_KEY_ID": "x"})
    config = AlpacaConfig.from_env({"APCA_API_KEY_ID": "k", "APCA_API_SECRET_KEY": "s"})
    assert config.trading_url == alpaca.PAPER_URL and config.stock_feed == "iex"


def test_bad_keys_are_an_error(fake, calendar):
    venue = AlpacaVenue(
        AlpacaConfig(KEY, "wrong"), DEFAULT_INSTRUMENTS, calendar, client=fake.client()
    )
    with pytest.raises(AlpacaError, match="403"):
        venue.positions()


# --- orders ------------------------------------------------------------------------------------


def test_regular_session_equity_order(fake, calendar):
    venue = venue_for(fake, calendar)
    ack = venue.place(Order("c1", "VOO", Side.BUY, 1.5, 561.123, OrderType.LIMIT, TimeInForce.DAY))
    assert ack.accepted and ack.order_id == "ord-1"
    assert last_order_body(fake) == {
        "symbol": "VOO",
        "qty": "1.5",
        "side": "buy",
        "type": "limit",
        "time_in_force": "day",
        "client_order_id": "c1",
        "limit_price": "561.12",
    }


def test_overnight_equity_orders_are_flagged_extended_hours(fake, calendar):
    fake.now = NIGHT
    fake.set_quote("VOO", 559.9, 560.1)
    venue = venue_for(fake, calendar)
    assert venue.session_status("VOO").session == "overnight"
    venue.place(Order("c1", "VOO", Side.BUY, 1, 561.0, OrderType.LIMIT, TimeInForce.DAY))
    assert last_order_body(fake)["extended_hours"] is True


def test_crypto_orders_use_the_broker_symbol_and_gtc(fake, calendar):
    venue = venue_for(fake, calendar)
    assert venue.session_status("BTC-USD").time_in_force == {TimeInForce.GTC, TimeInForce.IOC}
    venue.place(
        Order(
            "c1", "BTC-USD", Side.SELL, 0.000308, 99_800.12345678, OrderType.LIMIT, TimeInForce.GTC
        )
    )
    body = last_order_body(fake)
    assert (body["symbol"], body["qty"], body["limit_price"], body["time_in_force"]) == (
        "BTC/USD",
        "0.000308",
        "99800.12345678",
        "gtc",
    )
    assert "extended_hours" not in body


def test_a_refused_order_is_a_rejection_with_the_brokers_reason(fake, calendar):
    fake.reject_next_post = (403, "insufficient buying power")
    ack = venue_for(fake, calendar).place(Order("c1", "VOO", Side.BUY, 1, 561.0))
    assert not ack.accepted and ack.reason == "403: insufficient buying power"


def test_a_server_error_raises(fake, calendar):
    fake.reject_next_post = (500, "internal error")
    with pytest.raises(AlpacaError, match="500"):
        venue_for(fake, calendar).place(Order("c1", "VOO", Side.BUY, 1, 561.0))


@pytest.mark.parametrize(("when", "accepted"), [("after", True), ("before", False)])
def test_a_timeout_is_settled_by_looking_the_order_up(fake, calendar, when, accepted):
    fake.timeout_next_post = when
    ack = venue_for(fake, calendar).place(Order("c1", "VOO", Side.BUY, 1, 561.0))
    assert ack.accepted is accepted
    assert ("confirmed by lookup" if accepted else "not received") in ack.reason


def test_find_order_and_cancel(fake, calendar):
    fake.fill_orders = False
    venue = venue_for(fake, calendar)
    ack = venue.place(Order("c1", "VOO", Side.BUY, 1, 561.0))
    assert venue.find_order("c1") == ack.order_id
    assert venue.find_order("never-sent") is None
    assert venue.cancel(ack.order_id) is True
    assert venue.cancel(ack.order_id) is False  # already cancelled


def test_cancelling_a_filled_order_returns_false(fake, calendar):
    venue = venue_for(fake, calendar)
    ack = venue.place(Order("c1", "VOO", Side.BUY, 1, 561.0))
    assert venue.cancel(ack.order_id) is False


# --- account state -----------------------------------------------------------------------------


def test_positions_map_broker_symbols_and_report_cash(fake, calendar):
    fake.positions = {"VOO": 10.0, "BTCUSD": 0.25, "AAPL": 3.0}
    held = venue_for(fake, calendar).positions()
    assert held == {"VOO": 10.0, "BTC-USD": 0.25, "AAPL": 3.0, "USD": 10_000.0}


def test_quotes_parse_and_skip_empty_books(fake, calendar):
    fake.set_quote("VOO", 559.9, 560.1, at=TUESDAY - timedelta(seconds=5))
    venue = venue_for(fake, calendar)
    quote = venue.quote("VOO")
    assert (quote.bid, quote.ask) == (559.9, 560.1)
    assert quote.age(TUESDAY) == timedelta(seconds=5)
    assert venue.quote("BTC-USD").mid == pytest.approx(100_000.0)
    fake.set_quote("VXUS", 0.0, 68.01)
    assert venue.quote("VXUS") is None


def test_a_failed_quote_is_none_not_a_crash(fake, calendar):
    del fake.quotes["VOO"]
    venue = venue_for(fake, calendar)
    assert venue.quote("VOO") is None
    assert "404" in venue.last_quote_error


def test_nanosecond_timestamps():
    ts = parse_time("2026-11-03T15:00:00.123456789Z")
    assert (ts.microsecond, ts.utcoffset()) == (123456, timedelta(0))
    assert parse_time("2026-11-03T15:00:00Z").second == 0


def test_fills_page_through_and_skip_what_isnt_ours(fake, calendar, monkeypatch):
    monkeypatch.setattr(alpaca, "PAGE_SIZE", 2)
    venue = venue_for(fake, calendar)
    for i in range(3):
        venue.place(Order(f"c{i}", "VOO", Side.BUY, 1, 561.0))
    fake.activities.append({**fake.activities[0], "id": "x", "symbol": "AAPL", "order_id": "other"})
    fills = venue.fills(TUESDAY - timedelta(minutes=1))
    assert [f.order_id for f in fills] == ["ord-1", "ord-2", "ord-3"]
    assert all(f.instrument == "VOO" and f.side is Side.BUY and f.qty == 1 for f in fills)
    assert venue.fills(TUESDAY + timedelta(minutes=1)) == []


# --- with the engine -----------------------------------------------------------------------------


def test_the_engine_trades_and_reconciles_through_the_adapter(fake, calendar):
    growth = load_model(MODELS / "growth-247.yaml")
    # A $100k account that's BTC-heavy: BTC at 26% against a 17-23% band.
    fake.cash = 7_000.0
    fake.positions = {"VOO": 75.0, "VXUS": 220.588, "BTCUSD": 0.26, "ETHUSD": 2.857}
    venue = venue_for(fake, calendar)
    engine = Engine(growth, {"alpaca": venue}, DEFAULT_INSTRUMENTS, calendar)
    engine.start(TUESDAY)

    report = engine.cycle(TUESDAY)
    assert [(o.instrument, o.side, o.status) for o in report.orders] == [
        ("BTC-USD", "sell", "filled")
    ]
    assert last_order_body(fake)["symbol"] == "BTC/USD"

    fake.now = TUESDAY + timedelta(minutes=1)
    for symbol, (bid, ask) in list(fake.quotes.items()):
        fake.set_quote(symbol, bid, ask)
    later = engine.cycle(fake.now)
    assert later.halted is None and later.orders == []
    assert engine.ledger["BTC-USD"] == pytest.approx(fake.positions["BTCUSD"])
    assert engine.ledger["USD"] == pytest.approx(fake.cash)


def test_check_reports_the_account_and_round_trips_an_order(fake, calendar):
    lines = []
    check(venue_for(fake, calendar), order_test=True, out=lines.append)
    text = "\n".join(lines)
    assert "account PA0001 (ACTIVE)" in text
    assert "VOO: session regular, bid 559.9 ask 560.1" in text
    assert "order test: cancel -> True" in text
    assert "final status -> canceled" in text
    placed = last_order_body(fake)
    assert float(placed["limit_price"]) == pytest.approx(559.9 * 0.5, abs=0.01)
