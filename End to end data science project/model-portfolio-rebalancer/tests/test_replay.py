from datetime import timedelta

import pandas as pd
import pytest
from conftest import make_steps, ramp

from rebalancer.engine import EngineConfig
from rebalancer.replay import load_price_csv, run_replay
from rebalancer.scenarios import generate_prices
from rebalancer.sessions import ET, MarketCalendar

EQUITY = {"VOO", "VXUS"}
FROZEN = {"weekend", "closed", "pause"}


def et(*args):
    from datetime import datetime

    return pd.Timestamp(datetime(*args, tzinfo=ET))


def flat(price):
    return lambda ts: price


def off_hours_turnover(orders, calendar):
    placed = orders[orders["status"] != "rejected"].copy()
    placed = placed[placed["session"] != "regular"]
    placed["window"] = [calendar.last_regular_close(ts.to_pydatetime()) for ts in placed["ts"]]
    return placed.groupby("window")["notional"].sum()


# --- weekend crypto crash ------------------------------------------------------------------

CRASH_START = et(2026, 11, 14, 2, 0)  # Saturday 2am ET
CRASH_END = et(2026, 11, 14, 8, 0)


def crash_steps(calendar, drop):
    return make_steps(
        calendar,
        et(2026, 11, 13, 10, 0),
        et(2026, 11, 16, 12, 0),
        {
            "VOO": flat(560.0),
            "VXUS": flat(68.0),
            "BTC-USD": ramp(100_000, 100_000 * (1 - drop), CRASH_START, CRASH_END),
            "ETH-USD": ramp(3_500, 3_500 * (1 - drop), CRASH_START, CRASH_END),
        },
    )


@pytest.fixture(scope="module")
def severe_crash(crypto_tilt, calendar):
    return run_replay(crypto_tilt, crash_steps(calendar, 0.20), calendar=calendar)


def test_weekend_crash_never_touches_frozen_equities(severe_crash):
    orders = severe_crash.orders
    assert not orders.empty
    equity = orders[orders["instrument"].isin(EQUITY)]
    assert not equity["session"].isin(FROZEN).any()
    weekend = orders[
        (orders["ts"] >= et(2026, 11, 13, 20, 0)) & (orders["ts"] < et(2026, 11, 15, 20, 0))
    ]
    assert set(weekend["instrument"]) <= {"BTC-USD", "ETH-USD"}


def test_weekend_crash_pauses_crypto_buys_while_down_fifteen_percent(severe_crash):
    alerts = severe_crash.alerts
    paused = alerts[alerts["message"].str.startswith("crypto buys paused")]
    resumed = alerts[alerts["message"].str.startswith("crypto buys resumed")]
    assert len(paused) == 1 and len(resumed) == 1
    pause_at, resume_at = paused["ts"].iloc[0], resumed["ts"].iloc[0]
    # The 15% line is crossed 4.5h into a 6h, 20% slide; it clears when the pre-crash prices
    # above $94,118 roll out of the 24h window.
    assert pause_at == et(2026, 11, 14, 6, 30)
    assert resume_at == et(2026, 11, 15, 4, 0)

    orders = severe_crash.orders
    crypto_buys = orders[orders["instrument"].str.endswith("-USD") & (orders["side"] == "buy")]
    assert not ((crypto_buys["ts"] >= pause_at) & (crypto_buys["ts"] < resume_at)).any()
    holds = severe_crash.decisions
    assert holds["detail"].str.contains("crypto buys paused").any()


def test_weekend_crash_buys_back_after_the_pause_within_the_off_hours_budget(
    severe_crash, calendar
):
    orders = severe_crash.orders
    after = orders[
        (orders["ts"] >= et(2026, 11, 15, 4, 0)) & (orders["ts"] < et(2026, 11, 15, 20, 0))
    ]
    assert set(after["side"]) == {"buy"} and set(after["status"]) == {"filled"}
    per_window = off_hours_turnover(orders, calendar)
    assert (per_window <= 0.05 * 100_000 * 1.001).all()
    for _, row in orders.iterrows():
        assert row["order_type"] == "limit"
        assert abs(row["limit_price"] / row["reference_price"] - 1) <= 0.02


def test_moderate_weekend_dip_is_bought_on_saturday(crypto_tilt, calendar):
    result = run_replay(crypto_tilt, crash_steps(calendar, 0.12), calendar=calendar)
    orders = result.orders
    saturday = orders[
        (orders["ts"] >= et(2026, 11, 14, 0, 0)) & (orders["ts"] < et(2026, 11, 15, 0, 0))
    ]
    assert not saturday.empty
    assert set(saturday["instrument"]) <= {"BTC-USD", "ETH-USD"}
    assert set(saturday["side"]) == {"buy"}
    assert result.alerts.empty


# --- Monday gap open -----------------------------------------------------------------------

FRIDAY_CLOSE = 560.0
MONDAY_PRICE = 504.0  # 10% gap down


def gap_steps(calendar, first_print):
    """Equities close Friday at $560 and next print at $504, either in Sunday's overnight
    session or not until Monday's regular open."""
    reopen = et(2026, 11, 15, 20, 0) if first_print == "overnight" else et(2026, 11, 16, 9, 30)

    def voo(ts):
        if ts < et(2026, 11, 13, 20, 0):
            return FRIDAY_CLOSE
        return MONDAY_PRICE if ts >= reopen else None

    def vxus(ts):
        if ts < et(2026, 11, 13, 20, 0):
            return 68.0
        return 61.2 if ts >= reopen else None

    return make_steps(
        calendar,
        et(2026, 11, 13, 15, 0),
        et(2026, 11, 16, 12, 0),
        {"VOO": voo, "VXUS": vxus, "BTC-USD": flat(100_000.0), "ETH-USD": flat(3_500.0)},
    )


# US large cap starts near the bottom of its 42-48% band, so a 10% gap takes it out. The account
# is $500k because on $100k a quarter of the shortfall is less than one VOO share.
GAP_WEIGHTS = {"us_large_cap": 0.43, "intl_equity": 0.15, "btc": 0.20, "eth": 0.10, "cash": 0.12}


@pytest.fixture(scope="module", params=["overnight", "regular"])
def gap(request, growth, calendar):
    result = run_replay(
        growth,
        gap_steps(calendar, request.param),
        calendar=calendar,
        weights=GAP_WEIGHTS,
        value=500_000,
    )
    return request.param, result


def test_monday_gap_no_equity_orders_until_a_fresh_print(gap):
    first_print, result = gap
    orders = result.orders
    equity = orders[orders["instrument"].isin(EQUITY)]
    assert not equity.empty
    reopen = et(2026, 11, 15, 20, 0) if first_print == "overnight" else et(2026, 11, 16, 9, 30)
    assert equity["ts"].min() >= reopen
    # Over the weekend the equity sleeves sat at Friday's price with the stale flag set.
    snaps = result.snapshots
    weekend = snaps[
        (snaps["ts"] > et(2026, 11, 14, 0, 0)) & (snaps["ts"] < et(2026, 11, 15, 12, 0))
    ]
    assert weekend["stale_us_large_cap"].all()


def test_monday_gap_orders_are_priced_and_sized_off_the_gapped_price(gap):
    first_print, result = gap
    orders = result.orders
    voo = orders[orders["instrument"] == "VOO"].iloc[0]
    assert voo["side"] == "buy"
    assert voo["reference_price"] == pytest.approx(MONDAY_PRICE, rel=1e-3)
    assert abs(voo["limit_price"] / MONDAY_PRICE - 1) <= 0.02

    snaps = result.snapshots.set_index("ts")
    before = snaps.loc[voo["ts"]]
    # The engine re-marks from the first live print before sizing: the need is measured at $504.
    need = (0.42 - before["w_us_large_cap"]) * before["value"]
    assert need > 0
    if first_print == "overnight":
        assert voo["session"] == "overnight"
        assert voo["notional"] <= 0.25 * need + MONDAY_PRICE
    else:
        assert voo["session"] == "regular"
        assert voo["notional"] > 0.25 * need


# --- whole sample replay -------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_steps():
    return generate_prices()


@pytest.mark.parametrize("model_name", ["crypto_tilt", "growth"])
def test_sample_replay_respects_every_rule(model_name, request, sample_steps, calendar):
    model = request.getfixturevalue(model_name)
    result = run_replay(model, sample_steps, calendar=calendar)
    orders, snaps = result.orders, result.snapshots
    assert not orders.empty
    assert (snaps["halted"] == "").all()
    assert snaps[f"w_{model.cash.id}"].min() >= model.cash_floor

    values = snaps.set_index("ts")["value"]
    live = orders[orders["status"] != "rejected"]
    assert (live["order_type"] == "limit").all()
    assert ((live["limit_price"] / live["reference_price"] - 1).abs() <= 0.02).all()
    assert (live["notional"] <= 0.05 * live["ts"].map(values) + 1e-6).all()
    equity = live[live["instrument"].isin(EQUITY)]
    assert not equity["session"].isin(FROZEN).any()

    daily = live.groupby(live["ts"].map(lambda t: t.tz_convert(ET).date()))["notional"].sum()
    assert (daily <= 0.15 * values.max()).all()
    assert (off_hours_turnover(orders, calendar) <= 0.05 * values.max()).all()


def test_regular_hours_only_baseline_trades_only_in_the_regular_session(
    growth, sample_steps, calendar
):
    result = run_replay(
        growth, sample_steps, calendar=calendar, config=EngineConfig(regular_hours_only=True)
    )
    assert set(result.orders["session"]) == {"regular"}


def test_price_csv_round_trip(tmp_path, growth, calendar):
    steps = generate_prices(end=generate_prices()[0][0] + timedelta(days=3))
    path = tmp_path / "prices.csv"
    rows = ["ts,instrument,price"] + [
        f"{ts.isoformat()},{inst},{p}" for ts, prices in steps for inst, p in prices.items()
    ]
    path.write_text("\n".join(rows) + "\n")
    loaded = load_price_csv(path)
    assert len(loaded) == len(steps)
    assert run_replay(growth, loaded, calendar=calendar).summary["cycles"] == len(steps)


def test_dec_6_pause_is_frozen_for_equities(growth, calendar):
    assert calendar.session_at(et(2026, 12, 8, 20, 30).to_pydatetime()).session.value == "pause"
    result = run_replay(growth, generate_prices(), calendar=MarketCalendar())
    pause_orders = result.orders[result.orders["session"] == "pause"]
    assert not pause_orders["instrument"].isin(EQUITY).any()
