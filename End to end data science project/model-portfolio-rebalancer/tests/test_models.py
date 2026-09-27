import copy

import pytest
from conftest import MODELS

from rebalancer.models import Band, ModelError, load_models, parse_model

BASE = {
    "model": "test",
    "version": 1,
    "cash_floor": 0.02,
    "sleeves": [
        {
            "id": "us",
            "target": 0.6,
            "band": {"abs": 0.03},
            "instruments": {"regular": ["VOO"], "overnight": ["VOO"], "weekend": []},
        },
        {
            "id": "btc",
            "target": 0.3,
            "band": {"abs": 0.03, "rel": 0.15},
            "instruments": {"any": ["BTC-USD"]},
        },
        {"id": "cash", "target": 0.1, "instruments": {"any": ["USD"]}},
    ],
}


def test_starter_models_have_the_spec_splits():
    models = load_models(MODELS)
    splits = {}
    for name, model in models.items():
        weights = {s.id: s.target for s in model.sleeves}
        splits[name] = (
            round(weights["us_large_cap"] + weights["intl_equity"], 6),
            round(weights["btc"] + weights["eth"], 6),
            weights["cash"],
        )
        assert model.cash_floor == 0.02
    assert splits == {
        "core-247": (0.80, 0.10, 0.10),
        "growth-247": (0.60, 0.30, 0.10),
        "crypto-tilt": (0.40, 0.50, 0.10),
    }


def test_equity_sleeves_freeze_on_weekends_and_use_overnight_list_in_extended_hours():
    us = load_models(MODELS)["growth-247"].sleeve("us_large_cap")
    assert us.instruments_for("weekend") == ()
    assert us.instruments_for("closed") == ()
    assert us.instruments_for("pause") == ()
    assert us.instruments_for("pre") == ("VOO",)
    assert us.instruments_for("overnight") == ("VOO",)
    btc = load_models(MODELS)["growth-247"].sleeve("btc")
    assert all(
        btc.instruments_for(s) == ("BTC-USD",) for s in ("regular", "overnight", "weekend", "pause")
    )


def test_band_is_the_tighter_of_abs_and_rel():
    assert Band(abs=0.03, rel=0.20).width(0.45) == pytest.approx(0.03)
    assert Band(abs=0.03, rel=0.10).width(0.10) == pytest.approx(0.01)
    assert Band(rel=0.2).width(0.5) == pytest.approx(0.1)


def test_base_model_parses():
    model = parse_model(BASE)
    assert [s.id for s in model.sleeves] == ["us", "btc", "cash"]
    assert model.sleeve("us").max_off_hours_pct == 0.25


def _broken(change):
    raw = copy.deepcopy(BASE)
    change(raw)
    return raw


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r["sleeves"][0].update(target=0.7), "sum to"),
        (lambda r: r["sleeves"][0]["instruments"].update(sunday=["VOO"]), "unknown session"),
        (lambda r: r["sleeves"][1]["instruments"].update(any=["VOO"]), "is in both"),
        (lambda r: r.update(cash_floor=0.15), "below the cash floor"),
        (lambda r: r["sleeves"][0].pop("band"), "band is required"),
        (lambda r: r["sleeves"][2].update(id="money"), "exactly one sleeve"),
        (
            lambda r: r["sleeves"][2]["instruments"].update(any=["USD", "BTC-USD"]),
            "cash sleeve can only hold",
        ),
        (lambda r: r["sleeves"][0].update(target=-0.1), "between 0 and 1"),
        (lambda r: r.update(version="3"), "positive integer"),
        (lambda r: r.update(model="Growth Model"), "lowercase slug"),
        (
            lambda r: r["sleeves"][0].update(session_policy={"max_off_hours_pct": 2}),
            "max_off_hours_pct",
        ),
        (lambda r: r["sleeves"][1].update(id="us"), "more than once"),
        (lambda r: r["sleeves"][0].update(instruments={"weekend": []}), "no tradable instrument"),
    ],
)
def test_invalid_models_are_rejected(change, message):
    with pytest.raises(ModelError, match=message):
        parse_model(_broken(change))
