import numpy as np
import pytest

from offerlift import data, segments


@pytest.fixture
def contrast(synthetic_experiment):
    subset = data.contrast(synthetic_experiment, "segment", "No E-Mail", "Mens E-Mail")
    return segments.add_segments(subset)


def test_purchase_segments_follow_flags(contrast):
    expected = np.select(
        [(contrast.mens == 1) & (contrast.womens == 1), contrast.mens == 1, contrast.womens == 1],
        ["both", "mens only", "womens only"],
        "neither",
    )
    assert (contrast["purchased"] == expected).all()


def test_segment_lifts_recover_known_effects(contrast):
    rows = {row["level"]: row for row in segments.segment_lifts(contrast, "recency_band", "visit")}
    # True uplift is 0.15 for recency 1-4 and 0 otherwise.
    assert rows["1-3"]["ci_low"] < 0.15 < rows["1-3"]["ci_high"]
    assert rows["10-12"]["ci_low"] < 0 < rows["10-12"]["ci_high"]
    assert sum(row["n"] for row in rows.values()) == len(contrast)


def test_lift_difference_detects_responsive_group(contrast):
    responsive = (contrast["true_uplift"] > 0).to_numpy()
    result = segments.lift_difference(contrast, responsive, "visit")
    assert result["ci_low"] > 0
    assert result["ci_low"] < 0.15 < result["ci_high"]


def test_lift_difference_finds_nothing_in_irrelevant_split(contrast):
    result = segments.lift_difference(contrast, (contrast["mens"] == 1).to_numpy(), "visit")
    assert result["ci_low"] < 0 < result["ci_high"]


def test_profile_compares_targeted_with_rest(contrast):
    targeted = (contrast["recency"] <= 2).to_numpy()
    profile = segments.targeted_profile(contrast, targeted)
    assert profile["numeric"]["recency"]["targeted"] < profile["numeric"]["recency"]["rest"]
    shares = profile["categorical"]["channel"]["targeted"]
    assert sum(shares.values()) == pytest.approx(1)
