import numpy as np
import pandas as pd
import pytest

from offerlift import data, experiment

ALLOCATION = {"No E-Mail": 1 / 3, "Mens E-Mail": 1 / 3, "Womens E-Mail": 1 / 3}


def test_balanced_arms_pass_srm(synthetic_experiment):
    result = experiment.sample_ratio_mismatch(synthetic_experiment["segment"], ALLOCATION)
    assert not result.mismatch


def test_skewed_arms_fail_srm():
    arms = pd.Series(["No E-Mail"] * 4000 + ["Mens E-Mail"] * 3000 + ["Womens E-Mail"] * 3000)
    assert experiment.sample_ratio_mismatch(arms, ALLOCATION).mismatch


def test_unplanned_arm_is_rejected():
    with pytest.raises(ValueError, match="not in the planned allocation"):
        experiment.sample_ratio_mismatch(pd.Series(["Other"]), ALLOCATION)


def test_randomized_features_are_balanced(synthetic_experiment):
    subset = data.contrast(synthetic_experiment, "segment", "No E-Mail", "Mens E-Mail")
    smd = experiment.standardized_mean_differences(data.feature_matrix(subset), subset["treated"])
    assert smd.max() < 0.1


def test_difference_in_means_covers_known_effect():
    rng = np.random.default_rng(0)
    treated = rng.integers(0, 2, 50_000)
    outcome = rng.normal(10 + 0.5 * treated, 2)
    estimate = experiment.difference_in_means(outcome, treated)
    assert estimate.ci_low < 0.5 < estimate.ci_high
    assert estimate.p_value < 1e-6


def test_cuped_keeps_estimate_and_shrinks_interval():
    rng = np.random.default_rng(1)
    n = 20_000
    covariate = rng.normal(100, 20, n)
    treated = rng.integers(0, 2, n)
    outcome = 0.8 * covariate + 1.0 * treated + rng.normal(0, 5, n)
    raw = experiment.difference_in_means(outcome, treated)
    adjusted = experiment.difference_in_means(
        experiment.cuped_adjust(outcome, covariate), treated
    )
    assert adjusted.ci_low < 1.0 < adjusted.ci_high
    assert adjusted.standard_error < 0.5 * raw.standard_error


def test_mde_and_sample_size_are_consistent():
    n = experiment.required_sample_size(0.10, 0.01)
    mde = experiment.minimum_detectable_effect(0.10, n)
    assert mde == pytest.approx(0.01, rel=0.1)


def test_holm_is_monotone_and_capped():
    adjusted = experiment.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.08, "c": 0.5})
