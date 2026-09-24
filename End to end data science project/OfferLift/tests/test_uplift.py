import numpy as np
import pytest

from offerlift import data, evaluation, policy, uplift


@pytest.fixture
def contrast(synthetic_experiment):
    subset = data.contrast(synthetic_experiment, "segment", "No E-Mail", "Mens E-Mail")
    return subset, data.feature_matrix(subset)


def test_perfect_score_beats_random_on_qini(contrast):
    subset, _ = contrast
    y, t = subset["visit"].to_numpy(), subset["treated"].to_numpy()
    oracle = evaluation.qini_coefficient(subset["true_uplift"].to_numpy(), y, t)
    rng = np.random.default_rng(0)
    random = evaluation.qini_coefficient(rng.random(len(y)), y, t)
    assert oracle > 0
    assert oracle > random + 0.005


def test_qini_curve_ends_at_total_incremental(contrast):
    subset, _ = contrast
    y, t = subset["visit"].to_numpy(), subset["treated"].to_numpy()
    _, incremental = evaluation.qini_curve(np.zeros(len(y)), y, t)
    expected = y[t == 1].sum() - y[t == 0].sum() * (t == 1).sum() / (t == 0).sum()
    assert incremental[-1] == pytest.approx(expected)


@pytest.mark.parametrize(
    "model",
    [uplift.TLearner(uplift.logistic_classifier()), uplift.TLearner(), uplift.TransformedOutcome()],
)
def test_models_rank_responsive_customers_first(contrast, model):
    subset, features = contrast
    y, t = subset["visit"].to_numpy(), subset["treated"].to_numpy()
    score = model.fit(features, y, t).predict_uplift(features)
    responsive = subset["true_uplift"].to_numpy() > 0
    assert score[responsive].mean() > score[~responsive].mean()


def test_budget_table_targets_better_than_random(contrast):
    subset, _ = contrast
    y, t = subset["visit"].to_numpy(), subset["treated"].to_numpy()
    table = policy.budget_table(
        {"oracle": subset["true_uplift"].to_numpy()}, y, t, [0.3], n_boot=100
    )
    by_policy = table.set_index("policy")
    assert by_policy.loc["oracle", "incremental_per_customers"] > by_policy.loc[
        "random", "incremental_per_customers"
    ]
    assert by_policy.loc["oracle", "gain_ci_low"] > 0


def test_budget_table_intervals_bracket_estimates(contrast):
    subset, _ = contrast
    y, t = subset["visit"].to_numpy(), subset["treated"].to_numpy()
    table = policy.budget_table(
        {"oracle": subset["true_uplift"].to_numpy()}, y, t, [0.1, 0.5], n_boot=100
    )
    assert (table["uplift_ci_low"] <= table["uplift_in_targeted"]).all()
    assert (table["uplift_in_targeted"] <= table["uplift_ci_high"]).all()
    random_rows = table[table["policy"] == "random"]
    assert (random_rows[["gain_vs_random_per_customers", "gain_ci_low", "gain_ci_high"]] == 0).all().all()


def test_noise_score_gain_interval_includes_zero(contrast):
    subset, _ = contrast
    y, t = subset["visit"].to_numpy(), subset["treated"].to_numpy()
    noise = np.random.default_rng(3).random(len(y))
    row = policy.budget_table({"noise": noise}, y, t, [0.3], n_boot=200).set_index("policy")
    assert row.loc["noise", "gain_ci_low"] < 0 < row.loc["noise", "gain_ci_high"]


def test_uplift_at_fraction_rejects_bad_fraction(contrast):
    subset, _ = contrast
    with pytest.raises(ValueError):
        evaluation.uplift_at_fraction(np.zeros(len(subset)), subset["visit"], subset["treated"], 0)
