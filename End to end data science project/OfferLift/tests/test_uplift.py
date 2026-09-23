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
    table = policy.budget_table({"oracle": subset["true_uplift"].to_numpy()}, y, t, [0.3])
    by_policy = table.set_index("policy")["incremental_per_customers"]
    assert by_policy["oracle"] > by_policy["random"]


def test_uplift_at_fraction_rejects_bad_fraction(contrast):
    subset, _ = contrast
    with pytest.raises(ValueError):
        evaluation.uplift_at_fraction(np.zeros(len(subset)), subset["visit"], subset["treated"], 0)
