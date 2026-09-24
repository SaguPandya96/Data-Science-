import numpy as np
import pytest

from offerlift import confirmation, data, uplift


class NoiseModel:
    """Scores unrelated to the customer: should never beat random targeting."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def fit(self, features, outcome, treated):
        return self

    def predict_uplift(self, features):
        return self.rng.random(len(features))


@pytest.fixture
def contrast(synthetic_experiment):
    subset = data.contrast(synthetic_experiment, "segment", "No E-Mail", "Mens E-Mail")
    return (
        data.feature_matrix(subset),
        subset["visit"].to_numpy(),
        subset["treated"].to_numpy(),
    )


def test_every_customer_is_scored_out_of_fold(contrast):
    features, y, t = contrast
    scores, fold_ids = confirmation.cross_fitted_scores(
        features, y, t, uplift.MODELS["t_learner_logistic"], folds=4, random_state=0
    )
    assert np.isfinite(scores).all()
    assert set(fold_ids) == {0, 1, 2, 3}


def test_top_is_taken_within_each_fold():
    scores = np.array([10.0, 9.0, 8.0, 7.0, 0.4, 0.3, 0.2, 0.1])
    fold_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    top = confirmation.top_within_folds(scores, fold_ids, 0.25)
    # Fold 1's scores are all lower, but its best customer is still targeted.
    assert top.tolist() == [True, False, False, False, True, False, False, False]


def test_gain_is_zero_when_everyone_is_targeted(contrast):
    _, y, t = contrast
    assert confirmation.gain_vs_random(y, t, np.ones(len(y), bool)) == pytest.approx(0)


def test_real_heterogeneity_is_confirmed(contrast):
    features, y, t = contrast
    result = confirmation.repeated_cross_fit(
        features, y, t, uplift.MODELS["t_learner_logistic"], 0.2,
        folds=3, repeats=3, n_boot=100, random_state=1,
    )
    assert result["confirmed"]
    assert result["ci_low"] > 0
    assert len(result["per_repeat"]) == 3
    share = result["top_share"]
    assert share.shape == y.shape and ((share >= 0) & (share <= 1)).all()
    # Every repeat targets 20% of each fold, so the average share is 20%.
    assert share.mean() == pytest.approx(0.2, abs=0.01)


def test_noise_scores_are_not_confirmed(contrast):
    features, y, t = contrast
    result = confirmation.repeated_cross_fit(
        features, y, t, NoiseModel, 0.2, folds=3, repeats=3, n_boot=100, random_state=1,
    )
    assert not result["confirmed"]
    assert result["ci_low"] < 0 < result["ci_high"]
