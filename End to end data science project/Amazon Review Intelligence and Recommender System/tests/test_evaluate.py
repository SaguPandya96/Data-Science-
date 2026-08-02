import numpy as np
import pandas as pd

from src.models.evaluate import _ndcg_at_k, ranking_eval, rating_rmse_mae


def test_rmse_and_mae_are_zero_for_perfect_predictions():
    y = np.array([1.0, 3.0, 5.0])
    metrics = rating_rmse_mae(y, y.copy())

    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0


def test_rmse_penalizes_large_errors_more_than_mae():
    y_true = np.array([1.0, 1.0, 1.0, 1.0])
    y_pred = np.array([1.0, 1.0, 1.0, 5.0])
    metrics = rating_rmse_mae(y_true, y_pred)

    assert metrics["rmse"] > metrics["mae"]


def test_ndcg_is_highest_at_rank_zero():
    assert _ndcg_at_k(0, k=10) == 1.0
    assert _ndcg_at_k(0, k=10) > _ndcg_at_k(5, k=10)


def test_ndcg_is_zero_when_item_falls_outside_k():
    assert _ndcg_at_k(10, k=10) == 0.0
    assert _ndcg_at_k(None, k=10) == 0.0


def _ranking_fixture():
    train = pd.DataFrame({
        "user_id": ["U1"] * 3 + ["U2"] * 3,
        "parent_asin": ["I1", "I2", "I3", "I1", "I4", "I5"],
    })
    test = pd.DataFrame({"user_id": ["U1", "U2"], "parent_asin": ["I9", "I8"]})
    all_items = np.array([f"I{n}" for n in range(1, 30)])
    return train, test, all_items


def test_ranking_eval_gives_perfect_score_to_an_oracle_ranker():
    train, test, all_items = _ranking_fixture()
    held_out = dict(zip(test["user_id"], test["parent_asin"], strict=True))

    def oracle(user_id, candidates):
        return np.array([1.0 if c == held_out[user_id] else 0.0 for c in candidates])

    result = ranking_eval(oracle, train, test, all_items, n_negatives=20, k=10)

    assert result["hit_rate_at_k"] == 1.0
    assert result["ndcg_at_k"] == 1.0


def test_ranking_eval_never_samples_items_the_user_already_saw():
    """A ranker that scores a seen item highest must not be rewarded - seen items should
    never appear as candidates in the first place."""
    train, test, all_items = _ranking_fixture()
    seen_by_user = train.groupby("user_id")["parent_asin"].apply(set).to_dict()
    leaked = []

    def spy(user_id, candidates):
        leaked.extend(set(candidates) & seen_by_user[user_id])
        return np.zeros(len(candidates))

    ranking_eval(spy, train, test, all_items, n_negatives=20, k=10)

    assert leaked == []
