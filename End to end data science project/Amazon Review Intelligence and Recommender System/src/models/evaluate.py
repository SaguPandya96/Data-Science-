"""Evaluation for the rating predictor and the recommender.

Ranking evaluation uses the standard sampled leave-one-out protocol: each user's held-out
(most recent) review is the one relevant item, scored against N sampled negatives.

Two negative-sampling protocols are supported, and the difference matters:

  uniform    - negatives drawn uniformly from the catalogue. This is the common default,
               but it *flatters popularity-based rankers*: the held-out true item is
               usually a popular one, while uniform negatives are mostly obscure long-tail
               items, so "rank by popularity" gets an artificial edge.

  popularity - negatives drawn proportional to item popularity. This removes that confound
               and is the more honest test of whether a model has learned anything beyond
               "this item is popular".

Reporting both is deliberate: the gap between them quantifies how much of a ranker's
apparent skill is really just popularity bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rating_rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"rmse": rmse, "mae": mae}


def _ndcg_at_k(rank: int | None, k: int) -> float:
    if rank is None or rank >= k:
        return 0.0
    return 1.0 / np.log2(rank + 2)


def ranking_eval(
    score_fn,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_items: np.ndarray,
    sampling: str = "uniform",
    n_negatives: int = 100,
    k: int = 10,
    max_users: int | None = 2000,
    random_state: int = 42,
) -> dict:
    """score_fn(user_id, candidate_item_ids) -> array of scores, higher = better."""
    rng = np.random.default_rng(random_state)
    seen_by_user = train_df.groupby("user_id")["parent_asin"].apply(set)
    all_items = np.asarray(all_items)

    if max_users is not None and len(test_df) > max_users:
        test_df = test_df.sample(n=max_users, random_state=random_state)

    if sampling == "popularity":
        counts = train_df["parent_asin"].value_counts()
        weights = np.array([counts.get(i, 0) for i in all_items], dtype=float)
    else:
        weights = None

    hits, ndcgs, evaluated = 0, [], 0
    for _, row in test_df.iterrows():
        user_id, held_out = row["user_id"], row["parent_asin"]
        excluded = seen_by_user.get(user_id, set()) | {held_out}
        mask = ~np.isin(all_items, list(excluded))
        pool = all_items[mask]
        if len(pool) < n_negatives:
            continue

        if weights is None:
            negatives = rng.choice(pool, size=n_negatives, replace=False)
        else:
            pool_weights = weights[mask]
            total = pool_weights.sum()
            if total <= 0:
                continue
            negatives = rng.choice(pool, size=n_negatives, replace=False, p=pool_weights / total)

        candidates = list(negatives) + [held_out]
        scores = np.asarray(score_fn(user_id, candidates), dtype=float)
        order = np.argsort(-scores)
        rank = int(np.where(order == len(candidates) - 1)[0][0])

        evaluated += 1
        hits += rank < k
        ndcgs.append(_ndcg_at_k(rank, k))

    return {
        "hit_rate_at_k": round(hits / evaluated, 4) if evaluated else 0.0,
        "ndcg_at_k": round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0,
        "n_evaluated": evaluated,
        "k": k,
        "sampling": sampling,
    }


def build_baselines(train_df: pd.DataFrame, model) -> dict:
    """The rankers we compare. A popularity baseline is included deliberately - on sparse
    review data it is a genuinely strong competitor, not a strawman."""
    popularity = train_df["parent_asin"].value_counts()
    popularity_norm = popularity / popularity.max()
    rng = np.random.default_rng(0)

    return {
        "random": lambda u, c: rng.random(len(c)),
        "popularity": lambda u, c: np.array([popularity.get(x, 0) for x in c], dtype=float),
        "matrix_factorization": lambda u, c: model.cf_model.score_items(u, c),
        "hybrid (MF + popularity)": lambda u, c: (
            model.cf_model.score_items(u, c) / 5.0
            + 0.5 * np.array([popularity_norm.get(x, 0) for x in c], dtype=float)
        ),
    }


def compare_rankers(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 10,
    max_users: int = 2000,
) -> pd.DataFrame:
    all_items = train_df["parent_asin"].unique()
    rows = []
    for sampling in ["uniform", "popularity"]:
        for name, score_fn in build_baselines(train_df, model).items():
            metrics = ranking_eval(
                score_fn, train_df, test_df, all_items,
                sampling=sampling, k=k, max_users=max_users,
            )
            rows.append({
                "ranker": name,
                "negative_sampling": sampling,
                f"HR@{k}": metrics["hit_rate_at_k"],
                f"NDCG@{k}": metrics["ndcg_at_k"],
                "n_users": metrics["n_evaluated"],
            })
    return pd.DataFrame(rows)


def evaluate_by_user_activity(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 10,
    max_users_per_segment: int = 1500,
) -> pd.DataFrame:
    """Does personalization help more for users with more history? Segmenting the
    evaluation this way is what justifies (or refutes) a hybrid routing strategy."""
    all_items = train_df["parent_asin"].unique()
    n_train = train_df["user_id"].value_counts()

    test_df = test_df.copy()
    test_df["n_train"] = test_df["user_id"].map(n_train).fillna(0).astype(int)

    segments = {
        "cold start (0)": test_df[test_df["n_train"] == 0],
        "1 interaction": test_df[test_df["n_train"] == 1],
        "2-4 interactions": test_df[test_df["n_train"].between(2, 4)],
        "5+ interactions": test_df[test_df["n_train"] >= 5],
    }
    scorers = build_baselines(train_df, model)

    rows = []
    for segment_name, subset in segments.items():
        if subset.empty:
            continue
        for name, score_fn in scorers.items():
            metrics = ranking_eval(
                score_fn, train_df, subset, all_items,
                sampling="uniform", k=k, max_users=max_users_per_segment,
            )
            rows.append({
                "segment": segment_name,
                "population": len(subset),
                "ranker": name,
                f"HR@{k}": metrics["hit_rate_at_k"],
                f"NDCG@{k}": metrics["ndcg_at_k"],
            })
    return pd.DataFrame(rows)
