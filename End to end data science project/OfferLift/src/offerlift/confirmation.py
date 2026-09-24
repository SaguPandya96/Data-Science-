"""Repeated cross-fitting to confirm that a model beats random targeting at one budget.

Aggregation across repeats follows Chernozhukov, Demirer, Duflo and Fernández-Val,
"Generic Machine Learning Inference on Heterogeneous Treatment Effects": median
estimate, median of the (1 - alpha/2) interval bounds, and 2 x median p-value.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def cross_fitted_scores(
    features: pd.DataFrame,
    outcome: np.ndarray,
    treated: np.ndarray,
    make_model: Callable[[], object],
    folds: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold uplift scores and the fold each customer was scored in."""
    outcome = np.asarray(outcome)
    treated = np.asarray(treated)
    scores = np.empty(len(outcome), dtype=float)
    fold_ids = np.empty(len(outcome), dtype=int)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    for fold, (train, held_out) in enumerate(splitter.split(features, treated * 2 + outcome)):
        model = make_model().fit(features.iloc[train], outcome[train], treated[train])
        scores[held_out] = model.predict_uplift(features.iloc[held_out])
        fold_ids[held_out] = fold
    return scores, fold_ids


def top_within_folds(scores: np.ndarray, fold_ids: np.ndarray, fraction: float) -> np.ndarray:
    """Mark the top ``fraction`` by score inside each fold.

    Each fold was scored by a different fitted model, so scores are only ranked
    against others from the same model.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    top = np.zeros(len(scores), dtype=bool)
    for fold in np.unique(fold_ids):
        members = np.flatnonzero(fold_ids == fold)
        n_top = max(1, int(round(fraction * len(members))))
        order = np.argsort(-scores[members], kind="stable")
        top[members[order[:n_top]]] = True
    return top


def gain_vs_random(outcome: np.ndarray, treated: np.ndarray, top: np.ndarray) -> float:
    """Uplift among targeted customers minus the population uplift (random targeting)."""
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treated).astype(bool)
    top = np.asarray(top).astype(bool)
    targeted = y[top & t].mean() - y[top & ~t].mean()
    population = y[t].mean() - y[~t].mean()
    return float(targeted - population)


def bootstrap_gain(
    outcome: np.ndarray,
    treated: np.ndarray,
    top: np.ndarray,
    level: float,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Percentile interval at ``level`` and a two-sided bootstrap p-value for the gain."""
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treated).astype(bool)
    top = np.asarray(top).astype(bool)
    n = len(y)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        index = rng.integers(0, n, n)
        draws[b] = gain_vs_random(y[index], t[index], top[index])
    tail = (1 - level) / 2
    low, high = np.quantile(draws, [tail, 1 - tail])
    p_value = min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))
    return float(low), float(high), float(p_value)


def repeated_cross_fit(
    features: pd.DataFrame,
    outcome: np.ndarray,
    treated: np.ndarray,
    make_model: Callable[[], object],
    fraction: float,
    folds: int = 5,
    repeats: int = 20,
    n_boot: int = 400,
    alpha: float = 0.05,
    random_state: int = 0,
    per_customers: int = 1000,
) -> dict:
    """Gain over random targeting at ``fraction``, aggregated over repeated cross-fits.

    Gains are reported as extra outcomes per ``per_customers`` customers in the whole
    list, i.e. ``fraction * per_customers`` mailed customers times the per-customer gain.
    ``top_share`` is, for each customer, the share of repeats in which they were targeted.
    """
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**31 - 1, repeats)
    mailed = fraction * per_customers
    per_repeat = []
    top_count = np.zeros(len(outcome))
    for seed in seeds:
        scores, fold_ids = cross_fitted_scores(
            features, outcome, treated, make_model, folds, int(seed)
        )
        top = top_within_folds(scores, fold_ids, fraction)
        top_count += top
        gain = gain_vs_random(outcome, treated, top)
        low, high, p_value = bootstrap_gain(
            outcome, treated, top, 1 - alpha / 2, n_boot, np.random.default_rng(int(seed))
        )
        per_repeat.append(
            {
                "seed": int(seed),
                "gain_per_customers": mailed * gain,
                "ci_low": mailed * low,
                "ci_high": mailed * high,
                "p_value": p_value,
            }
        )
    table = pd.DataFrame(per_repeat)
    return {
        "budget_fraction": fraction,
        "folds": folds,
        "repeats": repeats,
        "gain_per_customers": float(table["gain_per_customers"].median()),
        "ci_low": float(table["ci_low"].median()),
        "ci_high": float(table["ci_high"].median()),
        "p_value": float(min(1.0, 2 * table["p_value"].median())),
        "confirmed": bool(table["ci_low"].median() > 0),
        "share_of_repeats_positive": float((table["gain_per_customers"] > 0).mean()),
        "per_repeat": per_repeat,
        "top_share": top_count / repeats,
    }
