"""Turn uplift scores into a targeting decision under a mailing budget."""

from __future__ import annotations

import numpy as np
import pandas as pd

from offerlift.evaluation import uplift_at_fraction


def _average_effect(outcome: np.ndarray, treated: np.ndarray) -> float:
    return float(outcome[treated].mean() - outcome[~treated].mean())


def budget_table(
    scores: dict[str, np.ndarray],
    outcome: np.ndarray,
    treated: np.ndarray,
    fractions: list[float],
    per_customers: int = 1000,
    n_boot: int = 500,
    alpha: float = 0.05,
    random_state: int = 0,
) -> pd.DataFrame:
    """Expected incremental outcomes per ``per_customers`` customers for each policy.

    Random targeting earns the average effect on every mailed customer, so the
    model only adds value where its row beats the ``random`` row at the same budget.
    ``gain_vs_random`` is that difference; its interval comes from a paired bootstrap
    (model and random are re-estimated on the same resample), so an interval that
    includes zero means the model has not shown it beats random targeting.
    """
    outcome = np.asarray(outcome, dtype=float)
    treated = np.asarray(treated).astype(bool)
    scores = {name: np.asarray(score) for name, score in scores.items()}
    rng = np.random.default_rng(random_state)
    n = len(outcome)

    point = {"random": {f: _average_effect(outcome, treated) for f in fractions}}
    for name, score in scores.items():
        point[name] = {f: uplift_at_fraction(score, outcome, treated, f) for f in fractions}

    draws = {name: {f: [] for f in fractions} for name in point}
    for _ in range(n_boot):
        index = rng.integers(0, n, n)
        y, t = outcome[index], treated[index]
        effect = _average_effect(y, t)
        for f in fractions:
            draws["random"][f].append(effect)
            for name, score in scores.items():
                draws[name][f].append(uplift_at_fraction(score[index], y, t, f))

    quantiles = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    rows = []
    for name in point:
        for f in fractions:
            mailed = f * per_customers
            lift = np.asarray(draws[name][f])
            gain = lift - np.asarray(draws["random"][f])
            lift_low, lift_high = np.nanpercentile(lift, quantiles)
            gain_low, gain_high = np.nanpercentile(gain, quantiles)
            rows.append(
                {
                    "policy": name,
                    "budget_fraction": f,
                    "uplift_in_targeted": point[name][f],
                    "uplift_ci_low": float(lift_low),
                    "uplift_ci_high": float(lift_high),
                    "incremental_per_customers": mailed * point[name][f],
                    "incremental_ci_low": float(mailed * lift_low),
                    "incremental_ci_high": float(mailed * lift_high),
                    "gain_vs_random_per_customers": mailed
                    * (point[name][f] - point["random"][f]),
                    "gain_ci_low": float(mailed * gain_low),
                    "gain_ci_high": float(mailed * gain_high),
                }
            )
    return pd.DataFrame(rows).sort_values(["budget_fraction", "policy"], ignore_index=True)
