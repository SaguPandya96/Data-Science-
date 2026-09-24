"""Evaluate uplift rankings on held-out randomized data."""

from __future__ import annotations

import numpy as np


def qini_curve(
    score: np.ndarray, outcome: np.ndarray, treated: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative incremental outcomes when targeting customers in score order.

    Returns (fraction targeted, incremental outcomes), both starting at 0.
    """
    order = np.argsort(-np.asarray(score), kind="stable")
    y = np.asarray(outcome, dtype=float)[order]
    t = np.asarray(treated).astype(bool)[order]
    n_t = np.cumsum(t)
    n_c = np.cumsum(~t)
    y_t = np.cumsum(y * t)
    y_c = np.cumsum(y * ~t)
    with np.errstate(divide="ignore", invalid="ignore"):
        incremental = np.where(n_c > 0, y_t - y_c * n_t / n_c, 0.0)
    fraction = np.arange(1, len(y) + 1) / len(y)
    return np.concatenate([[0.0], fraction]), np.concatenate([[0.0], incremental])


def qini_coefficient(score: np.ndarray, outcome: np.ndarray, treated: np.ndarray) -> float:
    """Area between the Qini curve and random targeting, per customer.

    Positive means the ranking puts responsive customers first; zero is random.
    """
    fraction, incremental = qini_curve(score, outcome, treated)
    random_line = fraction * incremental[-1]
    return float(np.trapezoid(incremental - random_line, fraction) / len(outcome))


def uplift_at_fraction(
    score: np.ndarray, outcome: np.ndarray, treated: np.ndarray, fraction: float
) -> float:
    """Treated minus control mean outcome among the top ``fraction`` by score."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    n_top = max(1, int(round(fraction * len(score))))
    top = np.argsort(-np.asarray(score), kind="stable")[:n_top]
    y = np.asarray(outcome, dtype=float)[top]
    t = np.asarray(treated).astype(bool)[top]
    if t.all() or (~t).all():
        return float("nan")
    return float(y[t].mean() - y[~t].mean())


def bootstrap_interval(
    statistic,
    score: np.ndarray,
    outcome: np.ndarray,
    treated: np.ndarray,
    n_boot: int = 200,
    alpha: float = 0.05,
    random_state: int = 0,
) -> tuple[float, float]:
    """Percentile interval for ``statistic(score, outcome, treated)``."""
    rng = np.random.default_rng(random_state)
    score, outcome, treated = map(np.asarray, (score, outcome, treated))
    n = len(score)
    draws = []
    for _ in range(n_boot):
        index = rng.integers(0, n, n)
        draws.append(statistic(score[index], outcome[index], treated[index]))
    low, high = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)
