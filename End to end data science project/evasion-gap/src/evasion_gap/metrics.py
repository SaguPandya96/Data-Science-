"""Threshold selection and rate estimation."""

from typing import Tuple

import numpy as np


def threshold_at_recall(clean_toxic_scores: np.ndarray, target_recall: float = 0.95) -> float:
    """Return the threshold that achieves the target recall on clean toxic text.

    When the score distribution has a long lower tail, a high target pushes this
    threshold close to zero. Rates measured there change very little regardless of
    the input, so results should be read alongside threshold_at_fpr.
    """
    return float(np.quantile(clean_toxic_scores, 1.0 - target_recall))


def threshold_at_fpr(benign_scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """Return the threshold that keeps the false positive rate within the budget.

    This matches how moderation systems are usually configured: the acceptable
    rate of incorrectly flagged benign content is fixed first, and recall is
    whatever that threshold produces.
    """
    return float(np.quantile(benign_scores, 1.0 - target_fpr))


def rate_above(scores: np.ndarray, threshold: float) -> float:
    """Return the fraction of scores at or above the threshold.

    On toxic text this is recall. On benign text it is the false positive rate.
    """
    return float((scores >= threshold).mean())


def bootstrap_ci(
    scores: np.ndarray,
    threshold: float,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Return a percentile bootstrap confidence interval for rate_above."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(scores, size=(n_boot, len(scores)), replace=True)
    rates = (draws >= threshold).mean(axis=1)
    lo, hi = np.quantile(rates, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
