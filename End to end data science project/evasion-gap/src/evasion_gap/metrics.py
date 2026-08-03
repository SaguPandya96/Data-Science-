"""Operating-point metrics.

Moderation systems run at a chosen recall under a false-positive budget, so every
number here is computed at a threshold pinned to a target recall on clean toxic
text -- never at 0.5.
"""

from typing import Tuple

import numpy as np


def threshold_at_recall(clean_toxic_scores: np.ndarray, target_recall: float = 0.95) -> float:
    """Lowest threshold that still catches `target_recall` of clean toxic text.

    Note the failure mode: if the toxic score distribution has a long left tail, a
    high target recall drives this threshold toward zero, and every downstream rate
    becomes insensitive to anything the attack does. Pair it with `threshold_at_fpr`.
    """
    return float(np.quantile(clean_toxic_scores, 1.0 - target_recall))


def threshold_at_fpr(benign_scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """Threshold admitting at most `target_fpr` false positives on benign text.

    This is the operating point a real platform runs at: over-blocking benign users
    is the expensive error, so the false-positive budget sets the threshold and
    recall is whatever falls out.
    """
    return float(np.quantile(benign_scores, 1.0 - target_fpr))


def rate_above(scores: np.ndarray, threshold: float) -> float:
    """Fraction of scores at or above the threshold.

    On toxic text this is recall; on benign text it is the false-positive rate.
    """
    return float((scores >= threshold).mean())


def bootstrap_ci(
    scores: np.ndarray,
    threshold: float,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Percentile bootstrap interval for `rate_above`.

    A few hundred examples carries real sampling noise; reporting a bare point
    estimate over-claims.
    """
    rng = np.random.default_rng(seed)
    n = len(scores)
    draws = rng.choice(scores, size=(n_boot, n), replace=True)
    rates = (draws >= threshold).mean(axis=1)
    lo, hi = np.quantile(rates, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
