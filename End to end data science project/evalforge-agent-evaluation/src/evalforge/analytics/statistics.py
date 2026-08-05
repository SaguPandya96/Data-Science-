"""Statistical helpers, each chosen against a specific question.

Applying a test without a reason is worse than reporting a bare number, because it lends
false authority. Every function here documents *why* it is the right tool for the
question it answers, and the module deliberately reports no p-values: with ~25 metrics
across several subgroups and no multiple-comparison correction, significance claims
would be unjustified. These are descriptive diagnostics.

See ``docs/EVALUATION_METHODOLOGY.md`` §7 for the full rationale and limitations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from evalforge.rng import seeded_numpy

#: Below this many observations, an interval is too wide to be worth reporting and is
#: suppressed rather than shown as if it meant something.
MIN_N_FOR_INTERVAL = 10

#: Bootstrap resamples. 10,000 is enough for stable percentile bounds at this scale.
BOOTSTRAP_RESAMPLES = 10_000


@dataclass(frozen=True)
class Interval:
    """A point estimate with an uncertainty interval.

    Args:
        point: The estimate.
        lower: Lower bound, or ``None`` when the sample was too small.
        upper: Upper bound, or ``None`` when the sample was too small.
        n: Number of observations.
        method: Which procedure produced the bounds.
    """

    point: float
    lower: float | None
    upper: float | None
    n: int
    method: str

    @property
    def has_interval(self) -> bool:
        """Whether bounds were computed."""
        return self.lower is not None and self.upper is not None

    def as_dict(self) -> dict[str, float | int | str | None]:
        """Serialise for reports and dashboards."""
        return {
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "n": self.n,
            "method": self.method,
        }

    def render(self, precision: int = 3) -> str:
        """Human-readable rendering, e.g. ``0.873 [0.812, 0.921]``."""
        if not self.has_interval:
            return f"{self.point:.{precision}f} (n={self.n}, interval suppressed)"
        return (
            f"{self.point:.{precision}f} [{self.lower:.{precision}f}, {self.upper:.{precision}f}]"
        )


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> Interval:
    """95% Wilson score interval for a proportion.

    Chosen over the normal approximation because the interesting pass rates here sit
    near the boundaries — injection resistance around 0.98 — where the normal
    approximation misbehaves badly and can produce bounds above 1.0. Wilson is
    well-behaved at the boundaries and at small *n*, which is exactly this regime.
    """
    if total <= 0:
        return Interval(point=0.0, lower=None, upper=None, n=0, method="wilson")

    proportion = successes / total
    if total < MIN_N_FOR_INTERVAL:
        return Interval(point=proportion, lower=None, upper=None, n=total, method="wilson")

    z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    )
    return Interval(
        point=proportion,
        lower=max(0.0, centre - margin),
        upper=min(1.0, centre + margin),
        n=total,
        method="wilson",
    )


def bootstrap_mean(
    values: list[float],
    confidence: float = 0.95,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 42,
) -> Interval:
    """Percentile bootstrap interval for a mean.

    Used for anything that is not a simple proportion — overall score, latency, cost per
    success — because those have unknown, skewed sampling distributions and bootstrapping
    assumes nothing about their shape.

    *Assumption:* sessions are exchangeable. Defensible for a generated suite, where
    scenarios are drawn independently from seeds. It is **not** a claim about production
    traffic. Resampling is seeded so a stored interval is reproducible.
    """
    if not values:
        return Interval(point=0.0, lower=None, upper=None, n=0, method="bootstrap")

    point = float(np.mean(values))
    if len(values) < MIN_N_FOR_INTERVAL:
        return Interval(point=point, lower=None, upper=None, n=len(values), method="bootstrap")

    rng = seeded_numpy(seed, "bootstrap", len(values))
    array = np.asarray(values, dtype=float)
    samples = rng.choice(array, size=(resamples, len(array)), replace=True)
    means = samples.mean(axis=1)
    alpha = (1 - confidence) / 2
    return Interval(
        point=point,
        lower=float(np.quantile(means, alpha)),
        upper=float(np.quantile(means, 1 - alpha)),
        n=len(values),
        method="bootstrap",
    )


def cohens_h(proportion_a: float, proportion_b: float) -> float:
    """Effect size between two proportions.

    The arcsine transform matters near the boundaries: 0.98 → 0.96 is a far larger
    relative change than 0.50 → 0.48, and a raw percentage-point difference treats them
    as identical. Used for baseline-versus-candidate comparison of rates.

    Conventional reading: 0.2 small, 0.5 medium, 0.8 large.
    """
    phi_a = 2 * math.asin(math.sqrt(max(0.0, min(1.0, proportion_a))))
    phi_b = 2 * math.asin(math.sqrt(max(0.0, min(1.0, proportion_b))))
    return phi_a - phi_b


def cliffs_delta(sample_a: list[float], sample_b: list[float]) -> float:
    """Non-parametric effect size between two score distributions.

    Ordinal and distribution-free, which suits 0..1 session scores — they are bounded,
    heavily skewed toward 1.0, and nowhere near normal, so Cohen's *d* would be
    misleading.

    Returns a value in ``[-1, 1]``: positive means ``sample_a`` tends to exceed
    ``sample_b``. Conventional reading: 0.147 small, 0.33 medium, 0.474 large.
    """
    if not sample_a or not sample_b:
        return 0.0
    left = np.asarray(sample_a, dtype=float)[:, None]
    right = np.asarray(sample_b, dtype=float)[None, :]
    greater = int(np.sum(left > right))
    less = int(np.sum(left < right))
    return (greater - less) / (len(sample_a) * len(sample_b))


def spearman(sample_a: list[float], sample_b: list[float]) -> float:
    """Spearman rank correlation.

    Rank-based because human ratings are ordinal (a 1..5 rubric) and the relationship
    with automated scores need only be monotonic. Pearson would assume interval spacing
    that a rubric scale does not have.

    Returns 0.0 rather than NaN when the sample is too small or degenerate, so callers
    do not have to special-case it.
    """
    if len(sample_a) < 3 or len(sample_b) < 3 or len(sample_a) != len(sample_b):
        return 0.0
    if len(set(sample_a)) < 2 or len(set(sample_b)) < 2:
        return 0.0
    result = stats.spearmanr(sample_a, sample_b)
    value = float(result.statistic)
    return 0.0 if math.isnan(value) else value


def percentile(values: list[float], q: float) -> float:
    """Percentile of a sample, or 0.0 when empty."""
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def rate(successes: int, total: int) -> float:
    """Safe division for a rate, returning 0.0 when the denominator is zero."""
    return successes / total if total else 0.0


def interpret_kappa(value: float) -> str:
    """Landis and Koch's conventional reading of a kappa statistic.

    The labels are a widely used convention, not a law; they are included so a report
    does not leave a bare 0.61 for a reader to interpret unaided.
    """
    if value < 0.0:
        return "worse than chance"
    if value < 0.21:
        return "slight"
    if value < 0.41:
        return "fair"
    if value < 0.61:
        return "moderate"
    if value < 0.81:
        return "substantial"
    return "almost perfect"


def interpret_effect(value: float) -> str:
    """Conventional reading of an absolute effect size."""
    magnitude = abs(value)
    if magnitude < 0.147:
        return "negligible"
    if magnitude < 0.33:
        return "small"
    if magnitude < 0.474:
        return "medium"
    return "large"


def summarize_distribution(values: list[float]) -> dict[str, float]:
    """Point summary of a numeric sample for report tables."""
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=float)
    return {
        "n": float(len(values)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }
