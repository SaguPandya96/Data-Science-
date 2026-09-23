"""Checks and estimates for a randomized experiment readout."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class SRMResult:
    observed: dict[str, int]
    expected: dict[str, float]
    chi2: float
    p_value: float
    mismatch: bool


def sample_ratio_mismatch(
    arms: pd.Series, expected_allocation: dict[str, float], alpha: float = 0.001
) -> SRMResult:
    """Chi-square test that arm sizes match the planned allocation."""
    counts = arms.value_counts()
    unexpected = set(counts.index) - set(expected_allocation)
    if unexpected:
        raise ValueError(f"Arms not in the planned allocation: {sorted(unexpected)}")
    names = list(expected_allocation)
    shares = np.array([expected_allocation[name] for name in names], dtype=float)
    shares = shares / shares.sum()
    observed = np.array([counts.get(name, 0) for name in names], dtype=float)
    expected = shares * observed.sum()
    chi2, p_value = stats.chisquare(observed, expected)
    return SRMResult(
        observed={name: int(value) for name, value in zip(names, observed)},
        expected={name: float(value) for name, value in zip(names, expected)},
        chi2=float(chi2),
        p_value=float(p_value),
        mismatch=bool(p_value < alpha),
    )


def standardized_mean_differences(features: pd.DataFrame, treated: pd.Series) -> pd.Series:
    """Absolute SMD per pre-treatment feature; above ~0.1 suggests imbalance."""
    treated = treated.astype(bool).to_numpy()
    values = features.to_numpy(dtype=float)
    mean_t, mean_c = values[treated].mean(axis=0), values[~treated].mean(axis=0)
    var_t, var_c = values[treated].var(axis=0, ddof=1), values[~treated].var(axis=0, ddof=1)
    pooled = np.sqrt((var_t + var_c) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        smd = np.where(pooled > 0, np.abs(mean_t - mean_c) / pooled, 0.0)
    return pd.Series(smd, index=features.columns).sort_values(ascending=False)


def minimum_detectable_effect(
    baseline_rate: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Smallest absolute lift in a rate a two-sided test can detect at this sample size."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * np.sqrt(2 * baseline_rate * (1 - baseline_rate) / n_per_arm))


def required_sample_size(
    baseline_rate: float, absolute_lift: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Users per arm needed to detect ``absolute_lift`` over ``baseline_rate``."""
    if absolute_lift <= 0:
        raise ValueError("absolute_lift must be positive")
    p1, p2 = baseline_rate, baseline_rate + absolute_lift
    z_alpha, z_beta = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    numerator = (
        z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return int(np.ceil(numerator / absolute_lift**2))


@dataclass(frozen=True)
class EffectEstimate:
    control_mean: float
    treatment_mean: float
    absolute_lift: float
    relative_lift: float
    ci_low: float
    ci_high: float
    p_value: float
    standard_error: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def difference_in_means(
    outcome: pd.Series | np.ndarray, treated: pd.Series | np.ndarray, alpha: float = 0.05
) -> EffectEstimate:
    """Welch-style difference in means with a normal-approximation interval."""
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treated).astype(bool)
    y_t, y_c = y[t], y[~t]
    lift = y_t.mean() - y_c.mean()
    se = np.sqrt(y_t.var(ddof=1) / len(y_t) + y_c.var(ddof=1) / len(y_c))
    z = stats.norm.ppf(1 - alpha / 2)
    if se > 0:
        p_value = 2 * stats.norm.sf(abs(lift / se))
    else:
        p_value = 1.0 if lift == 0 else 0.0
    return EffectEstimate(
        control_mean=float(y_c.mean()),
        treatment_mean=float(y_t.mean()),
        absolute_lift=float(lift),
        relative_lift=float(lift / y_c.mean()) if y_c.mean() != 0 else float("nan"),
        ci_low=float(lift - z * se),
        ci_high=float(lift + z * se),
        p_value=float(p_value),
        standard_error=float(se),
    )


def cuped_adjust(
    outcome: pd.Series | np.ndarray, covariate: pd.Series | np.ndarray
) -> np.ndarray:
    """Remove the part of the outcome explained by a pre-experiment covariate.

    theta is estimated on the pooled sample, which keeps the adjusted difference
    in means unbiased under randomization while shrinking its variance.
    """
    y = np.asarray(outcome, dtype=float)
    x = np.asarray(covariate, dtype=float)
    variance = x.var(ddof=1)
    if variance == 0:
        return y.copy()
    theta = np.cov(y, x, ddof=1)[0, 1] / variance
    return y - theta * (x - x.mean())


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values for a family of comparisons."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[name] = running
    return adjusted
