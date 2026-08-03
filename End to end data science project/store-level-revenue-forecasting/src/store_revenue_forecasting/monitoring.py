from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def build_monitoring_baseline(
    training_features: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Any]:
    """Build lightweight reference distributions for scheduled forecast monitoring."""
    numeric: dict[str, dict[str, float | None]] = {}
    for column in numeric_features:
        series = pd.to_numeric(training_features[column], errors="coerce")
        numeric[column] = {
            "mean": _finite_or_none(series.mean()),
            "std": _finite_or_none(series.std()),
            "min": _finite_or_none(series.min()),
            "max": _finite_or_none(series.max()),
            "missing_rate": float(series.isna().mean()),
        }

    categorical: dict[str, dict[str, Any]] = {}
    for column in categorical_features:
        series = training_features[column].fillna("<MISSING>").astype(str)
        frequencies = series.value_counts(normalize=True).head(20)
        categorical[column] = {
            "unique_count": int(series.nunique()),
            "missing_rate": float(training_features[column].isna().mean()),
            "top_frequencies": {str(key): float(value) for key, value in frequencies.items()},
        }

    return {
        "row_count": int(len(training_features)),
        "numeric_features": numeric,
        "categorical_features": categorical,
    }
