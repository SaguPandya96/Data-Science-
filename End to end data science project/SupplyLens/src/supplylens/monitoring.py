"""Monitoring baselines and drift statistics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def population_stability_index(
    reference: Iterable[float], current: Iterable[float], bins: int = 10
) -> float:
    reference_values = np.asarray(list(reference), dtype=float)
    current_values = np.asarray(list(current), dtype=float)
    reference_values = reference_values[np.isfinite(reference_values)]
    current_values = current_values[np.isfinite(current_values)]
    if not len(reference_values) or not len(current_values):
        return float("nan")
    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference_values, edges)[0] / len(reference_values)
    observed = np.histogram(current_values, edges)[0] / len(current_values)
    expected = np.clip(expected, 1e-6, None)
    observed = np.clip(observed, 1e-6, None)
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def category_frequency_distance(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.fillna("Unknown").astype(str).value_counts(normalize=True)
    cur = current.fillna("Unknown").astype(str).value_counts(normalize=True)
    categories = ref.index.union(cur.index)
    return float(0.5 * (ref.reindex(categories, fill_value=0) - cur.reindex(categories, fill_value=0)).abs().sum())


def build_monitoring_baseline(frame: pd.DataFrame, probabilities: np.ndarray) -> dict:
    numeric = {}
    for column in ("total_quantity", "total_value_usd", "weight_kg", "freight_cost_usd"):
        values = pd.to_numeric(frame[column], errors="coerce")
        numeric[column] = {
            "missing_rate": float(values.isna().mean()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p10": float(values.quantile(0.10)),
            "p50": float(values.quantile(0.50)),
            "p90": float(values.quantile(0.90)),
        }
    categorical = {}
    for column in ("supplier", "country", "shipment_mode", "fulfill_via", "product_group"):
        categorical[column] = frame[column].fillna("Unknown").astype(str).value_counts(normalize=True).head(20).to_dict()
    scores = np.asarray(probabilities, dtype=float)
    return {
        "row_count": int(len(frame)),
        "numeric": numeric,
        "categorical_top_frequencies": categorical,
        "prediction": {
            "mean_score": float(scores.mean()),
            "score_std": float(scores.std()),
            "p10": float(np.quantile(scores, 0.10)),
            "p50": float(np.quantile(scores, 0.50)),
            "p90": float(np.quantile(scores, 0.90)),
        },
    }

