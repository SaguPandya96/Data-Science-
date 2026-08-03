from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .evaluation import nonnegative_predictions

PROMOTION_COLUMNS = ["Promo", "Promo2Active", "PromoActive"]


def apply_scenario(features: pd.DataFrame, scenario: dict[str, Any]) -> pd.DataFrame:
    """Apply a declared planning sensitivity to a copy of forecast features."""
    changed = features.copy()
    kind = scenario["kind"]

    if kind == "baseline":
        return changed
    if kind == "set_promotions":
        value = int(scenario["value"])
        if value not in {0, 1}:
            raise ValueError("Promotion scenario value must be 0 or 1.")
        changed[PROMOTION_COLUMNS] = value
        return changed
    if kind == "scale_recent_demand":
        factor = float(scenario["factor"])
        if factor < 0:
            raise ValueError("Demand scenario factor must be non-negative.")
        lag_columns = [column for column in changed if column.startswith("Sales_Lag_")]
        if not lag_columns:
            raise ValueError("Demand scenario requires at least one Sales_Lag_ feature.")
        changed[lag_columns] = changed[lag_columns] * factor
        return changed
    raise ValueError(f"Unsupported scenario kind: {kind}")


def run_scenarios(
    model: Pipeline,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    scenarios: Sequence[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score all configured scenarios and compare their totals with baseline."""
    if len(features) != len(metadata):
        raise ValueError("Scenario features and metadata must have equal row counts.")
    baseline_names = [item["name"] for item in scenarios if item["kind"] == "baseline"]
    if len(baseline_names) != 1:
        raise ValueError("Scenario configuration must contain exactly one baseline.")

    rows: list[dict[str, float | str]] = []
    daily_frames: list[pd.DataFrame] = []
    metadata = metadata[["Date", "Store"]].reset_index(drop=True)

    for scenario in scenarios:
        changed = apply_scenario(features, scenario)
        predictions = nonnegative_predictions(model.predict(changed))
        rows.append(
            {"Scenario": str(scenario["name"]), "Forecasted Revenue": float(predictions.sum())}
        )
        daily = metadata.copy()
        daily["Scenario"] = str(scenario["name"])
        daily["PredictedSales"] = predictions
        daily_frames.append(daily)

    summary = pd.DataFrame(rows)
    baseline_total = float(
        summary.loc[summary["Scenario"].eq(baseline_names[0]), "Forecasted Revenue"].iloc[0]
    )
    summary["vs Baseline"] = summary["Forecasted Revenue"] - baseline_total
    summary["% Change"] = np.where(
        baseline_total != 0, summary["vs Baseline"] / baseline_total, np.nan
    )
    return summary, pd.concat(daily_frames, ignore_index=True)
