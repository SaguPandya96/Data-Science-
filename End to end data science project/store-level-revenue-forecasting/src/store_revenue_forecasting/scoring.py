from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import DataValidationError, merge_inputs, read_inputs
from .evaluation import nonnegative_predictions
from .features import build_features, feature_columns

FUTURE_REQUIRED = {
    "Store",
    "Date",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
}


def read_future_plan(path: Path) -> pd.DataFrame:
    """Read a one-day-ahead operating plan from CSV."""
    future = pd.read_csv(path, parse_dates=["Date"], dtype={"StateHoliday": "string"})
    missing = FUTURE_REQUIRED.difference(future.columns)
    if missing:
        raise DataValidationError(f"Future plan is missing required columns: {sorted(missing)}")
    if future.empty:
        raise DataValidationError("Future plan must contain at least one store row.")
    if future[["Store", "Date"]].isna().any().any():
        raise DataValidationError("Future plan contains missing Store or Date keys.")
    if future.duplicated(["Store", "Date"]).any():
        raise DataValidationError("Future plan contains duplicate Store/Date observations.")
    if future["Date"].nunique() != 1:
        raise DataValidationError(
            "This scoring contract accepts exactly one forecast date. "
            "Use recursive forecasting for a multi-day plan."
        )
    return future


def prepare_scoring_features(
    history: pd.DataFrame,
    future: pd.DataFrame,
    stores: pd.DataFrame,
    lag_windows: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one-day-ahead features from actual history and a future operating plan."""
    forecast_date = future["Date"].iloc[0]
    if forecast_date <= history["Date"].max():
        raise DataValidationError("Forecast date must be later than every historical date.")

    future_store_ids = set(future["Store"])
    unknown_metadata = future_store_ids.difference(stores["Store"])
    if unknown_metadata:
        raise DataValidationError(
            f"Future plan contains stores missing from metadata: {sorted(unknown_metadata)}"
        )
    missing_history = future_store_ids.difference(history["Store"])
    if missing_history:
        raise DataValidationError(
            "Version 1 requires prior sales for every forecast store. Missing history for: "
            f"{sorted(missing_history)}"
        )

    future_rows = future.copy()
    future_rows["DayOfWeek"] = future_rows["Date"].dt.dayofweek + 1
    future_rows["Sales"] = np.nan
    future_rows["Customers"] = np.nan

    combined = pd.concat([history, future_rows], ignore_index=True, sort=False)
    featured = build_features(merge_inputs(combined, stores), lag_windows)
    forecast_rows = featured.loc[featured["Date"].eq(forecast_date)].copy()
    if len(forecast_rows) != len(future):
        raise DataValidationError("Feature build changed the number of future plan rows.")

    columns = feature_columns(lag_windows)
    return forecast_rows[columns], forecast_rows[["Date", "Store"]]


def score_future_plan(
    model_path: Path,
    history_path: Path,
    future_path: Path,
    store_path: Path,
    lag_windows: Sequence[int],
    output_path: Path,
) -> pd.DataFrame:
    """Load a trusted model artifact and score one future date."""
    history, stores = read_inputs(history_path, store_path)
    future = read_future_plan(future_path)
    features, metadata = prepare_scoring_features(history, future, stores, lag_windows)

    model = joblib.load(model_path)
    predictions = nonnegative_predictions(model.predict(features))
    scored = metadata.reset_index(drop=True).copy()
    scored["PredictedSales"] = predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    return scored
