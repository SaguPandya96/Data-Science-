from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def chronological_split(
    frame: pd.DataFrame, holdout_weeks: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split on the final configured weeks without shuffling observations."""
    if holdout_weeks <= 0:
        raise ValueError("holdout_weeks must be positive.")
    maximum_date = frame["Date"].max()
    cutoff = maximum_date - pd.Timedelta(weeks=holdout_weeks)
    train = frame.loc[frame["Date"] <= cutoff].copy()
    test = frame.loc[frame["Date"] > cutoff].copy()
    if train.empty or test.empty:
        raise ValueError(
            f"Chronological split produced an empty partition at cutoff {cutoff.date()}."
        )
    return train, test, cutoff


def nonnegative_predictions(predictions: np.ndarray | pd.Series) -> np.ndarray:
    return np.maximum(np.asarray(predictions, dtype=float), 0.0)


def regression_metrics(
    actual: np.ndarray | pd.Series, predictions: np.ndarray | pd.Series
) -> dict[str, float]:
    """Return scale, portfolio, bias, and descriptive fit metrics."""
    y_true = np.asarray(actual, dtype=float)
    y_pred = nonnegative_predictions(predictions)
    denominator = float(np.abs(y_true).sum())
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "WAPE": float(np.abs(y_true - y_pred).sum() / denominator) if denominator else float("nan"),
        "Forecast Bias": float((y_pred - y_true).sum() / denominator)
        if denominator
        else float("nan"),
        "R2": float(r2_score(y_true, y_pred)),
    }


def segment_error(
    metadata: pd.DataFrame,
    actual: np.ndarray | pd.Series,
    predictions: np.ndarray | pd.Series,
    segment: str,
) -> pd.DataFrame:
    """Aggregate error measures for an operational review segment."""
    errors = metadata[[segment]].reset_index(drop=True).copy()
    errors["Actual"] = np.asarray(actual, dtype=float)
    errors["Predicted"] = nonnegative_predictions(predictions)
    errors["AbsError"] = (errors["Predicted"] - errors["Actual"]).abs()

    summary = (
        errors.groupby(segment, dropna=False, observed=False)
        .agg(
            **{
                "Actual Sales": ("Actual", "sum"),
                "Predicted Sales": ("Predicted", "sum"),
                "MAE": ("AbsError", "mean"),
                "Absolute Error": ("AbsError", "sum"),
            }
        )
        .reset_index()
    )
    summary["WAPE"] = summary["Absolute Error"] / summary["Actual Sales"].replace(0, np.nan)
    summary["Forecast Bias"] = (summary["Predicted Sales"] - summary["Actual Sales"]) / summary[
        "Actual Sales"
    ].replace(0, np.nan)
    return summary.drop(columns="Absolute Error").sort_values("WAPE", ascending=False)
