from __future__ import annotations

import numpy as np

from store_revenue_forecasting.data import merge_inputs
from store_revenue_forecasting.evaluation import chronological_split, regression_metrics
from store_revenue_forecasting.features import build_features


def test_chronological_split_never_overlaps(rossmann_frames) -> None:
    sales, stores = rossmann_frames
    featured = build_features(merge_inputs(sales, stores), [7, 30])
    train, test, cutoff = chronological_split(featured, holdout_weeks=2)

    assert train["Date"].max() <= cutoff
    assert test["Date"].min() > cutoff
    assert train["Date"].max() < test["Date"].min()


def test_perfect_predictions_have_zero_error() -> None:
    actual = np.array([0.0, 10.0, 20.0])
    metrics = regression_metrics(actual, actual)

    assert metrics["RMSE"] == 0
    assert metrics["MAE"] == 0
    assert metrics["WAPE"] == 0
    assert metrics["Forecast Bias"] == 0
    assert metrics["R2"] == 1
