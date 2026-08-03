from __future__ import annotations

import pandas as pd
import pytest

from store_revenue_forecasting.data import DataValidationError, merge_inputs
from store_revenue_forecasting.features import build_features, feature_columns
from store_revenue_forecasting.scoring import prepare_scoring_features, read_future_plan


def test_future_features_use_latest_actual_history(rossmann_frames) -> None:
    sales, stores = rossmann_frames
    forecast_date = sales["Date"].max() + pd.Timedelta(days=1)
    future = pd.DataFrame(
        {
            "Store": stores["Store"],
            "Date": forecast_date,
            "Open": 1,
            "Promo": 0,
            "StateHoliday": "0",
            "SchoolHoliday": 0,
        }
    )

    scoring_features, metadata = prepare_scoring_features(sales, future, stores, [7, 30])
    historical_features = build_features(merge_inputs(sales, stores), [7, 30])
    store_one_history = historical_features.loc[historical_features["Store"].eq("1")]

    assert len(scoring_features) == len(stores)
    assert metadata["Date"].eq(forecast_date).all()
    assert scoring_features.columns.tolist() == feature_columns([7, 30])
    assert scoring_features.loc[metadata["Store"].eq("1"), "Sales_Lag_7_RollingAvg"].iloc[
        0
    ] == pytest.approx(store_one_history["Sales"].tail(7).mean())


def test_future_plan_rejects_multiple_dates(tmp_path) -> None:
    path = tmp_path / "future.csv"
    pd.DataFrame(
        {
            "Store": [1, 1],
            "Date": ["2024-03-01", "2024-03-02"],
            "Open": [1, 1],
            "Promo": [0, 0],
            "StateHoliday": ["0", "0"],
            "SchoolHoliday": [0, 0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(DataValidationError, match="exactly one forecast date"):
        read_future_plan(path)
