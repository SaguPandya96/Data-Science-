from __future__ import annotations

import pytest

from store_revenue_forecasting.data import merge_inputs
from store_revenue_forecasting.features import build_features, lag_column


def test_lags_only_use_prior_store_sales(rossmann_frames) -> None:
    sales, stores = rossmann_frames
    featured = build_features(merge_inputs(sales, stores), [7, 30])
    first_store = featured.loc[featured["Store"].eq("1")].reset_index(drop=True)

    assert first_store.loc[0, lag_column(7)] == 0
    assert first_store.loc[1, lag_column(7)] == first_store.loc[0, "Sales"]
    assert first_store.loc[7, lag_column(7)] == pytest.approx(first_store.loc[:6, "Sales"].mean())


def test_current_target_cannot_change_current_lag(rossmann_frames) -> None:
    sales, stores = rossmann_frames
    original = build_features(merge_inputs(sales, stores), [7])
    changed_sales = sales.copy()
    target_index = changed_sales.index[(changed_sales["Store"].eq(1))][10]
    changed_sales.loc[target_index, "Sales"] *= 10
    changed = build_features(merge_inputs(changed_sales, stores), [7])

    row_mask = original["Store"].eq("1") & original["Date"].eq(changed_sales.loc[target_index, "Date"])
    assert original.loc[row_mask, lag_column(7)].iloc[0] == changed.loc[
        row_mask, lag_column(7)
    ].iloc[0]
    assert "Customers" not in [column for column in original if column.startswith("Sales_Lag_")]
