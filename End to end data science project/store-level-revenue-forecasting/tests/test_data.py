from __future__ import annotations

import pandas as pd
import pytest

from store_revenue_forecasting.data import DataValidationError, merge_inputs, validate_inputs


def test_validation_and_merge_accept_well_formed_inputs(rossmann_frames) -> None:
    sales, stores = rossmann_frames
    validate_inputs(sales, stores)
    merged = merge_inputs(sales, stores)

    assert len(merged) == len(sales)
    assert merged["StoreType"].notna().all()
    assert merged.sort_values(["Store", "Date"]).index.equals(pd.RangeIndex(len(merged)))


def test_duplicate_store_date_is_rejected(rossmann_frames) -> None:
    sales, stores = rossmann_frames
    duplicated = pd.concat([sales, sales.iloc[[0]]], ignore_index=True)

    with pytest.raises(DataValidationError, match="duplicate Store/Date"):
        validate_inputs(duplicated, stores)
