"""Prediction-time leakage protection tests."""

import pandas as pd
import pytest

from supplylens.features import (
    LEAKAGE_BLOCKLIST,
    MODEL_FEATURES,
    add_shifted_historical_features,
    assert_no_leakage,
)


def test_model_allowlist_excludes_outcome_columns():
    assert not set(MODEL_FEATURES).intersection(LEAKAGE_BLOCKLIST)
    assert_no_leakage(MODEL_FEATURES)
    with pytest.raises(ValueError, match="Prohibited post-outcome"):
        assert_no_leakage([*MODEL_FEATURES, "actual_delivery_date"])


def test_historical_features_are_shifted_and_same_date_blocked(shipments):
    observed = shipments.head(500).copy()
    baseline = add_shifted_historical_features(observed)
    modified = observed.copy()
    current_date = modified.loc[250, "prediction_date"]
    modified.loc[modified["prediction_date"].eq(current_date), "severe_delay"] = 1
    changed = add_shifted_historical_features(modified)

    history_columns = [column for column in baseline if column.startswith("historical_")]
    same_date = baseline["prediction_date"].eq(current_date)
    pd.testing.assert_frame_equal(
        baseline.loc[same_date, history_columns].reset_index(drop=True),
        changed.loc[same_date, history_columns].reset_index(drop=True),
    )


def test_first_date_has_no_group_history(shipments):
    enriched = add_shifted_historical_features(shipments.head(200))
    first_date = enriched["prediction_date"].min()
    count_columns = [
        column for column in enriched if column.startswith("historical_") and column.endswith("_count")
    ]
    assert enriched.loc[enriched["prediction_date"].eq(first_date), count_columns].fillna(0).eq(0).all().all()
