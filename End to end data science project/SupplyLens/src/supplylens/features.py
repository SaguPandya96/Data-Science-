"""Prediction-time feature engineering and leakage controls."""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = [
    "country",
    "managed_by",
    "fulfill_via",
    "vendor_inco_term",
    "shipment_mode",
    "product_group",
    "sub_classification",
    "supplier",
    "molecule_test_type",
    "brand",
    "dosage_form",
    "manufacturing_site",
    "first_line_designation",
    "scheduled_month",
    "scheduled_quarter",
    "scheduled_day_of_week",
]

NUMERIC_FEATURES = [
    "log_total_quantity",
    "log_total_value_usd",
    "log_pack_price_median",
    "log_unit_price_median",
    "log_weight_kg",
    "log_freight_cost_usd",
    "log_insurance_usd",
    "source_line_count",
    "scheduled_lead_time_days",
    "weight_missing",
    "freight_missing",
    "po_date_missing",
    "peak_period",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

LEAKAGE_BLOCKLIST = {
    "actual_delivery_date",
    "delivery_recorded_date",
    "delivery_delay_days",
    "actual_lead_time_days",
    "severe_delay",
    "late_gt_0_days",
    "late_gt_3_days",
    "late_gt_7_days",
    "late_gt_14_days",
    "delivery_status",
    "final_delivery_status",
    "outcome",
    "target",
}


def assert_no_leakage(columns: list[str] | pd.Index) -> None:
    normalized = {str(column).strip().lower() for column in columns}
    prohibited = sorted(normalized & LEAKAGE_BLOCKLIST)
    if prohibited:
        raise ValueError(f"Prohibited post-outcome columns in model features: {prohibited}")


def prepare_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic features available by schedule commitment."""
    required = {
        "scheduled_delivery_date",
        "total_quantity",
        "total_value_usd",
        "pack_price_median",
        "unit_price_median",
        "weight_kg",
        "freight_cost_usd",
        "insurance_usd",
        "source_line_count",
        "scheduled_lead_time_days",
        *CATEGORICAL_FEATURES[:-3],
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Feature input is missing required columns: {sorted(missing)}")

    features = frame.copy()
    scheduled = pd.to_datetime(features["scheduled_delivery_date"], errors="coerce")
    features["scheduled_month"] = scheduled.dt.month.astype("Int64").astype("string")
    features["scheduled_quarter"] = scheduled.dt.quarter.astype("Int64").astype("string")
    features["scheduled_day_of_week"] = scheduled.dt.dayofweek.astype("Int64").astype("string")
    features["peak_period"] = scheduled.dt.month.isin([10, 11, 12]).astype(int)

    log_sources = {
        "log_total_quantity": "total_quantity",
        "log_total_value_usd": "total_value_usd",
        "log_pack_price_median": "pack_price_median",
        "log_unit_price_median": "unit_price_median",
        "log_weight_kg": "weight_kg",
        "log_freight_cost_usd": "freight_cost_usd",
        "log_insurance_usd": "insurance_usd",
    }
    for output, source in log_sources.items():
        numeric = pd.to_numeric(features[source], errors="coerce").clip(lower=0)
        features[output] = np.log1p(numeric)

    features["weight_missing"] = features["weight_kg"].isna().astype(int)
    features["freight_missing"] = features["freight_cost_usd"].isna().astype(int)
    if "po_sent_date" in features:
        features["po_date_missing"] = features["po_sent_date"].isna().astype(int)
    else:
        features["po_date_missing"] = features["scheduled_lead_time_days"].isna().astype(int)

    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype("string").fillna("Unknown")
    output = features[MODEL_FEATURES].copy()
    assert_no_leakage(output.columns)
    return output


def add_shifted_historical_features(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = ("supplier", "country", "product_group"),
    date_column: str = "prediction_date",
    target_column: str = "severe_delay",
    smoothing: float = 20.0,
    initial_rate: float = 0.05,
) -> pd.DataFrame:
    """Add date-blocked, shifted historical counts and smoothed delay rates.

    Records on the same date do not contribute to each other's history. These
    features are implemented and tested but excluded from the production model
    because the source lacks the timestamp when the scheduled commitment was
    originally recorded.
    """
    result = frame.copy()
    result[date_column] = pd.to_datetime(result[date_column], errors="raise")
    working = result[[date_column, target_column, *group_columns]].copy()
    working["_row_id"] = np.arange(len(working))

    daily_global = (
        working.groupby(date_column, as_index=False)[target_column]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values(date_column)
    )
    daily_global["prior_sum"] = daily_global["sum"].cumsum() - daily_global["sum"]
    daily_global["prior_count"] = (
        daily_global["count"].cumsum() - daily_global["count"]
    )
    daily_global["global_prior_rate"] = (
        daily_global["prior_sum"] + smoothing * initial_rate
    ) / (daily_global["prior_count"] + smoothing)
    global_rate = daily_global.set_index(date_column)["global_prior_rate"]

    for group in group_columns:
        daily = (
            working.groupby([group, date_column], dropna=False)[target_column]
            .agg(["sum", "count"])
            .reset_index()
            .sort_values([group, date_column])
        )
        daily["prior_sum"] = daily.groupby(group, dropna=False)["sum"].cumsum() - daily["sum"]
        daily["prior_count"] = (
            daily.groupby(group, dropna=False)["count"].cumsum() - daily["count"]
        )
        daily["fallback"] = daily[date_column].map(global_rate).fillna(initial_rate)
        daily[f"historical_{group}_count"] = daily["prior_count"].astype(float)
        daily[f"historical_{group}_delay_rate"] = (
            daily["prior_sum"] + smoothing * daily["fallback"]
        ) / (daily["prior_count"] + smoothing)
        result = result.merge(
            daily[
                [
                    group,
                    date_column,
                    f"historical_{group}_count",
                    f"historical_{group}_delay_rate",
                ]
            ],
            on=[group, date_column],
            how="left",
            validate="many_to_one",
        )
    return result

