"""Production scoring contract and risk-ranking output."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from supplylens.features import LEAKAGE_BLOCKLIST, prepare_model_frame
from supplylens.modeling import ModelBundle

SCORING_REQUIRED_COLUMNS = [
    "shipment_id",
    "scheduled_delivery_date",
    "po_sent_date",
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
    "total_quantity",
    "total_value_usd",
    "pack_price_median",
    "unit_price_median",
    "weight_kg",
    "freight_cost_usd",
    "insurance_usd",
    "source_line_count",
    "scheduled_lead_time_days",
]


def validate_scoring_input(frame: pd.DataFrame) -> None:
    missing = set(SCORING_REQUIRED_COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"Scoring input missing required columns: {sorted(missing)}")
    prohibited = set(frame.columns).intersection(LEAKAGE_BLOCKLIST)
    if prohibited:
        raise ValueError(f"Scoring input contains prohibited outcome columns: {sorted(prohibited)}")
    if frame.empty:
        raise ValueError("Scoring input contains no rows")
    if frame["shipment_id"].isna().any() or frame["shipment_id"].duplicated().any():
        raise ValueError("shipment_id must be present and unique")
    dates = pd.to_datetime(frame["scheduled_delivery_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("scheduled_delivery_date contains invalid or missing dates")
    for column in ("total_quantity", "total_value_usd", "source_line_count"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (values < 0).any():
            raise ValueError(f"{column} must be non-negative numeric data")
    prepare_model_frame(frame)


def score_frame(bundle: ModelBundle, frame: pd.DataFrame) -> pd.DataFrame:
    validate_scoring_input(frame)
    output = frame.copy()
    output["predicted_severe_delay_probability"] = bundle.predict_proba(output)
    output["risk_rank"] = (
        output["predicted_severe_delay_probability"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    review_count = max(1, math.ceil(len(output) * bundle.review_capacity))
    output["review_flag"] = (output["risk_rank"] <= review_count).astype(int)
    output["model_name"] = bundle.model_name
    output["calibration_method"] = bundle.calibrator.method
    if not np.isfinite(output["predicted_severe_delay_probability"]).all():
        raise RuntimeError("Model produced non-finite probabilities")
    return output.sort_values("risk_rank", kind="stable").reset_index(drop=True)

