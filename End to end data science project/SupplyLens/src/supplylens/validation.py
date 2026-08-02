"""Data-quality checks with explicit errors and retained warnings."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from supplylens.data import RAW_COLUMNS, parse_source_date, parse_source_numeric


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            raise ValueError("Data validation failed:\n- " + "\n- ".join(self.errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_raw_data(
    frame: pd.DataFrame,
    *,
    source_path: str | Path,
    expected_sha256: str,
    expected_rows: int,
    expected_columns: int,
) -> ValidationResult:
    result = ValidationResult()
    result.metrics.update(
        {
            "row_count": int(len(frame)),
            "column_count": int(frame.shape[1]),
            "duplicate_rows": int(frame.duplicated().sum()),
            "duplicate_ids": int(frame["ID"].duplicated().sum()) if "ID" in frame else None,
            "missing_cells": int(frame.isna().sum().sum()),
        }
    )

    observed_hash = file_sha256(source_path)
    result.metrics["sha256"] = observed_hash
    if observed_hash.lower() != expected_sha256.lower():
        result.errors.append(
            f"Checksum mismatch: expected {expected_sha256}, observed {observed_hash}"
        )
    if len(frame) != expected_rows:
        result.errors.append(f"Expected {expected_rows} rows, observed {len(frame)}")
    if frame.shape[1] != expected_columns:
        result.errors.append(
            f"Expected {expected_columns} columns, observed {frame.shape[1]}"
        )
    missing_columns = set(RAW_COLUMNS) - set(frame.columns)
    if missing_columns:
        result.errors.append(f"Missing required columns: {sorted(missing_columns)}")
        return result
    if result.metrics["duplicate_rows"]:
        result.errors.append(f"Found {result.metrics['duplicate_rows']} duplicate raw rows")
    if result.metrics["duplicate_ids"]:
        result.errors.append(f"Found {result.metrics['duplicate_ids']} duplicate line IDs")

    dates = {}
    for column in (
        "Scheduled Delivery Date",
        "Delivered to Client Date",
        "Delivery Recorded Date",
    ):
        dates[column] = parse_source_date(frame[column])
        invalid = int(dates[column].isna().sum())
        result.metrics[f"invalid_{column.lower().replace(' ', '_')}"] = invalid
        if invalid:
            result.errors.append(f"{column} has {invalid} unparseable values")

    po_sent = parse_source_date(frame["PO Sent to Vendor Date"])
    result.metrics["po_sent_date_available_rows"] = int(po_sent.notna().sum())
    result.metrics["po_sent_date_unavailable_rows"] = int(po_sent.isna().sum())

    numeric_columns = (
        "Line Item Quantity",
        "Line Item Value",
        "Pack Price",
        "Unit Price",
    )
    for column in numeric_columns:
        values = parse_source_numeric(frame[column])
        missing = int(values.isna().sum())
        negative = int((values < 0).sum())
        result.metrics[f"missing_{column.lower().replace(' ', '_')}"] = missing
        result.metrics[f"negative_{column.lower().replace(' ', '_')}"] = negative
        if missing:
            result.errors.append(f"{column} has {missing} nonnumeric values")
        if negative:
            result.errors.append(f"{column} has {negative} negative values")

    for column in ("Weight (Kilograms)", "Freight Cost (USD)", "Line Item Insurance (USD)"):
        values = parse_source_numeric(frame[column])
        result.metrics[f"numeric_{column.lower().replace(' ', '_')}_rows"] = int(values.notna().sum())
        if int((values < 0).sum()):
            result.errors.append(f"{column} contains negative values")

    recorded_before_delivery = int(
        (dates["Delivery Recorded Date"] < dates["Delivered to Client Date"]).sum()
    )
    extreme_early = int(
        (
            (dates["Scheduled Delivery Date"] - dates["Delivered to Client Date"]).dt.days
            > 365
        ).sum()
    )
    result.metrics["recorded_before_delivered_rows"] = recorded_before_delivery
    result.metrics["more_than_365_days_early_rows"] = extreme_early
    if recorded_before_delivery:
        result.warnings.append(
            f"{recorded_before_delivery} rows were recorded before the delivered date"
        )
    if extreme_early:
        result.warnings.append(
            f"{extreme_early} rows arrived more than 365 days before their scheduled date"
        )

    expected_modes = {"Air", "Truck", "Air Charter", "Ocean", "N/A"}
    unexpected_modes = sorted(set(frame["Shipment Mode"].dropna()) - expected_modes)
    if unexpected_modes:
        result.errors.append(f"Unexpected Shipment Mode values: {unexpected_modes}")
    expected_fulfillment = {"Direct Drop", "From RDC"}
    unexpected_fulfillment = sorted(
        set(frame["Fulfill Via"].dropna()) - expected_fulfillment
    )
    if unexpected_fulfillment:
        result.errors.append(f"Unexpected Fulfill Via values: {unexpected_fulfillment}")

    result.metrics["source_line_grain"] = "shipment line"
    result.metrics["unique_asn_dn"] = int(frame["ASN/DN #"].nunique(dropna=False))
    result.metrics["multi_line_asn_dn"] = int(
        (frame.groupby("ASN/DN #").size() > 1).sum()
    )
    return result


def validate_processed_data(frame: pd.DataFrame) -> ValidationResult:
    result = ValidationResult()
    required = {
        "shipment_id",
        "scheduled_delivery_date",
        "actual_delivery_date",
        "delivery_delay_days",
        "severe_delay",
        "supplier",
        "country",
        "shipment_mode",
        "total_quantity",
        "total_value_usd",
    }
    missing = required - set(frame)
    if missing:
        result.errors.append(f"Processed data missing columns: {sorted(missing)}")
        return result
    result.metrics = {
        "shipment_count": int(len(frame)),
        "duplicate_shipment_ids": int(frame["shipment_id"].duplicated().sum()),
        "missing_scheduled_dates": int(frame["scheduled_delivery_date"].isna().sum()),
        "missing_actual_dates": int(frame["actual_delivery_date"].isna().sum()),
        "severe_delay_count": int(frame["severe_delay"].sum()),
        "severe_delay_rate": float(frame["severe_delay"].mean()),
    }
    if result.metrics["duplicate_shipment_ids"]:
        result.errors.append("Processed shipment IDs must be unique")
    if result.metrics["missing_scheduled_dates"] or result.metrics["missing_actual_dates"]:
        result.errors.append("Processed data contains missing required outcome dates")
    if not frame["severe_delay"].isin([0, 1]).all():
        result.errors.append("severe_delay must contain only 0 and 1")
    if (frame["total_quantity"] < 0).any() or (frame["total_value_usd"] < 0).any():
        result.errors.append("Processed quantity and value fields must be non-negative")
    return result

