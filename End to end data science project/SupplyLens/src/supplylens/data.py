"""Raw-data ingestion and shipment-level preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

RAW_COLUMNS = [
    "ID",
    "Project Code",
    "PQ #",
    "PO / SO #",
    "ASN/DN #",
    "Country",
    "Managed By",
    "Fulfill Via",
    "Vendor INCO Term",
    "Shipment Mode",
    "PQ First Sent to Client Date",
    "PO Sent to Vendor Date",
    "Scheduled Delivery Date",
    "Delivered to Client Date",
    "Delivery Recorded Date",
    "Product Group",
    "Sub Classification",
    "Vendor",
    "Item Description",
    "Molecule/Test Type",
    "Brand",
    "Dosage",
    "Dosage Form",
    "Unit of Measure (Per Pack)",
    "Line Item Quantity",
    "Line Item Value",
    "Pack Price",
    "Unit Price",
    "Manufacturing Site",
    "First Line Designation",
    "Weight (Kilograms)",
    "Freight Cost (USD)",
    "Line Item Insurance (USD)",
]

STABLE_SHIPMENT_COLUMNS = {
    "project_code": "Project Code",
    "po_so_number": "PO / SO #",
    "country": "Country",
    "managed_by": "Managed By",
    "fulfill_via": "Fulfill Via",
    "vendor_inco_term": "Vendor INCO Term",
    "shipment_mode": "Shipment Mode",
    "product_group": "Product Group",
    "sub_classification": "Sub Classification",
    "supplier": "Vendor",
    "product": "Item Description",
    "molecule_test_type": "Molecule/Test Type",
    "brand": "Brand",
    "dosage_form": "Dosage Form",
    "manufacturing_site": "Manufacturing Site",
    "first_line_designation": "First Line Designation",
}


def read_raw_data(path: str | Path) -> pd.DataFrame:
    """Read the byte-preserved source CSV without mutating it."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {source}. Run `python scripts/download_data.py`."
        )
    return pd.read_csv(source, encoding="utf-8-sig", low_memory=False)


def parse_source_date(series: pd.Series) -> pd.Series:
    """Parse source dates while coercing documented non-date status text."""
    values = series.astype("string").str.strip()
    parsed = pd.to_datetime(values, format="%d-%b-%y", errors="coerce")
    remaining = parsed.isna()
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(
            values.loc[remaining], format="%m/%d/%y", errors="coerce"
        )
    return parsed


def parse_source_numeric(series: pd.Series) -> pd.Series:
    """Parse numeric source values; textual cross-references remain missing."""
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _first_nonempty(series: pd.Series) -> object:
    values = series.dropna()
    if values.empty:
        return pd.NA
    stripped = values.astype("string").str.strip()
    stripped = stripped[stripped.ne("")]
    return stripped.iloc[0] if not stripped.empty else pd.NA


def clean_shipments(raw: pd.DataFrame, severe_delay_days: int = 7) -> pd.DataFrame:
    """Aggregate source shipment lines into one row per ASN/DN shipment.

    Vendor, destination, mode, fulfillment path, actual delivery date, and
    product group are stable within every observed ASN/DN. Ten ASN/DNs carry
    multiple line-level scheduled dates; the latest scheduled date defines the
    commitment for the complete shipment.
    """
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Raw schema missing required columns: {sorted(missing)}")

    frame = raw.copy()
    frame["_scheduled"] = parse_source_date(frame["Scheduled Delivery Date"])
    frame["_delivered"] = parse_source_date(frame["Delivered to Client Date"])
    frame["_recorded"] = parse_source_date(frame["Delivery Recorded Date"])
    frame["_po_sent"] = parse_source_date(frame["PO Sent to Vendor Date"])

    numeric_map = {
        "_quantity": "Line Item Quantity",
        "_value": "Line Item Value",
        "_pack_price": "Pack Price",
        "_unit_price": "Unit Price",
        "_weight": "Weight (Kilograms)",
        "_freight": "Freight Cost (USD)",
        "_insurance": "Line Item Insurance (USD)",
    }
    for cleaned, source in numeric_map.items():
        frame[cleaned] = parse_source_numeric(frame[source])

    grouped = frame.groupby("ASN/DN #", sort=False, dropna=False)
    shipments = pd.DataFrame(index=grouped.size().index)
    shipments.index.name = "shipment_id"
    shipments["source_line_count"] = grouped.size().astype(int)
    shipments["source_line_ids"] = grouped["ID"].agg(
        lambda values: "|".join(values.astype("string"))
    )
    for cleaned, source in STABLE_SHIPMENT_COLUMNS.items():
        shipments[cleaned] = grouped[source].agg(_first_nonempty)

    shipments["scheduled_delivery_date"] = grouped["_scheduled"].max()
    shipments["actual_delivery_date"] = grouped["_delivered"].max()
    shipments["delivery_recorded_date"] = grouped["_recorded"].max()
    shipments["po_sent_date"] = grouped["_po_sent"].min()
    shipments["total_quantity"] = grouped["_quantity"].sum(min_count=1)
    shipments["total_value_usd"] = grouped["_value"].sum(min_count=1)
    shipments["pack_price_median"] = grouped["_pack_price"].median()
    shipments["unit_price_median"] = grouped["_unit_price"].median()
    shipments["weight_kg"] = grouped["_weight"].max()
    shipments["freight_cost_usd"] = grouped["_freight"].max()
    shipments["insurance_usd"] = grouped["_insurance"].sum(min_count=1)
    shipments = shipments.reset_index()

    shipments["delivery_delay_days"] = (
        shipments["actual_delivery_date"] - shipments["scheduled_delivery_date"]
    ).dt.days
    shipments["actual_lead_time_days"] = (
        shipments["actual_delivery_date"] - shipments["po_sent_date"]
    ).dt.days
    shipments["scheduled_lead_time_days"] = (
        shipments["scheduled_delivery_date"] - shipments["po_sent_date"]
    ).dt.days
    for threshold in (0, 3, 7, 14):
        shipments[f"late_gt_{threshold}_days"] = (
            shipments["delivery_delay_days"] > threshold
        ).astype("int8")
    shipments["severe_delay"] = (
        shipments["delivery_delay_days"] > severe_delay_days
    ).astype("int8")
    shipments["prediction_date"] = shipments["scheduled_delivery_date"]

    shipments = shipments.sort_values(
        ["scheduled_delivery_date", "shipment_id"], kind="stable"
    ).reset_index(drop=True)
    return shipments


def write_processed_data(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write the reproducible shipment-level CSV with ISO date strings."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    copy = frame.copy()
    for column in copy.select_dtypes(include=["datetime", "datetimetz"]).columns:
        copy[column] = copy[column].dt.strftime("%Y-%m-%d")
    copy.to_csv(output, index=False)
    return output


def read_processed_data(path: str | Path) -> pd.DataFrame:
    """Read the reproducible shipment-level dataset."""
    date_columns: Iterable[str] = (
        "scheduled_delivery_date",
        "actual_delivery_date",
        "delivery_recorded_date",
        "po_sent_date",
        "prediction_date",
    )
    frame = pd.read_csv(path)
    for column in date_columns:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame

