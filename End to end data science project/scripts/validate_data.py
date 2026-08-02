"""Validate the immutable raw CSV and create the shipment-level dataset."""

from __future__ import annotations

import json
import sys

from supplylens.config import load_config, resolve_path
from supplylens.data import clean_shipments, read_raw_data, write_processed_data
from supplylens.validation import validate_processed_data, validate_raw_data


def main() -> int:
    config = load_config()
    raw_path = resolve_path(config["data"]["raw_path"])
    processed_path = resolve_path(config["data"]["processed_path"])
    raw = read_raw_data(raw_path)
    raw_result = validate_raw_data(
        raw,
        source_path=raw_path,
        expected_sha256=config["data"]["sha256"],
        expected_rows=int(config["data"]["expected_rows"]),
        expected_columns=int(config["data"]["expected_columns"]),
    )
    raw_result.raise_if_failed()

    shipments = clean_shipments(
        raw, severe_delay_days=int(config["target"]["delay_days_threshold"])
    )
    processed_result = validate_processed_data(shipments)
    processed_result.raise_if_failed()
    write_processed_data(shipments, processed_path)

    report = {
        "raw": raw_result.to_dict(),
        "processed": processed_result.to_dict(),
        "target_prevalence": {
            str(threshold): {
                "positive_shipments": int(shipments[f"late_gt_{threshold}_days"].sum()),
                "prevalence": float(shipments[f"late_gt_{threshold}_days"].mean()),
            }
            for threshold in (0, 3, 7, 14)
        },
        "date_coverage": {
            "scheduled_delivery_min": shipments["scheduled_delivery_date"].min().date().isoformat(),
            "scheduled_delivery_max": shipments["scheduled_delivery_date"].max().date().isoformat(),
            "actual_delivery_min": shipments["actual_delivery_date"].min().date().isoformat(),
            "actual_delivery_max": shipments["actual_delivery_date"].max().date().isoformat(),
        },
        "grain_decision": {
            "raw_grain": "shipment line",
            "modeling_grain": "ASN/DN shipment",
            "raw_rows": int(len(raw)),
            "modeled_shipments": int(len(shipments)),
            "multi_line_shipments": int((shipments["source_line_count"] > 1).sum()),
        },
    }
    output = resolve_path("reports/metrics/data_quality.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Validation passed: {len(raw):,} raw lines -> {len(shipments):,} shipments; "
        f"severe-delay prevalence {shipments['severe_delay'].mean():.3%}."
    )
    for warning in raw_result.warnings:
        print(f"WARNING: {warning}")
    print(f"Wrote {processed_path}")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        raise

