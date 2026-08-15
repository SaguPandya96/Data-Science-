"""Rebuild the audited transformer training partition without reading test data."""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any

from authentitext.data.cleaning import open_deterministic_gzip, sha256_file

REQUIRED_RECORD_FIELDS = {"record_id", "partition", "target", "text"}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class TransformerTrainError(RuntimeError):
    """Raised when train-only materialization violates the audited contract."""


def load_train_decisions(path: Path) -> dict[str, Any]:
    """Load and validate the text-free train exclusion decisions."""
    try:
        decisions = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TransformerTrainError(f"Could not read train decisions: {error}") from error

    if not isinstance(decisions, dict) or decisions.get("schema_version") != 1:
        raise TransformerTrainError("Unexpected train decision schema")
    if decisions.get("partition") != "train":
        raise TransformerTrainError("Train decisions must target only the train partition")

    record_ids = decisions.get("record_ids_to_drop")
    if (
        not isinstance(record_ids, list)
        or not record_ids
        or record_ids != sorted(record_ids)
        or len(record_ids) != len(set(record_ids))
        or not all(
            isinstance(value, str) and SHA256_PATTERN.fullmatch(value) for value in record_ids
        )
    ):
        raise TransformerTrainError("Train exclusion record IDs are invalid")

    for identity_name in ("expected_input", "expected_output"):
        identity = decisions.get(identity_name)
        if not isinstance(identity, dict):
            raise TransformerTrainError(f"Missing {identity_name} identity")
        if not isinstance(identity.get("rows"), int) or identity["rows"] <= 0:
            raise TransformerTrainError(f"Invalid {identity_name} row count")
        if not isinstance(identity.get("bytes"), int) or identity["bytes"] <= 0:
            raise TransformerTrainError(f"Invalid {identity_name} byte count")
        if not isinstance(identity.get("sha256"), str) or not SHA256_PATTERN.fullmatch(
            identity["sha256"]
        ):
            raise TransformerTrainError(f"Invalid {identity_name} SHA-256")

    if (
        decisions["expected_input"]["rows"] - len(record_ids)
        != decisions["expected_output"]["rows"]
    ):
        raise TransformerTrainError("Train decision row counts do not reconcile")
    return decisions


def _verify_identity(path: Path, identity: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise TransformerTrainError(f"Missing {label}: {path}")
    if path.stat().st_size != identity["bytes"]:
        raise TransformerTrainError(f"{label} byte count does not match the audited identity")
    if sha256_file(path) != identity["sha256"]:
        raise TransformerTrainError(f"{label} SHA-256 does not match the audited identity")


def materialize_transformer_train(
    cleaned_train_path: Path,
    output_path: Path,
    decisions: dict[str, Any],
) -> dict[str, Any]:
    """Apply only prespecified train exclusions and verify the final identity."""
    _verify_identity(cleaned_train_path, decisions["expected_input"], "cleaned train input")
    drop_ids = set(decisions["record_ids_to_drop"])
    encountered_drop_ids: set[str] = set()
    seen_record_ids: set[str] = set()
    rows_seen = 0
    rows_written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")

    try:
        with gzip.open(cleaned_train_path, "rt", encoding="utf-8", newline="") as source:
            with open_deterministic_gzip(temporary_path) as destination:
                for line_number, line in enumerate(source, start=1):
                    rows_seen += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise TransformerTrainError(
                            f"Cleaned train line {line_number} is invalid JSON"
                        ) from error
                    if not isinstance(record, dict) or not REQUIRED_RECORD_FIELDS <= record.keys():
                        raise TransformerTrainError(
                            f"Cleaned train line {line_number} is missing required fields"
                        )
                    if record["partition"] != "train" or record["target"] not in (0, 1):
                        raise TransformerTrainError(
                            f"Cleaned train line {line_number} has invalid role metadata"
                        )
                    record_id = record["record_id"]
                    if not isinstance(record_id, str) or not SHA256_PATTERN.fullmatch(record_id):
                        raise TransformerTrainError(
                            f"Cleaned train line {line_number} has an invalid record ID"
                        )
                    if record_id in seen_record_ids:
                        raise TransformerTrainError(
                            f"Duplicate cleaned train record ID: {record_id}"
                        )
                    seen_record_ids.add(record_id)
                    if record_id in drop_ids:
                        encountered_drop_ids.add(record_id)
                        continue
                    destination.write(
                        json.dumps(
                            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                        )
                        + "\n"
                    )
                    rows_written += 1
        if rows_seen != decisions["expected_input"]["rows"]:
            raise TransformerTrainError(
                "Cleaned train row count does not match the audited identity"
            )
        if encountered_drop_ids != drop_ids:
            raise TransformerTrainError("Not every audited train exclusion was encountered")
        if rows_written != decisions["expected_output"]["rows"]:
            raise TransformerTrainError("Materialized train row count does not match")
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    _verify_identity(output_path, decisions["expected_output"], "materialized train output")

    return {
        "schema_version": 1,
        "dataset_id": decisions["dataset_id"],
        "revision": decisions["revision"],
        "partition": "train",
        "rows_seen": rows_seen,
        "rows_dropped": len(encountered_drop_ids),
        "rows_written": rows_written,
        "output_bytes": output_path.stat().st_size,
        "output_sha256": sha256_file(output_path),
        "test_data_read": False,
        "source_text_in_report": False,
        "status": "pass",
    }
