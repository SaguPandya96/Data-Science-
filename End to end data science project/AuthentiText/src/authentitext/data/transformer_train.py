"""Rebuild audited modeling partitions without reading test data."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from authentitext.data.cleaning import open_deterministic_gzip, sha256_file

REQUIRED_RECORD_FIELDS = {"record_id", "partition", "target", "text"}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class TransformerTrainError(RuntimeError):
    """Raised when train-only materialization violates the audited contract."""


def load_partition_decisions(path: Path) -> dict[str, Any]:
    """Load and validate text-free train or validation exclusion decisions."""
    try:
        decisions = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TransformerTrainError(f"Could not read partition decisions: {error}") from error

    if not isinstance(decisions, dict) or decisions.get("schema_version") != 1:
        raise TransformerTrainError("Unexpected partition decision schema")
    if decisions.get("partition") not in {"train", "validation"}:
        raise TransformerTrainError("Decisions must target only train or validation")

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
        raise TransformerTrainError("Partition exclusion record IDs are invalid")

    for identity_name in ("expected_input", "expected_output"):
        identity = decisions.get(identity_name)
        if not isinstance(identity, dict):
            raise TransformerTrainError(f"Missing {identity_name} identity")
        if not isinstance(identity.get("rows"), int) or identity["rows"] <= 0:
            raise TransformerTrainError(f"Invalid {identity_name} row count")
        if (
            not isinstance(identity.get("reference_gzip_bytes"), int)
            or identity["reference_gzip_bytes"] <= 0
        ):
            raise TransformerTrainError(f"Invalid {identity_name} reference byte count")
        for digest_name in ("content_sha256", "reference_gzip_sha256"):
            if not isinstance(identity.get(digest_name), str) or not SHA256_PATTERN.fullmatch(
                identity[digest_name]
            ):
                raise TransformerTrainError(f"Invalid {identity_name} {digest_name}")

    if (
        decisions["expected_input"]["rows"] - len(record_ids)
        != decisions["expected_output"]["rows"]
    ):
        raise TransformerTrainError("Partition decision row counts do not reconcile")
    return decisions


def load_train_decisions(path: Path) -> dict[str, Any]:
    """Load decisions and require the training partition for probe compatibility."""
    decisions = load_partition_decisions(path)
    if decisions["partition"] != "train":
        raise TransformerTrainError("Expected training decisions")
    return decisions


def sha256_gzip_content(path: Path) -> str:
    """Hash decompressed gzip bytes so identity is independent of zlib output."""
    if not path.is_file():
        raise TransformerTrainError(f"Missing gzip file: {path}")
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_transformer_partition(
    cleaned_partition_path: Path,
    output_path: Path,
    decisions: dict[str, Any],
) -> dict[str, Any]:
    """Apply only prespecified exclusions and verify the final content identity."""
    partition = decisions["partition"]
    if partition not in {"train", "validation"}:
        raise TransformerTrainError("Materialization cannot read a sealed partition")
    if not cleaned_partition_path.is_file():
        raise TransformerTrainError(f"Missing cleaned {partition} input: {cleaned_partition_path}")
    drop_ids = set(decisions["record_ids_to_drop"])
    encountered_drop_ids: set[str] = set()
    seen_record_ids: set[str] = set()
    rows_seen = 0
    rows_written = 0
    input_content_digest = hashlib.sha256()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")

    try:
        with gzip.open(cleaned_partition_path, "rt", encoding="utf-8", newline="") as source:
            with open_deterministic_gzip(temporary_path) as destination:
                for line_number, line in enumerate(source, start=1):
                    rows_seen += 1
                    input_content_digest.update(line.encode("utf-8"))
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise TransformerTrainError(
                            f"Cleaned {partition} line {line_number} is invalid JSON"
                        ) from error
                    if not isinstance(record, dict) or not REQUIRED_RECORD_FIELDS <= record.keys():
                        raise TransformerTrainError(
                            f"Cleaned {partition} line {line_number} is missing required fields"
                        )
                    if record["partition"] != partition or record["target"] not in (0, 1):
                        raise TransformerTrainError(
                            f"Cleaned {partition} line {line_number} has invalid role metadata"
                        )
                    record_id = record["record_id"]
                    if not isinstance(record_id, str) or not SHA256_PATTERN.fullmatch(record_id):
                        raise TransformerTrainError(
                            f"Cleaned {partition} line {line_number} has an invalid record ID"
                        )
                    if record_id in seen_record_ids:
                        raise TransformerTrainError(
                            f"Duplicate cleaned {partition} record ID: {record_id}"
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
                f"Cleaned {partition} row count does not match the audited identity"
            )
        if encountered_drop_ids != drop_ids:
            raise TransformerTrainError(f"Not every audited {partition} exclusion was encountered")
        if input_content_digest.hexdigest() != decisions["expected_input"]["content_sha256"]:
            raise TransformerTrainError(
                f"Cleaned {partition} content does not match the audited identity"
            )
        if rows_written != decisions["expected_output"]["rows"]:
            raise TransformerTrainError(f"Materialized {partition} row count does not match")
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    output_content_sha256 = sha256_gzip_content(output_path)
    if output_content_sha256 != decisions["expected_output"]["content_sha256"]:
        raise TransformerTrainError(f"Materialized {partition} content does not match")

    return {
        "schema_version": 1,
        "dataset_id": decisions["dataset_id"],
        "revision": decisions["revision"],
        "partition": partition,
        "rows_seen": rows_seen,
        "rows_dropped": len(encountered_drop_ids),
        "rows_written": rows_written,
        "output_content_sha256": output_content_sha256,
        "output_gzip_bytes": output_path.stat().st_size,
        "output_gzip_sha256": sha256_file(output_path),
        "test_data_read": False,
        "source_text_in_report": False,
        "status": "pass",
    }


def materialize_transformer_train(
    cleaned_train_path: Path,
    output_path: Path,
    decisions: dict[str, Any],
) -> dict[str, Any]:
    """Materialize train while retaining the original probe-facing API."""
    if decisions.get("partition") != "train":
        raise TransformerTrainError("Expected training decisions")
    return materialize_transformer_partition(cleaned_train_path, output_path, decisions)
