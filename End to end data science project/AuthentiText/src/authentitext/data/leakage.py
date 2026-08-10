"""Exact and normalized leakage analysis for processed text partitions."""

from __future__ import annotations

import gzip
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from authentitext.data.cleaning import make_record_id

ANALYZER_VERSION = 1
REPORT_SCHEMA_VERSION = 1
REQUIRED_RECORD_FIELDS = {
    "content_id",
    "dataset_id",
    "partition",
    "raw_label",
    "record_id",
    "revision",
    "source",
    "target",
    "text",
}


class LeakageError(RuntimeError):
    """Raised when processed inputs cannot support a valid leakage analysis."""


def normalize_text(text: str) -> str:
    """Normalize compatibility forms, case, and whitespace for equality checks."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mask_key(mask: int, partition_bits: dict[str, int]) -> str:
    return "|".join(name for name, bit in partition_bits.items() if mask & bit)


def _summarize_groups(
    groups: dict[str, list[Any]], partition_bits: dict[str, int]
) -> dict[str, Any]:
    duplicate_groups = [group for group in groups.values() if group[1] > 1]
    cross_partition_groups = [group for group in duplicate_groups if group[0] & (group[0] - 1)]
    cross_sets: Counter[str] = Counter(
        _mask_key(group[0], partition_bits) for group in cross_partition_groups
    )
    conflicting_sets: Counter[str] = Counter(
        _mask_key(group[0], partition_bits)
        for group in duplicate_groups
        if group[2].bit_count() > 1
    )
    return {
        "unique_values": len(groups),
        "duplicate_groups": len(duplicate_groups),
        "rows_in_duplicate_groups": sum(group[1] for group in duplicate_groups),
        "duplicate_extra_rows": sum(group[1] - 1 for group in duplicate_groups),
        "within_partition_duplicate_groups": len(duplicate_groups) - len(cross_partition_groups),
        "cross_partition_groups": len(cross_partition_groups),
        "rows_in_cross_partition_groups": sum(group[1] for group in cross_partition_groups),
        "cross_partition_set_counts": {key: cross_sets[key] for key in sorted(cross_sets)},
        "conflicting_target_groups": sum(group[2].bit_count() > 1 for group in duplicate_groups),
        "conflicting_target_partition_set_counts": {
            key: conflicting_sets[key] for key in sorted(conflicting_sets)
        },
    }


def analyze_processed_data(cleaning_report: dict[str, Any], output_root: Path) -> dict[str, Any]:
    """Analyze exact and normalized equality without emitting record content."""
    partitions = [partition["partition"] for partition in cleaning_report["partitions"]]
    partition_bits = {name: 1 << index for index, name in enumerate(partitions)}
    all_partitions_mask = sum(partition_bits.values())

    exact_groups: dict[str, list[Any]] = {}
    normalized_groups: dict[str, list[Any]] = {}
    source_masks: dict[str, int] = {}
    partition_counts: Counter[str] = Counter()
    record_ids: set[str] = set()
    duplicate_record_ids = 0
    rows = 0
    raw_to_target = cleaning_report["configuration"]["raw_to_canonical_target"]

    input_identities = []
    for partition_report in cleaning_report["partitions"]:
        partition = partition_report["partition"]
        bit = partition_bits[partition]
        path = output_root / partition_report["output_relative_path"]
        input_identities.append(
            {
                "partition": partition,
                "relative_path": partition_report["output_relative_path"],
                "bytes": partition_report["output_bytes"],
                "sha256": partition_report["output_sha256"],
            }
        )

        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                if set(record) != REQUIRED_RECORD_FIELDS:
                    raise LeakageError(f"{path}:{line_number} has unexpected record fields")
                if record["partition"] != partition:
                    raise LeakageError(
                        f"{path}:{line_number} has partition {record['partition']!r}"
                    )
                if record["dataset_id"] != cleaning_report["dataset_id"]:
                    raise LeakageError(f"{path}:{line_number} has the wrong dataset")
                if record["revision"] != cleaning_report["revision"]:
                    raise LeakageError(f"{path}:{line_number} has the wrong revision")

                content_id = record["content_id"]
                if content_id != _sha256_text(record["text"]):
                    raise LeakageError(f"{path}:{line_number} has an invalid content ID")
                if record["raw_label"] not in raw_to_target:
                    raise LeakageError(f"{path}:{line_number} has an unknown raw label")
                if record["target"] != raw_to_target[record["raw_label"]]:
                    raise LeakageError(f"{path}:{line_number} has an invalid target mapping")

                record_id = record["record_id"]
                expected_record_id = make_record_id(
                    dataset_id=record["dataset_id"],
                    revision=record["revision"],
                    partition=partition,
                    source=record["source"],
                    raw_label=record["raw_label"],
                    content_id=content_id,
                )
                if record_id != expected_record_id:
                    raise LeakageError(f"{path}:{line_number} has an invalid record ID")
                if record_id in record_ids:
                    duplicate_record_ids += 1
                record_ids.add(record_id)

                target_mask = 1 << int(record["target"])
                exact_group = exact_groups.setdefault(content_id, [0, 0, 0])
                exact_group[0] |= bit
                exact_group[1] += 1
                exact_group[2] |= target_mask

                normalized_id = _sha256_text(normalize_text(record["text"]))
                normalized_group = normalized_groups.setdefault(
                    normalized_id,
                    [0, 0, 0, content_id, False],
                )
                normalized_group[0] |= bit
                normalized_group[1] += 1
                normalized_group[2] |= target_mask
                if normalized_group[3] != content_id:
                    normalized_group[4] = True

                source = record["source"]
                source_masks[source] = source_masks.get(source, 0) | bit
                partition_counts[partition] += 1
                rows += 1

    exact_summary = _summarize_groups(exact_groups, partition_bits)
    normalized_summary = _summarize_groups(normalized_groups, partition_bits)
    normalized_summary["normalization_only_cross_partition_groups"] = sum(
        group[0] & (group[0] - 1) and group[4]
        for group in normalized_groups.values()
        if group[1] > 1
    )

    source_partition_sets: Counter[str] = Counter(
        _mask_key(mask, partition_bits) for mask in source_masks.values()
    )
    expected_rows = cleaning_report["aggregate"]["rows_written"]
    violations = []
    if rows != expected_rows:
        violations.append(f"observed {rows} rows; expected {expected_rows}")
    if duplicate_record_ids:
        violations.append(f"found {duplicate_record_ids} duplicate record IDs")
    warnings = []
    if exact_summary["conflicting_target_groups"]:
        warnings.append("exact duplicate groups have conflicting targets")
    if normalized_summary["conflicting_target_groups"]:
        warnings.append("normalized duplicate groups have conflicting targets")

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "analyzer_version": ANALYZER_VERSION,
        "dataset_id": cleaning_report["dataset_id"],
        "revision": cleaning_report["revision"],
        "inputs": input_identities,
        "configuration": {
            "exact_key": "processed content_id = sha256(utf8(text))",
            "normalized_key": ("sha256(utf8(collapse_whitespace(casefold(NFKC(text)))))"),
            "near_duplicate_analysis": "not_run_in_this_report",
        },
        "population": {
            "rows": rows,
            "partition_counts": {
                partition: partition_counts[partition] for partition in partitions
            },
            "unique_record_ids": len(record_ids),
        },
        "exact": exact_summary,
        "normalized_exact": normalized_summary,
        "source_overlap": {
            "distinct_source_values": len(source_masks),
            "sources_spanning_all_partitions": sum(
                mask == all_partitions_mask for mask in source_masks.values()
            ),
            "partition_set_counts": {
                key: source_partition_sets[key] for key in sorted(source_partition_sets)
            },
        },
        "validation": {
            "status": "pass" if not violations else "fail",
            "violations": violations,
            "warnings": warnings,
        },
    }


def render_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
