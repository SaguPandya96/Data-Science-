"""Deterministic cleaning for pinned MAGE development partitions."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

CLEANER_VERSION = 1
REPORT_SCHEMA_VERSION = 1
REQUIRED_COLUMNS = ("text", "label", "src")
EXCLUDED_SOURCE_PREFIXES = ("wp_",)
EXCLUSION_REASON = "shared_upstream_domain_with_sealed_external_corpus"


class CleaningError(RuntimeError):
    """Raised when input data cannot be transformed without ambiguity."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def make_content_id(text: str) -> str:
    """Identify exact decoded text without storing it in metadata."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_record_id(
    *,
    dataset_id: str,
    revision: str,
    partition: str,
    source: str,
    raw_label: str,
    content_id: str,
) -> str:
    """Derive a stable record ID from pinned source and content fields."""
    identity = "\0".join((dataset_id, revision, partition, source, raw_label, content_id))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


@contextmanager
def open_deterministic_gzip(path: Path) -> Iterator[io.TextIOWrapper]:
    """Open a gzip text writer whose bytes are stable across runs."""
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_handle,
            mtime=0,
        ) as gzip_handle:
            with io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="\n") as text_handle:
                yield text_handle


def clean_partition(
    *,
    input_path: Path,
    output_path: Path,
    dataset_id: str,
    revision: str,
    partition: str,
    raw_to_target: dict[str, int],
) -> dict[str, Any]:
    """Stream one CSV partition to deterministic compressed JSON Lines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    rows_seen = 0
    rows_written = 0
    rows_excluded = 0
    excluded_raw_labels: Counter[str] = Counter()
    output_targets: Counter[str] = Counter()
    output_raw_labels: Counter[str] = Counter()
    output_sources: set[str] = set()
    record_ids: set[str] = set()

    try:
        with open_deterministic_gzip(temporary) as output_handle:
            with input_path.open("r", encoding="utf-8-sig", newline="") as input_handle:
                reader = csv.DictReader(input_handle)
                if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
                    raise CleaningError(
                        f"{input_path} has columns {reader.fieldnames!r}; "
                        f"expected {REQUIRED_COLUMNS!r}"
                    )

                for row_number, row in enumerate(reader, start=2):
                    rows_seen += 1
                    if None in row or any(row.get(field) is None for field in REQUIRED_COLUMNS):
                        raise CleaningError(f"{input_path}:{row_number} is malformed")

                    text = row["text"]
                    raw_label = row["label"]
                    source = row["src"]
                    if not text.strip() or not raw_label.strip() or not source.strip():
                        raise CleaningError(f"{input_path}:{row_number} has a blank required value")
                    if raw_label not in raw_to_target:
                        raise CleaningError(
                            f"{input_path}:{row_number} has unknown label {raw_label!r}"
                        )

                    if source.startswith(EXCLUDED_SOURCE_PREFIXES):
                        rows_excluded += 1
                        excluded_raw_labels[raw_label] += 1
                        continue

                    content_id = make_content_id(text)
                    record_id = make_record_id(
                        dataset_id=dataset_id,
                        revision=revision,
                        partition=partition,
                        source=source,
                        raw_label=raw_label,
                        content_id=content_id,
                    )
                    if record_id in record_ids:
                        raise CleaningError(
                            f"{input_path}:{row_number} produces duplicate record ID {record_id}"
                        )
                    record_ids.add(record_id)

                    target = raw_to_target[raw_label]
                    record = {
                        "content_id": content_id,
                        "dataset_id": dataset_id,
                        "partition": partition,
                        "raw_label": raw_label,
                        "record_id": record_id,
                        "revision": revision,
                        "source": source,
                        "target": target,
                        "text": text,
                    }
                    output_handle.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                    output_handle.write("\n")
                    rows_written += 1
                    output_targets[str(target)] += 1
                    output_raw_labels[raw_label] += 1
                    output_sources.add(source)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    if rows_seen != rows_written + rows_excluded:
        temporary.unlink(missing_ok=True)
        raise CleaningError(f"{partition} row reconciliation failed")

    temporary.replace(output_path)
    return {
        "partition": partition,
        "input_relative_path": input_path.name,
        "output_relative_path": output_path.name,
        "rows_seen": rows_seen,
        "rows_written": rows_written,
        "rows_excluded": rows_excluded,
        "excluded_raw_label_counts": _sorted_counts(excluded_raw_labels),
        "output_raw_label_counts": _sorted_counts(output_raw_labels),
        "output_target_counts": _sorted_counts(output_targets),
        "distinct_output_sources": len(output_sources),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": sha256_file(output_path),
    }


def clean_manifest(manifest: dict[str, Any], input_root: Path, output_root: Path) -> dict[str, Any]:
    """Clean all files in a validated acquisition manifest."""
    semantics = manifest.get("label_semantics", {})
    raw_to_target = semantics.get("canonical_machine_positive_mapping")
    if not isinstance(raw_to_target, dict) or set(raw_to_target.values()) != {0, 1}:
        raise CleaningError("Manifest has an invalid canonical label mapping")

    partition_reports = []
    for entry in manifest["files"]:
        input_path = input_root / Path(entry["relative_path"])
        output_path = output_root / f"{entry['name']}.jsonl.gz"
        partition_report = clean_partition(
            input_path=input_path,
            output_path=output_path,
            dataset_id=manifest["dataset_id"],
            revision=manifest["revision"],
            partition=entry["name"],
            raw_to_target=raw_to_target,
        )
        partition_report["input_relative_path"] = entry["relative_path"]
        partition_reports.append(partition_report)

    aggregate_targets: Counter[str] = Counter()
    aggregate_raw_labels: Counter[str] = Counter()
    aggregate_excluded: Counter[str] = Counter()
    for report in partition_reports:
        aggregate_targets.update(report["output_target_counts"])
        aggregate_raw_labels.update(report["output_raw_label_counts"])
        aggregate_excluded.update(report["excluded_raw_label_counts"])

    rows_seen = sum(report["rows_seen"] for report in partition_reports)
    rows_written = sum(report["rows_written"] for report in partition_reports)
    rows_excluded = sum(report["rows_excluded"] for report in partition_reports)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "cleaner_version": CLEANER_VERSION,
        "dataset_id": manifest["dataset_id"],
        "revision": manifest["revision"],
        "configuration": {
            "text_policy": "preserve exact decoded text",
            "excluded_source_prefixes": list(EXCLUDED_SOURCE_PREFIXES),
            "exclusion_reason": EXCLUSION_REASON,
            "raw_to_canonical_target": raw_to_target,
            "canonical_positive_class": "machine_generated",
            "content_id": "sha256(utf8(text))",
            "record_id": (
                "sha256(dataset_id\\0revision\\0partition\\0source\\0raw_label\\0content_id)"
            ),
            "output_format": "deterministic gzip JSON Lines with mtime=0",
        },
        "partitions": partition_reports,
        "aggregate": {
            "rows_seen": rows_seen,
            "rows_written": rows_written,
            "rows_excluded": rows_excluded,
            "excluded_raw_label_counts": _sorted_counts(aggregate_excluded),
            "output_raw_label_counts": _sorted_counts(aggregate_raw_labels),
            "output_target_counts": _sorted_counts(aggregate_targets),
        },
        "validation": {
            "status": "pass" if rows_seen == rows_written + rows_excluded else "fail",
            "record_id_collisions": 0,
        },
    }


def verify_cleaned_outputs(report: dict[str, Any], output_root: Path) -> None:
    """Verify processed output identities recorded by a cleaning report."""
    total_rows = 0
    for partition in report["partitions"]:
        path = output_root / partition["output_relative_path"]
        if not path.is_file():
            raise CleaningError(f"Missing processed output: {path}")
        if path.stat().st_size != partition["output_bytes"]:
            raise CleaningError(f"Processed output size mismatch: {path}")
        if sha256_file(path) != partition["output_sha256"]:
            raise CleaningError(f"Processed output SHA-256 mismatch: {path}")
        total_rows += partition["rows_written"]
    if total_rows != report["aggregate"]["rows_written"]:
        raise CleaningError("Processed output row reconciliation failed")


def render_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
