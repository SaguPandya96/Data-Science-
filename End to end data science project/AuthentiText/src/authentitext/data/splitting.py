"""Create leakage-sanitized in-distribution MAGE partitions."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from authentitext.data.cleaning import open_deterministic_gzip, sha256_file
from authentitext.data.leakage import normalize_text

SPLITTER_VERSION = 1
REPORT_SCHEMA_VERSION = 1
PARTITION_PRECEDENCE = ("train", "test", "validation")
REQUIRED_FIELDS = {"partition", "record_id", "target", "text"}


class SplitError(RuntimeError):
    """Raised when sanitized splits cannot be produced unambiguously."""


@dataclass(frozen=True)
class RecordMetadata:
    record_id: str
    partition: str
    target: int
    normalized_id: str


class DisjointSet:
    def __init__(self) -> None:
        self.parent: list[int] = []
        self.size: list[int] = []

    def add(self) -> int:
        index = len(self.parent)
        self.parent.append(index)
        self.size.append(1)
        return index

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if self.size[first_root] < self.size[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        self.size[first_root] += self.size[second_root]


def _normalized_id(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _iter_partition_records(path: Path, expected_partition: str):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SplitError(f"{path}:{line_number} is not valid JSON") from error
            if not isinstance(record, dict) or not REQUIRED_FIELDS <= record.keys():
                raise SplitError(f"{path}:{line_number} is missing required fields")
            if record["partition"] != expected_partition:
                raise SplitError(
                    f"{path}:{line_number} has partition {record['partition']!r}; "
                    f"expected {expected_partition!r}"
                )
            if record["target"] not in (0, 1):
                raise SplitError(f"{path}:{line_number} has an invalid target")
            if not isinstance(record["record_id"], str) or not isinstance(record["text"], str):
                raise SplitError(f"{path}:{line_number} has invalid field types")
            yield record


def _validate_reports(cleaning_report: dict[str, Any], near_report: dict[str, Any]) -> None:
    for key in ("dataset_id", "revision"):
        if near_report.get(key) != cleaning_report.get(key):
            raise SplitError(f"Near-duplicate report {key} does not match cleaning report")
    findings = near_report.get("high_overlap_findings", {})
    edges = findings.get("confirmed_edges")
    if not isinstance(edges, list):
        raise SplitError("Near-duplicate report does not contain confirmed edges")
    if findings.get("confirmed_pairs") != len(edges):
        raise SplitError("Near-duplicate confirmed edge count does not reconcile")


def create_id_split(
    cleaning_report: dict[str, Any],
    near_report: dict[str, Any],
    input_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Collapse audited overlap components using explicit partition precedence."""
    _validate_reports(cleaning_report, near_report)
    partitions = [item["partition"] for item in cleaning_report["partitions"]]
    if set(partitions) != set(PARTITION_PRECEDENCE):
        raise SplitError(
            f"Expected partitions {PARTITION_PRECEDENCE!r}, found {tuple(partitions)!r}"
        )
    partition_reports = {item["partition"]: item for item in cleaning_report["partitions"]}
    precedence = {partition: index for index, partition in enumerate(PARTITION_PRECEDENCE)}

    records: list[RecordMetadata] = []
    record_indexes: dict[str, int] = {}
    normalized_indexes: dict[str, int] = {}
    normalized_counts: Counter[str] = Counter()
    input_partition_counts: Counter[str] = Counter()
    input_target_counts: Counter[str] = Counter()
    disjoint_set = DisjointSet()

    for partition in partitions:
        path = input_root / partition_reports[partition]["output_relative_path"]
        for record in _iter_partition_records(path, partition):
            record_id = record["record_id"]
            if record_id in record_indexes:
                raise SplitError(f"Duplicate input record ID: {record_id}")
            normalized_id = _normalized_id(record["text"])
            index = disjoint_set.add()
            metadata = RecordMetadata(record_id, partition, record["target"], normalized_id)
            records.append(metadata)
            record_indexes[record_id] = index
            normalized_counts[normalized_id] += 1
            input_partition_counts[partition] += 1
            input_target_counts[str(record["target"])] += 1
            previous = normalized_indexes.setdefault(normalized_id, index)
            disjoint_set.union(previous, index)

    expected_rows = cleaning_report["aggregate"]["rows_written"]
    if len(records) != expected_rows:
        raise SplitError(f"Read {len(records)} records; cleaning report declares {expected_rows}")

    near_edges = near_report["high_overlap_findings"]["confirmed_edges"]
    near_edge_indexes: list[tuple[int, int]] = []
    for edge_number, edge in enumerate(near_edges, start=1):
        try:
            first = record_indexes[edge["first_record_id"]]
            second = record_indexes[edge["second_record_id"]]
        except (KeyError, TypeError) as error:
            raise SplitError(
                f"Near-duplicate edge {edge_number} references an unknown record"
            ) from error
        if first == second:
            raise SplitError(f"Near-duplicate edge {edge_number} is a self-edge")
        disjoint_set.union(first, second)
        near_edge_indexes.append((first, second))

    components: defaultdict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        components[disjoint_set.find(index)].append(index)

    keep_ids: set[str] = set()
    drop_reasons: dict[str, str] = {}
    conflicting_components = 0
    grouped_components = 0
    for members in components.values():
        if len(members) > 1:
            grouped_components += 1
        targets = {records[index].target for index in members}
        if len(targets) > 1:
            conflicting_components += 1
            for index in members:
                drop_reasons[records[index].record_id] = "conflicting_target_component"
            continue
        winner = min(
            members,
            key=lambda index: (
                precedence[records[index].partition],
                records[index].record_id,
            ),
        )
        keep_ids.add(records[winner].record_id)
        for index in members:
            if index != winner:
                drop_reasons[records[index].record_id] = "grouped_overlap"

    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = {partition: output_root / f"{partition}.jsonl.gz" for partition in partitions}
    temporary_paths = {
        partition: path.with_suffix(f"{path.suffix}.tmp")
        for partition, path in output_paths.items()
    }
    output_partition_counts: Counter[str] = Counter()
    output_target_counts: Counter[str] = Counter()
    dropped_partition_counts: Counter[str] = Counter()
    dropped_partition_reason_counts: Counter[tuple[str, str]] = Counter()
    dropped_target_counts: Counter[str] = Counter()
    dropped_reason_counts: Counter[str] = Counter()
    written_ids: set[str] = set()
    try:
        with ExitStack() as stack:
            handles = {
                partition: stack.enter_context(open_deterministic_gzip(temporary_paths[partition]))
                for partition in partitions
            }
            for partition in partitions:
                path = input_root / partition_reports[partition]["output_relative_path"]
                for record in _iter_partition_records(path, partition):
                    record_id = record["record_id"]
                    if record_id in keep_ids:
                        handles[partition].write(
                            json.dumps(
                                record,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        if record_id in written_ids:
                            raise SplitError(f"Output record written twice: {record_id}")
                        written_ids.add(record_id)
                        output_partition_counts[partition] += 1
                        output_target_counts[str(record["target"])] += 1
                    else:
                        reason = drop_reasons.get(record_id)
                        if reason is None:
                            raise SplitError(f"No split decision for input record: {record_id}")
                        dropped_partition_counts[partition] += 1
                        dropped_partition_reason_counts[(partition, reason)] += 1
                        dropped_target_counts[str(record["target"])] += 1
                        dropped_reason_counts[reason] += 1
        if written_ids != keep_ids:
            raise SplitError("Output record IDs do not match the split decisions")
        for partition in partitions:
            temporary_paths[partition].replace(output_paths[partition])
    except BaseException:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise

    output_reports = []
    for partition in partitions:
        path = output_paths[partition]
        output_reports.append(
            {
                "partition": partition,
                "relative_path": path.name,
                "rows_written": output_partition_counts[partition],
                "target_counts": {
                    key: sum(
                        1
                        for record_id in keep_ids
                        if records[record_indexes[record_id]].partition == partition
                        and str(records[record_indexes[record_id]].target) == key
                    )
                    for key in ("0", "1")
                },
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    output_rows = len(keep_ids)
    dropped_rows = len(drop_reasons)
    near_component_roots = {disjoint_set.find(first) for first, _ in near_edge_indexes} | {
        disjoint_set.find(second) for _, second in near_edge_indexes
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "splitter_version": SPLITTER_VERSION,
        "dataset_id": cleaning_report["dataset_id"],
        "revision": cleaning_report["revision"],
        "split_name": "id",
        "configuration": {
            "normalization": "NFKC, case-fold, collapse whitespace",
            "overlap_edges": "confirmed edges from bounded near-duplicate audit",
            "component_policy": "drop conflicting targets; otherwise retain one record",
            "partition_precedence": list(PARTITION_PRECEDENCE),
            "within_partition_tie_break": "lowest record_id",
            "record_policy": "records are retained in their published partition; none are moved",
            "output_format": "deterministic gzip JSON Lines with mtime=0",
        },
        "inputs": [
            {
                "partition": item["partition"],
                "relative_path": item["output_relative_path"],
                "rows": item["rows_written"],
                "bytes": item["output_bytes"],
                "sha256": item["output_sha256"],
            }
            for item in cleaning_report["partitions"]
        ],
        "outputs": output_reports,
        "components": {
            "total": len(components),
            "grouped": grouped_components,
            "conflicting_target": conflicting_components,
            "with_confirmed_near_duplicate_edge": len(near_component_roots),
            "normalized_duplicate_values": sum(count > 1 for count in normalized_counts.values()),
            "confirmed_near_duplicate_edges": len(near_edge_indexes),
        },
        "aggregate": {
            "input_rows": len(records),
            "input_partition_counts": _sorted_counts(input_partition_counts),
            "input_target_counts": _sorted_counts(input_target_counts),
            "output_rows": output_rows,
            "output_partition_counts": _sorted_counts(output_partition_counts),
            "output_target_counts": _sorted_counts(output_target_counts),
            "dropped_rows": dropped_rows,
            "dropped_partition_counts": _sorted_counts(dropped_partition_counts),
            "dropped_partition_reason_counts": {
                partition: {
                    reason: dropped_partition_reason_counts[(partition, reason)]
                    for reason in ("conflicting_target_component", "grouped_overlap")
                }
                for partition in partitions
            },
            "dropped_target_counts": _sorted_counts(dropped_target_counts),
            "dropped_reason_counts": _sorted_counts(dropped_reason_counts),
        },
        "validation": {
            "status": "pass",
            "input_equals_output_plus_dropped": len(records) == output_rows + dropped_rows,
            "all_keep_decisions_written_once": len(written_ids) == output_rows,
            "unknown_near_duplicate_edge_records": 0,
        },
    }


def verify_id_split(report: dict[str, Any], output_root: Path) -> None:
    """Verify split file identities, counts, partitions, and unique record IDs."""
    total_rows = 0
    record_ids: set[str] = set()
    for output in report["outputs"]:
        path = output_root / output["relative_path"]
        if not path.is_file():
            raise SplitError(f"Missing split output: {path}")
        if path.stat().st_size != output["bytes"]:
            raise SplitError(f"Split output size mismatch: {path}")
        if sha256_file(path) != output["sha256"]:
            raise SplitError(f"Split output SHA-256 mismatch: {path}")
        rows = 0
        targets: Counter[str] = Counter()
        for record in _iter_partition_records(path, output["partition"]):
            if record["record_id"] in record_ids:
                raise SplitError(f"Duplicate record ID across split outputs: {record['record_id']}")
            record_ids.add(record["record_id"])
            targets[str(record["target"])] += 1
            rows += 1
        if rows != output["rows_written"]:
            raise SplitError(f"Split output row count mismatch: {path}")
        if {key: targets[key] for key in ("0", "1")} != output["target_counts"]:
            raise SplitError(f"Split output target counts mismatch: {path}")
        total_rows += rows
    if total_rows != report["aggregate"]["output_rows"]:
        raise SplitError("Split aggregate row count does not reconcile")
    if report["aggregate"]["input_rows"] != (
        report["aggregate"]["output_rows"] + report["aggregate"]["dropped_rows"]
    ):
        raise SplitError("Split report input/output/drop counts do not reconcile")


def render_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
