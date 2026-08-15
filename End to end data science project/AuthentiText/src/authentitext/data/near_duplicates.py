"""Bounded, deterministic high-overlap audit for processed text partitions."""

from __future__ import annotations

import gzip
import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from authentitext.data.leakage import REQUIRED_RECORD_FIELDS, normalize_text

AUDITOR_VERSION = 2
REPORT_SCHEMA_VERSION = 1
TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


class NearDuplicateError(RuntimeError):
    """Raised when a bounded near-duplicate audit cannot complete safely."""


@dataclass(frozen=True, slots=True)
class SampledRecord:
    record_id: str
    partition: str
    source: str
    target: int
    text: str
    normalized_id: str


@dataclass(frozen=True, slots=True)
class RecordFeatures:
    shingle_hashes: frozenset[int]
    block_keys: tuple[tuple[str, int], ...]


def _hash64(value: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(),
        byteorder="big",
    )


def _normalized_id(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def select_source_balanced_sample(
    cleaning_report: dict[str, Any],
    input_root: Path,
    *,
    per_source: int,
) -> list[SampledRecord]:
    """Select the lowest stable record IDs within each partition/source stratum."""
    if per_source <= 0:
        raise NearDuplicateError("per_source must be positive")

    heaps: defaultdict[tuple[str, str], list[tuple[int, str, SampledRecord]]] = defaultdict(list)
    for partition_report in cleaning_report["partitions"]:
        partition = partition_report["partition"]
        path = input_root / partition_report["output_relative_path"]
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                if set(record) != REQUIRED_RECORD_FIELDS:
                    raise NearDuplicateError(f"{path}:{line_number} has unexpected record fields")
                if record["partition"] != partition:
                    raise NearDuplicateError(f"{path}:{line_number} has an unexpected partition")
                sampled = SampledRecord(
                    record_id=record["record_id"],
                    partition=partition,
                    source=record["source"],
                    target=int(record["target"]),
                    text=record["text"],
                    normalized_id=_normalized_id(record["text"]),
                )
                score = int(sampled.record_id, 16)
                heap = heaps[(partition, sampled.source)]
                item = (-score, sampled.record_id, sampled)
                if len(heap) < per_source:
                    heapq.heappush(heap, item)
                elif score < -heap[0][0]:
                    heapq.heapreplace(heap, item)

    selected = [item[2] for heap in heaps.values() for item in heap]
    partition_order = {
        partition["partition"]: index
        for index, partition in enumerate(cleaning_report["partitions"])
    }
    return sorted(
        selected,
        key=lambda record: (
            partition_order[record.partition],
            record.source,
            record.record_id,
        ),
    )


def _record_features(
    record: SampledRecord,
    *,
    shingle_size: int,
    sketch_size: int,
    edge_tokens: int,
) -> RecordFeatures:
    tokens = TOKEN_PATTERN.findall(normalize_text(record.text))
    if not tokens:
        raise NearDuplicateError(f"Record {record.record_id} has no normalized tokens")
    if len(tokens) < shingle_size:
        shingles = {_hash64(token) for token in tokens}
    else:
        shingles = {
            _hash64("\x1f".join(tokens[index : index + shingle_size]))
            for index in range(len(tokens) - shingle_size + 1)
        }
    bottom = sorted(shingles)[:sketch_size]
    prefix = _hash64("\x1f".join(tokens[:edge_tokens]))
    suffix = _hash64("\x1f".join(tokens[-edge_tokens:]))
    block_keys = tuple(
        [("bottom_shingle", value) for value in bottom] + [("prefix", prefix), ("suffix", suffix)]
    )
    return RecordFeatures(frozenset(shingles), block_keys)


def _partition_pair(
    first: SampledRecord,
    second: SampledRecord,
    partition_order: dict[str, int],
) -> str:
    return "|".join(
        sorted(
            (first.partition, second.partition),
            key=partition_order.__getitem__,
        )
    )


def audit_near_duplicates(
    cleaning_report: dict[str, Any],
    input_root: Path,
    *,
    per_source: int = 20,
    shingle_size: int = 5,
    sketch_size: int = 8,
    edge_tokens: int = 8,
    max_bucket_size: int = 100,
    similarity_threshold: float = 0.8,
    max_candidate_pairs: int = 2_000_000,
) -> dict[str, Any]:
    """Audit sampled cross-partition pairs for high word-shingle overlap."""
    if shingle_size <= 0 or sketch_size <= 0 or edge_tokens <= 0:
        raise NearDuplicateError("feature sizes must be positive")
    if max_bucket_size < 2 or max_candidate_pairs <= 0:
        raise NearDuplicateError("candidate limits are invalid")
    if not 0 < similarity_threshold <= 1:
        raise NearDuplicateError("similarity_threshold must be in (0, 1]")

    sample = select_source_balanced_sample(
        cleaning_report,
        input_root,
        per_source=per_source,
    )
    if not sample:
        raise NearDuplicateError("The deterministic sample is empty")
    partition_order = {
        partition["partition"]: index
        for index, partition in enumerate(cleaning_report["partitions"])
    }
    features = [
        _record_features(
            record,
            shingle_size=shingle_size,
            sketch_size=sketch_size,
            edge_tokens=edge_tokens,
        )
        for record in sample
    ]

    blocks: defaultdict[tuple[str, int], list[int]] = defaultdict(list)
    for index, record_features in enumerate(features):
        for key in record_features.block_keys:
            blocks[key].append(index)

    candidate_pairs: set[tuple[int, int]] = set()
    skipped_blocks = 0
    skipped_block_memberships = 0
    for members in blocks.values():
        if len(members) > max_bucket_size:
            skipped_blocks += 1
            skipped_block_memberships += len(members)
            continue
        for first_index, second_index in combinations(members, 2):
            if sample[first_index].partition == sample[second_index].partition:
                continue
            pair = (
                min(first_index, second_index),
                max(first_index, second_index),
            )
            candidate_pairs.add(pair)
            if len(candidate_pairs) > max_candidate_pairs:
                raise NearDuplicateError(f"Candidate pair limit exceeded: {max_candidate_pairs}")

    normalized_equal_pairs = 0
    confirmed_pairs: list[tuple[int, int, float]] = []
    pair_partition_counts: Counter[str] = Counter()
    conflicting_target_pairs = 0
    for first_index, second_index in sorted(candidate_pairs):
        first = sample[first_index]
        second = sample[second_index]
        if first.normalized_id == second.normalized_id:
            normalized_equal_pairs += 1
            continue
        first_shingles = features[first_index].shingle_hashes
        second_shingles = features[second_index].shingle_hashes
        union_size = len(first_shingles | second_shingles)
        similarity = len(first_shingles & second_shingles) / union_size if union_size else 0.0
        if similarity >= similarity_threshold:
            confirmed_pairs.append((first_index, second_index, similarity))
            pair_partition_counts[_partition_pair(first, second, partition_order)] += 1
            conflicting_target_pairs += first.target != second.target

    parent = list(range(len(sample)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parent[second_root] = first_root

    for first_index, second_index, _ in confirmed_pairs:
        union(first_index, second_index)

    grouped_members: defaultdict[int, set[int]] = defaultdict(set)
    for first_index, second_index, _ in confirmed_pairs:
        root = find(first_index)
        grouped_members[root].update((first_index, second_index))
    groups = list(grouped_members.values())

    sample_partition_counts = Counter(record.partition for record in sample)
    sample_target_counts = Counter(str(record.target) for record in sample)
    sample_sources = {record.source for record in sample}
    sample_strata = {(record.partition, record.source) for record in sample}
    sample_fingerprint = hashlib.sha256(
        "\n".join(record.record_id for record in sample).encode("utf-8")
    ).hexdigest()

    inputs = [
        {
            "partition": partition["partition"],
            "relative_path": partition["output_relative_path"],
            "bytes": partition["output_bytes"],
            "sha256": partition["output_sha256"],
        }
        for partition in cleaning_report["partitions"]
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "auditor_version": AUDITOR_VERSION,
        "dataset_id": cleaning_report["dataset_id"],
        "revision": cleaning_report["revision"],
        "inputs": inputs,
        "configuration": {
            "selection": "lowest record IDs per partition/source stratum",
            "per_source": per_source,
            "normalization": "NFKC, case-fold, collapse whitespace, Unicode word tokens",
            "shingle_size": shingle_size,
            "sketch": f"bottom-{sketch_size} BLAKE2b-64 word-shingle hashes",
            "edge_token_blocks": edge_tokens,
            "max_bucket_size": max_bucket_size,
            "similarity": "Jaccard over hashed word shingles",
            "similarity_threshold": similarity_threshold,
            "max_candidate_pairs": max_candidate_pairs,
            "scope": "sampled cross-partition high-overlap audit",
        },
        "sample": {
            "rows": len(sample),
            "population_rows": cleaning_report["aggregate"]["rows_written"],
            "distinct_sources": len(sample_sources),
            "partition_source_strata": len(sample_strata),
            "partition_counts": {
                partition["partition"]: sample_partition_counts[partition["partition"]]
                for partition in cleaning_report["partitions"]
            },
            "target_counts": {
                key: sample_target_counts[key] for key in sorted(sample_target_counts)
            },
            "record_id_fingerprint": sample_fingerprint,
        },
        "candidate_generation": {
            "blocks": len(blocks),
            "skipped_oversize_blocks": skipped_blocks,
            "skipped_block_memberships": skipped_block_memberships,
            "candidate_pairs": len(candidate_pairs),
            "normalized_equal_candidate_pairs_excluded": normalized_equal_pairs,
        },
        "high_overlap_findings": {
            "confirmed_pairs": len(confirmed_pairs),
            "groups": len(groups),
            "rows_in_groups": len({index for group in groups for index in group}),
            "largest_group_rows": max((len(group) for group in groups), default=0),
            "pair_partition_counts": {
                key: pair_partition_counts[key] for key in sorted(pair_partition_counts)
            },
            "conflicting_target_pairs": conflicting_target_pairs,
            "minimum_similarity": (
                round(min(pair[2] for pair in confirmed_pairs), 6) if confirmed_pairs else None
            ),
            "maximum_similarity": (
                round(max(pair[2] for pair in confirmed_pairs), 6) if confirmed_pairs else None
            ),
            "confirmed_edges": [
                {
                    "first_record_id": sample[first_index].record_id,
                    "second_record_id": sample[second_index].record_id,
                    "partition_pair": _partition_pair(
                        sample[first_index], sample[second_index], partition_order
                    ),
                    "similarity": round(similarity, 6),
                }
                for first_index, second_index, similarity in confirmed_pairs
            ],
        },
        "validation": {
            "status": "pass",
            "limitations": [
                "This is a deterministic source-balanced sample, not a population census.",
                (
                    "The source-balanced sample is target-skewed and does not "
                    "estimate a population rate."
                ),
                "Candidate blocking can miss pairs that share no retained block key.",
                "Word-shingle Jaccard measures lexical overlap, not semantic paraphrase.",
                "Oversize blocks are skipped and counted to bound pair growth.",
            ],
        },
    }


def render_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
