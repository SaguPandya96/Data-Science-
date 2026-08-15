"""Population equality and bounded lexical-overlap audit across two datasets."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from authentitext.data.leakage import normalize_text

AUDITOR_VERSION = 1
REPORT_SCHEMA_VERSION = 1
TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
MAGE_REQUIRED_FIELDS = {"content_id", "partition", "record_id", "target", "text"}
EXTERNAL_REQUIRED_FIELDS = {
    "content_id",
    "normalized_id",
    "partition",
    "record_id",
    "target",
    "text",
}


class ExternalOverlapError(RuntimeError):
    """Raised when the external overlap gate cannot complete safely."""


@dataclass(frozen=True, slots=True)
class ExternalRecord:
    record_id: str
    content_id: str
    normalized_id: str
    target: int


@dataclass(frozen=True, slots=True)
class Features:
    shingle_hashes: frozenset[int]
    block_keys: tuple[tuple[str, int], ...]


def _hash64(value: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(),
        byteorder="big",
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalized_id(text: str) -> str:
    return _sha256_text(normalize_text(text))


def _features(
    text: str,
    *,
    shingle_size: int,
    sketch_size: int,
    edge_tokens: int,
) -> Features:
    tokens = TOKEN_PATTERN.findall(normalize_text(text))
    if not tokens:
        return Features(shingle_hashes=frozenset(), block_keys=())
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
    return Features(
        shingle_hashes=frozenset(shingles),
        block_keys=tuple(
            [("bottom_shingle", value) for value in bottom]
            + [("prefix", prefix), ("suffix", suffix)]
        ),
    )


def _iter_jsonl(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ExternalOverlapError(f"{path}:{line_number} is invalid JSON") from error
            if not isinstance(record, dict):
                raise ExternalOverlapError(f"{path}:{line_number} is not an object")
            yield line_number, record


def _validate_text_record(
    record: dict[str, Any],
    *,
    required: set[str],
    path: Path,
    line_number: int,
) -> None:
    if not required <= record.keys():
        raise ExternalOverlapError(f"{path}:{line_number} is missing required fields")
    if not isinstance(record["text"], str) or not record["text"].strip():
        raise ExternalOverlapError(f"{path}:{line_number} has invalid text")
    if not isinstance(record["record_id"], str) or not record["record_id"]:
        raise ExternalOverlapError(f"{path}:{line_number} has an invalid record ID")
    if record["target"] not in (0, 1):
        raise ExternalOverlapError(f"{path}:{line_number} has an invalid target")
    if record["content_id"] != _sha256_text(record["text"]):
        raise ExternalOverlapError(f"{path}:{line_number} has an invalid content ID")


def _group_summary(groups: dict[str, list[int]], records: list[ExternalRecord]) -> dict[str, Any]:
    duplicates = [members for members in groups.values() if len(members) > 1]
    return {
        "duplicate_groups": len(duplicates),
        "rows_in_duplicate_groups": sum(len(members) for members in duplicates),
        "duplicate_extra_rows": sum(len(members) - 1 for members in duplicates),
        "conflicting_target_groups": sum(
            len({records[index].target for index in members}) > 1 for members in duplicates
        ),
    }


def _edge_summary(edges: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "pairs": len(edges),
        "distinct_mage_records": len({edge["mage_record_id"] for edge in edges}),
        "distinct_external_records": len({edge["external_record_id"] for edge in edges}),
        "conflicting_target_pairs": sum(
            edge["mage_target"] != edge["external_target"] for edge in edges
        ),
    }


def audit_external_overlap(
    split_report: dict[str, Any],
    mage_root: Path,
    external_manifest: dict[str, Any],
    external_path: Path,
    *,
    shingle_size: int = 5,
    sketch_size: int = 8,
    edge_tokens: int = 8,
    max_bucket_size: int = 100,
    similarity_threshold: float = 0.8,
    max_candidate_pairs: int = 2_000_000,
) -> dict[str, Any]:
    """Audit all equality keys and bounded lexical candidates before scoring."""
    if shingle_size <= 0 or sketch_size <= 0 or edge_tokens <= 0:
        raise ExternalOverlapError("Feature sizes must be positive")
    if max_bucket_size < 2 or max_candidate_pairs <= 0:
        raise ExternalOverlapError("Candidate bounds are invalid")
    if not 0 < similarity_threshold <= 1:
        raise ExternalOverlapError("Similarity threshold must be in (0, 1]")

    external_records: list[ExternalRecord] = []
    external_exact: defaultdict[str, list[int]] = defaultdict(list)
    external_normalized: defaultdict[str, list[int]] = defaultdict(list)
    blocks: defaultdict[tuple[str, int], list[int]] = defaultdict(list)
    external_record_ids: set[str] = set()
    external_rows = 0
    tokenless_external_rows = 0
    for line_number, record in _iter_jsonl(external_path):
        _validate_text_record(
            record,
            required=EXTERNAL_REQUIRED_FIELDS,
            path=external_path,
            line_number=line_number,
        )
        if record["partition"] != "test":
            raise ExternalOverlapError(f"{external_path}:{line_number} is not test-only")
        normalized_id = _normalized_id(record["text"])
        if normalized_id != record["normalized_id"]:
            raise ExternalOverlapError(
                f"{external_path}:{line_number} has an invalid normalized ID"
            )
        if record["record_id"] in external_record_ids:
            raise ExternalOverlapError(f"Duplicate external record ID: {record['record_id']}")
        external_record_ids.add(record["record_id"])
        index = len(external_records)
        external_records.append(
            ExternalRecord(
                record_id=record["record_id"],
                content_id=record["content_id"],
                normalized_id=normalized_id,
                target=int(record["target"]),
            )
        )
        external_exact[record["content_id"]].append(index)
        external_normalized[normalized_id].append(index)
        features = _features(
            record["text"],
            shingle_size=shingle_size,
            sketch_size=sketch_size,
            edge_tokens=edge_tokens,
        )
        tokenless_external_rows += not features.shingle_hashes
        for key in features.block_keys:
            blocks[key].append(index)
        external_rows += 1
    if external_rows != external_manifest["output"]["rows"]:
        raise ExternalOverlapError("External row count does not match its manifest")

    skipped_blocks = {key for key, members in blocks.items() if len(members) > max_bucket_size}
    skipped_memberships = sum(len(blocks[key]) for key in skipped_blocks)
    for key in skipped_blocks:
        del blocks[key]

    exclusions: defaultdict[str, set[str]] = defaultdict(set)
    for members in external_normalized.values():
        if len(members) < 2:
            continue
        targets = {external_records[index].target for index in members}
        if len(targets) > 1:
            for index in members:
                exclusions[external_records[index].record_id].add(
                    "internal_normalized_conflicting_target"
                )
        else:
            winner = min(external_records[index].record_id for index in members)
            for index in members:
                record_id = external_records[index].record_id
                if record_id != winner:
                    exclusions[record_id].add("internal_normalized_duplicate")

    exact_edges: list[dict[str, Any]] = []
    normalized_edges: list[dict[str, Any]] = []
    candidate_pairs: defaultdict[str, set[int]] = defaultdict(set)
    candidate_mage_features: dict[str, Features] = {}
    candidate_count = 0
    mage_rows = 0
    mage_record_ids: set[str] = set()
    mage_partition_counts: Counter[str] = Counter()
    tokenless_mage_rows = 0
    for output in split_report["outputs"]:
        partition = output["partition"]
        path = mage_root / output["relative_path"]
        partition_rows = 0
        for line_number, record in _iter_jsonl(path):
            _validate_text_record(
                record,
                required=MAGE_REQUIRED_FIELDS,
                path=path,
                line_number=line_number,
            )
            if record["partition"] != partition:
                raise ExternalOverlapError(f"{path}:{line_number} has the wrong partition")
            mage_record_id = record["record_id"]
            if mage_record_id in mage_record_ids:
                raise ExternalOverlapError(f"Duplicate MAGE record ID: {mage_record_id}")
            mage_record_ids.add(mage_record_id)
            normalized_id = _normalized_id(record["text"])
            exact_external_indexes = external_exact.get(record["content_id"], [])
            normalized_external_indexes = external_normalized.get(normalized_id, [])
            for external_index in exact_external_indexes:
                external = external_records[external_index]
                exact_edges.append(
                    {
                        "external_record_id": external.record_id,
                        "external_target": external.target,
                        "mage_partition": partition,
                        "mage_record_id": mage_record_id,
                        "mage_target": int(record["target"]),
                    }
                )
                exclusions[external.record_id].add("cross_dataset_exact")
            for external_index in normalized_external_indexes:
                external = external_records[external_index]
                normalized_edges.append(
                    {
                        "external_record_id": external.record_id,
                        "external_target": external.target,
                        "mage_partition": partition,
                        "mage_record_id": mage_record_id,
                        "mage_target": int(record["target"]),
                    }
                )
                exclusions[external.record_id].add("cross_dataset_normalized")

            features = _features(
                record["text"],
                shingle_size=shingle_size,
                sketch_size=sketch_size,
                edge_tokens=edge_tokens,
            )
            tokenless_mage_rows += not features.shingle_hashes
            matched_indexes: set[int] = set()
            for key in features.block_keys:
                matched_indexes.update(blocks.get(key, ()))
            if matched_indexes:
                previous_count = len(candidate_pairs[mage_record_id])
                candidate_pairs[mage_record_id].update(matched_indexes)
                candidate_count += len(candidate_pairs[mage_record_id]) - previous_count
                candidate_mage_features[mage_record_id] = features
                if candidate_count > max_candidate_pairs:
                    raise ExternalOverlapError(
                        f"Candidate pair limit exceeded: {max_candidate_pairs}"
                    )
            partition_rows += 1
            mage_rows += 1
            mage_partition_counts[partition] += 1
        if partition_rows != output["rows_written"]:
            raise ExternalOverlapError(f"MAGE {partition} rows do not match split report")
    if mage_rows != split_report["aggregate"]["output_rows"]:
        raise ExternalOverlapError("MAGE aggregate rows do not match split report")

    required_external_indexes = {index for indexes in candidate_pairs.values() for index in indexes}
    external_candidate_features: dict[int, Features] = {}
    if required_external_indexes:
        for line_number, record in _iter_jsonl(external_path):
            index = line_number - 1
            if index in required_external_indexes:
                external_candidate_features[index] = _features(
                    record["text"],
                    shingle_size=shingle_size,
                    sketch_size=sketch_size,
                    edge_tokens=edge_tokens,
                )
    if external_candidate_features.keys() != required_external_indexes:
        raise ExternalOverlapError("Could not reload every external candidate record")

    near_edges: list[dict[str, Any]] = []
    normalized_pair_keys = {
        (edge["mage_record_id"], edge["external_record_id"]) for edge in normalized_edges
    }
    for mage_record_id in sorted(candidate_pairs):
        mage_features = candidate_mage_features[mage_record_id]
        for external_index in sorted(candidate_pairs[mage_record_id]):
            external = external_records[external_index]
            if (mage_record_id, external.record_id) in normalized_pair_keys:
                continue
            external_features = external_candidate_features[external_index]
            union_size = len(mage_features.shingle_hashes | external_features.shingle_hashes)
            similarity = (
                len(mage_features.shingle_hashes & external_features.shingle_hashes) / union_size
                if union_size
                else 0.0
            )
            if similarity >= similarity_threshold:
                near_edges.append(
                    {
                        "external_record_id": external.record_id,
                        "mage_record_id": mage_record_id,
                        "similarity": round(similarity, 6),
                    }
                )
                exclusions[external.record_id].add("cross_dataset_high_lexical_overlap")

    exact_edges.sort(key=lambda edge: (edge["mage_record_id"], edge["external_record_id"]))
    normalized_edges.sort(key=lambda edge: (edge["mage_record_id"], edge["external_record_id"]))
    near_edges.sort(key=lambda edge: (edge["mage_record_id"], edge["external_record_id"]))
    exclusion_rows = [
        {"external_record_id": record_id, "reasons": sorted(reasons)}
        for record_id, reasons in sorted(exclusions.items())
    ]
    excluded_targets = Counter(
        str(record.target) for record in external_records if record.record_id in exclusions
    )
    exact_summary = _edge_summary(exact_edges)
    normalized_summary = _edge_summary(normalized_edges)
    exact_keys = {(edge["mage_record_id"], edge["external_record_id"]) for edge in exact_edges}
    normalized_summary["normalization_only_pairs"] = sum(
        (edge["mage_record_id"], edge["external_record_id"]) not in exact_keys
        for edge in normalized_edges
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "auditor_version": AUDITOR_VERSION,
        "datasets": {
            "development": {
                "dataset_id": split_report["dataset_id"],
                "revision": split_report["revision"],
                "rows": mage_rows,
                "partition_counts": {
                    output["partition"]: mage_partition_counts[output["partition"]]
                    for output in split_report["outputs"]
                },
                "inputs": [
                    {
                        "bytes": output["bytes"],
                        "partition": output["partition"],
                        "relative_path": output["relative_path"],
                        "rows": output["rows_written"],
                        "sha256": output["sha256"],
                    }
                    for output in split_report["outputs"]
                ],
            },
            "external": {
                "dataset_id": external_manifest["dataset_id"],
                "revision": external_manifest["revision"],
                "rows": external_rows,
                "relative_path": external_manifest["output"]["relative_path"],
                "bytes": external_manifest["output"]["bytes"],
                "sha256": external_manifest["output"]["sha256"],
            },
        },
        "configuration": {
            "exact_key": "sha256(utf8(text))",
            "normalized_key": "sha256(utf8(collapse_whitespace(casefold(NFKC(text)))))",
            "near_scope": "population blocking across every development and external row",
            "tokenization": "Unicode word tokens after equality normalization",
            "shingle_size": shingle_size,
            "sketch": f"bottom-{sketch_size} BLAKE2b-64 word-shingle hashes",
            "edge_token_blocks": edge_tokens,
            "max_external_block_size": max_bucket_size,
            "similarity": "Jaccard over hashed word shingles",
            "similarity_threshold": similarity_threshold,
            "max_candidate_pairs": max_candidate_pairs,
            "external_scoring_policy": (
                "exclude every listed external record before frozen scoring"
            ),
        },
        "external_internal_overlap": {
            "exact": _group_summary(external_exact, external_records),
            "normalized": _group_summary(external_normalized, external_records),
        },
        "cross_dataset_exact": {**exact_summary, "edges": exact_edges},
        "cross_dataset_normalized": {**normalized_summary, "edges": normalized_edges},
        "cross_dataset_near": {
            "blocks_retained": len(blocks),
            "skipped_oversize_blocks": len(skipped_blocks),
            "skipped_block_memberships": skipped_memberships,
            "candidate_pairs": candidate_count,
            "candidate_development_records": len(candidate_pairs),
            "candidate_external_records": len(required_external_indexes),
            "tokenless_development_records_not_blocked": tokenless_mage_rows,
            "tokenless_external_records_not_blocked": tokenless_external_rows,
            "normalized_equal_candidate_pairs_excluded": sum(
                pair in normalized_pair_keys
                for mage_record_id, indexes in candidate_pairs.items()
                for pair in (
                    (mage_record_id, external_records[index].record_id) for index in indexes
                )
            ),
            "confirmed_pairs": len(near_edges),
            "distinct_development_records": len({edge["mage_record_id"] for edge in near_edges}),
            "distinct_external_records": len({edge["external_record_id"] for edge in near_edges}),
            "minimum_similarity": (
                min(edge["similarity"] for edge in near_edges) if near_edges else None
            ),
            "maximum_similarity": (
                max(edge["similarity"] for edge in near_edges) if near_edges else None
            ),
            "edges": near_edges,
        },
        "exclusions": {
            "external_records": len(exclusion_rows),
            "external_target_counts": {
                key: excluded_targets[key] for key in sorted(excluded_targets)
            },
            "external_rows_after_exclusion": external_rows - len(exclusion_rows),
            "records": exclusion_rows,
        },
        "validation": {
            "status": "pass",
            "raw_text_in_report": False,
            "model_scoring_performed": False,
            "outcome_data_used_to_define_exclusions": False,
            "all_confirmed_overlap_resolved_by_external_exclusion": True,
            "limitations": [
                (
                    "Exact and normalized comparisons are population censuses; the near check "
                    "uses bounded blocking and can miss pairs that share no retained block key."
                ),
                "Oversize external blocks are skipped and counted to bound candidate growth.",
                (
                    "Nonblank records without Unicode word tokens participate in exact and "
                    "normalized checks but not word-shingle blocking."
                ),
                "Word-shingle Jaccard measures lexical overlap, not semantic paraphrase.",
            ],
        },
    }


def render_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
