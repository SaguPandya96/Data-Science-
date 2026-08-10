"""Deterministic, text-free exploratory analysis of sanitized MAGE data."""

from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from authentitext.data.regimes import parse_source_id

EDA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
ANALYSIS_PARTITIONS = ("train", "validation")
ARTIFACT_DEFINITIONS = {
    "contains_markdown_fence": "text contains three consecutive backticks",
    "contains_newline": "text contains LF or CR",
    "contains_non_ascii": "text contains a code point above ASCII 127",
    "contains_repeated_space": "text contains two consecutive ASCII spaces",
    "contains_url_marker": "case-folded text contains http:// or https://",
    "leading_or_trailing_whitespace": "text differs from text.strip()",
    "long_over_512_tokens": "whitespace-token count is greater than 512",
    "short_under_50_tokens": "whitespace-token count is less than 50",
    "starts_with_heading_marker": "left-stripped text starts with '# '",
}


class EdaError(RuntimeError):
    """Raised when sanitized inputs cannot support a valid EDA report."""


def _length_summary(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {}
    ordered = sorted(values)

    def nearest_percentile(percentile: float) -> int:
        index = int((len(ordered) - 1) * percentile + 0.5)
        return ordered[index]

    middle = len(ordered) // 2
    if len(ordered) % 2:
        median: int | float = ordered[middle]
    else:
        median = (ordered[middle - 1] + ordered[middle]) / 2
    return {
        "min": ordered[0],
        "p05": nearest_percentile(0.05),
        "p25": nearest_percentile(0.25),
        "median": median,
        "p75": nearest_percentile(0.75),
        "p95": nearest_percentile(0.95),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 3),
    }


def _artifact_flags(text: str, token_count: int) -> dict[str, bool]:
    casefolded = text.casefold()
    return {
        "contains_markdown_fence": "```" in text,
        "contains_newline": "\n" in text or "\r" in text,
        "contains_non_ascii": any(ord(character) > 127 for character in text),
        "contains_repeated_space": "  " in text,
        "contains_url_marker": "http://" in casefolded or "https://" in casefolded,
        "leading_or_trailing_whitespace": text != text.strip(),
        "long_over_512_tokens": token_count > 512,
        "short_under_50_tokens": token_count < 50,
        "starts_with_heading_marker": text.lstrip().startswith("# "),
    }


def _rate(count: int, denominator: int) -> float:
    return round(count / denominator, 6) if denominator else 0.0


def analyze_split(split_report: dict[str, Any], input_root: Path) -> dict[str, Any]:
    """Aggregate composition, length, and predefined artifact indicators."""
    group_characters: defaultdict[str, list[int]] = defaultdict(list)
    group_tokens: defaultdict[str, list[int]] = defaultdict(list)
    group_rows: Counter[str] = Counter()
    partition_targets: Counter[tuple[str, str]] = Counter()
    domain_targets: Counter[tuple[str, str]] = Counter()
    generator_counts: Counter[str] = Counter()
    strategy_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    source_kinds: dict[str, str] = {}
    artifact_counts: Counter[tuple[str, str]] = Counter()
    rows = 0

    for output in split_report["outputs"]:
        partition = output["partition"]
        if partition not in ANALYSIS_PARTITIONS:
            continue
        path = input_root / output["relative_path"]
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise EdaError(f"{path}:{line_number} is not valid JSON") from error
                if not isinstance(record, dict) or record.get("partition") != partition:
                    raise EdaError(f"{path}:{line_number} has invalid partition metadata")
                text = record.get("text")
                source = record.get("source")
                target = record.get("target")
                if not isinstance(text, str) or not isinstance(source, str) or target not in (0, 1):
                    raise EdaError(f"{path}:{line_number} has invalid analysis fields")
                identity = parse_source_id(source)
                target_key = str(target)
                token_count = len(text.split())
                character_count = len(text)
                groups = (
                    "overall",
                    f"target:{target_key}",
                    f"partition:{partition}",
                    f"partition_target:{partition}:{target_key}",
                    f"domain_target:{identity.domain}:{target_key}",
                )
                for group in groups:
                    group_rows[group] += 1
                    group_characters[group].append(character_count)
                    group_tokens[group].append(token_count)
                partition_targets[(partition, target_key)] += 1
                domain_targets[(identity.domain, target_key)] += 1
                if identity.generator is not None:
                    generator_counts[identity.generator] += 1
                if identity.strategy is not None:
                    strategy_counts[identity.strategy] += 1
                source_counts[source] += 1
                source_kinds[source] = identity.kind
                for artifact, present in _artifact_flags(text, token_count).items():
                    if present:
                        artifact_counts[(artifact, "overall")] += 1
                        artifact_counts[(artifact, f"target:{target_key}")] += 1
                rows += 1

    expected_rows = sum(
        output["rows_written"]
        for output in split_report["outputs"]
        if output["partition"] in ANALYSIS_PARTITIONS
    )
    if rows != expected_rows:
        raise EdaError(f"Read {rows} rows; split report declares {expected_rows}")
    if any(not group_tokens[key] for key in group_rows):
        raise EdaError("Encountered an empty analysis group")

    length_groups = {
        key: {
            "rows": group_rows[key],
            "characters": _length_summary(group_characters[key]),
            "whitespace_tokens": _length_summary(group_tokens[key]),
        }
        for key in sorted(group_rows)
    }
    artifacts = {}
    for artifact in sorted(ARTIFACT_DEFINITIONS):
        overall_count = artifact_counts[(artifact, "overall")]
        by_target = {}
        for target in ("0", "1"):
            denominator = group_rows[f"target:{target}"]
            count = artifact_counts[(artifact, f"target:{target}")]
            by_target[target] = {
                "count": count,
                "rate": _rate(count, denominator),
                "rows": denominator,
            }
        artifacts[artifact] = {
            "definition": ARTIFACT_DEFINITIONS[artifact],
            "overall": {"count": overall_count, "rate": _rate(overall_count, rows)},
            "by_target": by_target,
            "absolute_target_rate_gap": round(
                abs(by_target["1"]["rate"] - by_target["0"]["rate"]), 6
            ),
        }

    source_count_ranges = {}
    for kind in ("human", "machine"):
        counts = [count for source, count in source_counts.items() if source_kinds[source] == kind]
        source_count_ranges[kind] = {
            "sources": len(counts),
            "minimum_rows": min(counts),
            "median_rows": _length_summary(counts)["median"],
            "maximum_rows": max(counts),
        }

    domains = sorted({domain for domain, _ in domain_targets})
    partitions = [
        output["partition"]
        for output in split_report["outputs"]
        if output["partition"] in ANALYSIS_PARTITIONS
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "eda_version": EDA_VERSION,
        "dataset_id": split_report["dataset_id"],
        "revision": split_report["revision"],
        "input_split": split_report["split_name"],
        "inputs": [
            {
                "partition": output["partition"],
                "relative_path": output["relative_path"],
                "rows": output["rows_written"],
                "bytes": output["bytes"],
                "sha256": output["sha256"],
                "used_for_eda": output["partition"] in ANALYSIS_PARTITIONS,
            }
            for output in split_report["outputs"]
        ],
        "configuration": {
            "token_measure": "Python str.split whitespace tokens; not a model tokenizer",
            "percentiles": "nearest percentile except median uses midpoint for even counts",
            "artifact_indicators": "predefined structural flags; counts do not imply causality",
            "privacy": "aggregate report contains no source text or record IDs",
            "analysis_partitions": list(ANALYSIS_PARTITIONS),
            "holdout_policy": "published test text and labels are excluded from EDA",
        },
        "population": {
            "analysis_rows": rows,
            "sanitized_split_rows": split_report["aggregate"]["output_rows"],
            "excluded_partition_counts": {
                output["partition"]: output["rows_written"]
                for output in split_report["outputs"]
                if output["partition"] not in ANALYSIS_PARTITIONS
            },
            "partition_target_counts": {
                partition: {target: partition_targets[(partition, target)] for target in ("0", "1")}
                for partition in partitions
            },
            "domain_target_counts": {
                domain: {target: domain_targets[(domain, target)] for target in ("0", "1")}
                for domain in domains
            },
            "generator_machine_counts": {
                generator: generator_counts[generator] for generator in sorted(generator_counts)
            },
            "strategy_machine_counts": {
                strategy: strategy_counts[strategy] for strategy in sorted(strategy_counts)
            },
            "source_count_ranges": source_count_ranges,
        },
        "length_groups": length_groups,
        "artifact_indicators": artifacts,
        "validation": {
            "status": "pass",
            "rows_reconcile_to_analysis_partitions": True,
            "published_test_excluded_from_eda": True,
            "unparsed_sources": 0,
            "raw_text_or_record_ids_emitted": False,
        },
    }


def render_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
