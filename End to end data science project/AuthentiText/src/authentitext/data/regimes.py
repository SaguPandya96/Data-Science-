"""Define source-disjoint domain and generator evaluation regimes."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REGIME_VERSION = 1
REPORT_SCHEMA_VERSION = 1
STRATEGIES = (
    "continuation",
    "specified",
    "topical",
    "original",
    "prompt1",
    "prompt2",
    "semantic",
    "writing",
)
ROLES = ("train", "validation", "test", "excluded")


class RegimeError(RuntimeError):
    """Raised when source metadata cannot define valid evaluation regimes."""


@dataclass(frozen=True)
class SourceIdentity:
    source: str
    domain: str
    kind: str
    strategy: str | None
    generator: str | None


def parse_source_id(source: str) -> SourceIdentity:
    """Parse the documented MAGE source naming convention without guessing."""
    if source.endswith("_human"):
        domain = source.removesuffix("_human")
        if not domain:
            raise RegimeError(f"Source has an empty domain: {source!r}")
        return SourceIdentity(source, domain, "human", None, None)

    domain, separator, remainder = source.partition("_machine_")
    if not separator or not domain:
        raise RegimeError(f"Source does not match a supported MAGE pattern: {source!r}")
    for strategy in STRATEGIES:
        prefix = f"{strategy}_"
        if remainder.startswith(prefix) and remainder != prefix:
            return SourceIdentity(
                source,
                domain,
                "machine",
                strategy,
                remainder.removeprefix(prefix),
            )
    raise RegimeError(f"Source has an unsupported generation strategy: {source!r}")


def assign_domain_role(partition: str, identity: SourceIdentity, held_domain: str) -> str:
    """Assign one record to a leave-one-domain-out role."""
    if partition == "train" and identity.domain != held_domain:
        return "train"
    if partition == "validation" and identity.domain != held_domain:
        return "validation"
    if partition == "test" and identity.domain == held_domain:
        return "test"
    return "excluded"


def assign_generator_role(partition: str, identity: SourceIdentity, held_generator: str) -> str:
    """Assign one record to a leave-one-exact-generator-out role."""
    is_held_machine = identity.kind == "machine" and identity.generator == held_generator
    if partition == "train" and not is_held_machine:
        return "train"
    if partition == "validation" and not is_held_machine:
        return "validation"
    if partition == "test" and (identity.kind == "human" or is_held_machine):
        return "test"
    return "excluded"


def _new_summary() -> dict[str, Any]:
    return {role: {"rows": 0, "targets": Counter(), "sources": set()} for role in ROLES}


def _add_cell(
    summary: dict[str, Any],
    role: str,
    source: str,
    target: int,
    rows: int,
) -> None:
    summary[role]["rows"] += rows
    summary[role]["targets"][str(target)] += rows
    summary[role]["sources"].add(source)


def _render_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        role: {
            "rows": summary[role]["rows"],
            "target_counts": {target: summary[role]["targets"][target] for target in ("0", "1")},
            "distinct_sources": len(summary[role]["sources"]),
        }
        for role in ROLES
    }


def _validate_role_summary(summary: dict[str, Any], expected_rows: int, regime_id: str) -> None:
    if sum(summary[role]["rows"] for role in ROLES) != expected_rows:
        raise RegimeError(f"{regime_id} role counts do not reconcile")
    for role in ("train", "validation", "test"):
        if summary[role]["targets"]["0"] == 0 or summary[role]["targets"]["1"] == 0:
            raise RegimeError(f"{regime_id} {role} role does not contain both targets")


def define_source_regimes(split_report: dict[str, Any], input_root: Path) -> dict[str, Any]:
    """Profile selector-only holdout regimes over an existing sanitized split."""
    source_identities: dict[str, SourceIdentity] = {}
    cell_counts: Counter[tuple[str, str, int]] = Counter()
    partition_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    rows = 0

    for output in split_report["outputs"]:
        partition = output["partition"]
        path = input_root / output["relative_path"]
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RegimeError(f"{path}:{line_number} is not valid JSON") from error
                if not isinstance(record, dict):
                    raise RegimeError(f"{path}:{line_number} is not a JSON object")
                if record.get("partition") != partition:
                    raise RegimeError(f"{path}:{line_number} has the wrong partition")
                source = record.get("source")
                target = record.get("target")
                if not isinstance(source, str) or target not in (0, 1):
                    raise RegimeError(f"{path}:{line_number} has invalid source metadata")
                identity = source_identities.setdefault(source, parse_source_id(source))
                expected_target = 0 if identity.kind == "human" else 1
                if target != expected_target:
                    raise RegimeError(
                        f"{path}:{line_number} target conflicts with source kind for {source!r}"
                    )
                cell_counts[(partition, source, target)] += 1
                partition_counts[partition] += 1
                target_counts[str(target)] += 1
                rows += 1

    expected_rows = split_report["aggregate"]["output_rows"]
    if rows != expected_rows:
        raise RegimeError(f"Read {rows} rows; split report declares {expected_rows}")
    domains = sorted({identity.domain for identity in source_identities.values()})
    generators = sorted(
        {
            identity.generator
            for identity in source_identities.values()
            if identity.generator is not None
        }
    )
    strategies = sorted(
        {
            identity.strategy
            for identity in source_identities.values()
            if identity.strategy is not None
        }
    )

    domain_holdouts = []
    for held_domain in domains:
        summary = _new_summary()
        for (partition, source, target), count in cell_counts.items():
            identity = source_identities[source]
            role = assign_domain_role(partition, identity, held_domain)
            _add_cell(summary, role, source, target, count)
        regime_id = f"domain:{held_domain}"
        _validate_role_summary(summary, rows, regime_id)
        domain_holdouts.append(
            {
                "regime_id": regime_id,
                "held_out_domain": held_domain,
                "roles": _render_summary(summary),
            }
        )

    generator_holdouts = []
    for held_generator in generators:
        summary = _new_summary()
        for (partition, source, target), count in cell_counts.items():
            identity = source_identities[source]
            role = assign_generator_role(partition, identity, held_generator)
            _add_cell(summary, role, source, target, count)
        regime_id = f"generator:{held_generator}"
        _validate_role_summary(summary, rows, regime_id)
        generator_holdouts.append(
            {
                "regime_id": regime_id,
                "held_out_generator": held_generator,
                "roles": _render_summary(summary),
            }
        )

    source_kind_counts = Counter(identity.kind for identity in source_identities.values())
    source_domain_counts = Counter(identity.domain for identity in source_identities.values())
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "regime_version": REGIME_VERSION,
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
            }
            for output in split_report["outputs"]
        ],
        "configuration": {
            "materialization": "selectors and measured counts only; text is not duplicated",
            "domain_train": "published train rows whose domain is not held out",
            "domain_validation": "published validation rows whose domain is not held out",
            "domain_test": "published test rows whose domain is held out",
            "generator_train": ("published train rows except machine rows from the held generator"),
            "generator_validation": (
                "published validation rows except machine rows from the held generator"
            ),
            "generator_test": (
                "published test human rows plus machine rows from the held generator"
            ),
            "generator_identity": "exact upstream identifier across all prompt strategies",
            "record_policy": "records are selected in place and never moved",
        },
        "population": {
            "rows": rows,
            "partition_counts": {
                partition: partition_counts[partition]
                for partition in ("train", "validation", "test")
            },
            "target_counts": {target: target_counts[target] for target in ("0", "1")},
            "distinct_sources": len(source_identities),
            "source_kind_counts": {kind: source_kind_counts[kind] for kind in ("human", "machine")},
            "source_domain_counts": {domain: source_domain_counts[domain] for domain in domains},
            "domains": domains,
            "generators": generators,
            "strategies": strategies,
        },
        "domain_holdouts": domain_holdouts,
        "generator_holdouts": generator_holdouts,
        "validation": {
            "status": "pass",
            "unparsed_sources": 0,
            "source_target_conflicts": 0,
            "domain_regimes": len(domain_holdouts),
            "generator_regimes": len(generator_holdouts),
            "every_regime_reconciles_population": True,
            "every_active_role_contains_both_targets": True,
        },
    }


def render_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
