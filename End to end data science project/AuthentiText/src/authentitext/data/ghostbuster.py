"""Prepare and verify the pinned Ghostbuster main external corpus."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from authentitext.data.cleaning import open_deterministic_gzip, sha256_file
from authentitext.data.leakage import normalize_text

DATASET_ID = "vivek3141/ghostbuster-data"
PREPARATION_VERSION = 1
REPORT_SCHEMA_VERSION = 1
EXPECTED_DOCUMENTS_PER_CELL = 1000

DOMAINS = {
    "essay": "student_essay",
    "reuter": "news",
    "wp": "creative_writing",
}
CONDITIONS = {
    "human": {"generator": None, "strategy": None, "target": 0},
    "gpt": {"generator": "gpt-3.5-turbo", "strategy": "original", "target": 1},
    "claude": {"generator": "claude", "strategy": "original", "target": 1},
    "gpt_prompt1": {"generator": "gpt-3.5-turbo", "strategy": "prompt1", "target": 1},
    "gpt_prompt2": {"generator": "gpt-3.5-turbo", "strategy": "prompt2", "target": 1},
    "gpt_semantic": {"generator": "gpt-3.5-turbo", "strategy": "semantic", "target": 1},
    "gpt_writing": {"generator": "gpt-3.5-turbo", "strategy": "writing", "target": 1},
}


class GhostbusterError(RuntimeError):
    """Raised when the sealed external corpus cannot be prepared or verified."""


@dataclass(frozen=True)
class GhostbusterDocument:
    path: Path
    relative_path: str
    repository_domain: str
    domain: str
    condition: str
    generator: str | None
    strategy: str | None
    target: int


def _numeric_path_key(path: Path) -> tuple[str, int]:
    if path.suffix != ".txt" or not path.stem.isdigit():
        raise GhostbusterError(f"Main-corpus path is not a numeric text file: {path}")
    return path.parent.as_posix(), int(path.stem)


def discover_main_documents(
    repository_root: Path,
    *,
    documents_per_cell: int = EXPECTED_DOCUMENTS_PER_CELL,
) -> list[GhostbusterDocument]:
    """Select only the paper's three-domain, seven-condition main corpus."""
    resolved_root = repository_root.resolve()
    documents = []
    seen_paths: set[str] = set()
    for repository_domain, domain in DOMAINS.items():
        for condition, metadata in CONDITIONS.items():
            condition_root = repository_root / repository_domain / condition
            if not condition_root.is_dir():
                raise GhostbusterError(f"Missing main-corpus directory: {condition_root}")
            if repository_domain == "reuter":
                paths = [
                    path
                    for author_root in condition_root.iterdir()
                    if author_root.is_dir() and author_root.name != "logprobs"
                    for path in author_root.glob("*.txt")
                ]
            else:
                paths = list(condition_root.glob("*.txt"))
            paths.sort(key=_numeric_path_key)
            if len(paths) != documents_per_cell:
                raise GhostbusterError(
                    f"{repository_domain}/{condition} has {len(paths)} selected documents; "
                    f"expected {documents_per_cell}"
                )
            for path in paths:
                resolved = path.resolve()
                if not resolved.is_relative_to(resolved_root):
                    raise GhostbusterError(f"Selected path escapes repository root: {path}")
                relative_path = resolved.relative_to(resolved_root).as_posix()
                if relative_path in seen_paths:
                    raise GhostbusterError(f"Duplicate selected path: {relative_path}")
                seen_paths.add(relative_path)
                documents.append(
                    GhostbusterDocument(
                        path=path,
                        relative_path=relative_path,
                        repository_domain=repository_domain,
                        domain=domain,
                        condition=condition,
                        generator=metadata["generator"],
                        strategy=metadata["strategy"],
                        target=metadata["target"],
                    )
                )
    return documents


def _record_id(revision: str, relative_path: str, content_id: str) -> str:
    identity = "\0".join((DATASET_ID, revision, relative_path, content_id))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _source(document: GhostbusterDocument) -> str:
    if document.target == 0:
        return f"{document.domain}_human"
    return f"{document.domain}_machine_{document.strategy}_{document.generator}"


def _input_identity(documents: list[GhostbusterDocument]) -> dict[str, Any]:
    digest = hashlib.sha256()
    total_bytes = 0
    for document in documents:
        file_digest = sha256_file(document.path)
        size = document.path.stat().st_size
        total_bytes += size
        digest.update(document.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return {
        "files": len(documents),
        "bytes": total_bytes,
        "path_size_sha256_manifest": digest.hexdigest(),
    }


def _length_summary(lengths: list[int]) -> dict[str, Any]:
    ordered = sorted(lengths)
    return {
        "minimum": ordered[0],
        "median": round(float(median(ordered)), 3),
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "maximum": ordered[-1],
    }


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def prepare_ghostbuster_main(
    *,
    repository_root: Path,
    output_path: Path,
    revision: str,
    documents_per_cell: int = EXPECTED_DOCUMENTS_PER_CELL,
) -> dict[str, Any]:
    """Write deterministic external-test JSON Lines and a text-free report."""
    documents = discover_main_documents(
        repository_root,
        documents_per_cell=documents_per_cell,
    )
    input_identity = _input_identity(documents)
    license_path = repository_root / "LICENSE"
    if not license_path.is_file():
        raise GhostbusterError("Ghostbuster repository has no LICENSE file")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    targets: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    conditions: Counter[str] = Counter()
    generators: Counter[str] = Counter()
    strategies: Counter[str] = Counter()
    lengths = []
    record_ids: set[str] = set()
    normalized_ids: set[str] = set()
    exact_ids: set[str] = set()
    blank_documents = 0
    blank_relative_paths = []
    try:
        with open_deterministic_gzip(temporary) as output:
            for document in documents:
                try:
                    text = document.path.read_text(encoding="utf-8")
                except UnicodeDecodeError as error:
                    raise GhostbusterError(f"{document.path} is not valid UTF-8") from error
                if not text.strip():
                    blank_documents += 1
                    blank_relative_paths.append(document.relative_path)
                    continue
                content_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
                normalized_id = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
                record_id = _record_id(revision, document.relative_path, content_id)
                if record_id in record_ids:
                    raise GhostbusterError(f"Duplicate record ID for {document.relative_path}")
                record_ids.add(record_id)
                exact_ids.add(content_id)
                normalized_ids.add(normalized_id)
                output.write(
                    json.dumps(
                        {
                            "content_id": content_id,
                            "dataset_id": DATASET_ID,
                            "domain": document.domain,
                            "generator": document.generator,
                            "normalized_id": normalized_id,
                            "partition": "test",
                            "raw_label": "human" if document.target == 0 else "machine",
                            "record_id": record_id,
                            "revision": revision,
                            "source": _source(document),
                            "source_condition": document.condition,
                            "source_relative_path": document.relative_path,
                            "strategy": document.strategy,
                            "target": document.target,
                            "text": text,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                targets[str(document.target)] += 1
                domains[document.domain] += 1
                conditions[document.condition] += 1
                if document.generator is not None:
                    generators[document.generator] += 1
                if document.strategy is not None:
                    strategies[document.strategy] += 1
                lengths.append(len(text.split()))
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    selected_rows = len(DOMAINS) * len(CONDITIONS) * documents_per_cell
    output_rows = selected_rows - blank_documents
    if len(record_ids) != output_rows:
        output_path.unlink(missing_ok=True)
        raise GhostbusterError("Prepared row count does not match the declared main corpus")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "preparation_version": PREPARATION_VERSION,
        "dataset_id": DATASET_ID,
        "revision": revision,
        "source_repository": {
            "url": "https://github.com/vivek3141/ghostbuster-data",
            "license": "CC BY 3.0",
            "license_relative_path": "LICENSE",
            "license_sha256": sha256_file(license_path),
        },
        "selection": {
            "domains": DOMAINS,
            "conditions": CONDITIONS,
            "documents_per_domain_condition": documents_per_cell,
            "included_pattern": "main text .txt files directly within each condition/author",
            "excluded": [
                "nested logprobs and logprobs_babbage",
                "prompts",
                "perturb and perturb_old",
                "other human-only and auxiliary evaluation sets",
            ],
            "paper_main_corpus_rows": selected_rows,
        },
        "input": input_identity,
        "output": {
            "relative_path": output_path.name,
            "rows": output_rows,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
        "profile": {
            "target_counts": _sorted_counts(targets),
            "domain_counts": _sorted_counts(domains),
            "source_condition_counts": _sorted_counts(conditions),
            "generator_machine_counts": _sorted_counts(generators),
            "strategy_machine_counts": _sorted_counts(strategies),
            "whitespace_tokens": _length_summary(lengths),
            "unique_exact_content_ids": len(exact_ids),
            "unique_normalized_content_ids": len(normalized_ids),
        },
        "validation": {
            "status": "pass",
            "blank_documents": blank_documents,
            "blank_documents_excluded": blank_documents,
            "blank_relative_paths": blank_relative_paths,
            "duplicate_record_ids": 0,
            "selected_files_match_declared_cells": True,
            "output_identity_recorded": True,
            "raw_text_in_report": False,
            "test_scoring_performed": False,
        },
    }


def verify_ghostbuster_main(
    report: dict[str, Any],
    repository_root: Path,
    output_path: Path,
) -> None:
    """Recompute repository, output, and aggregate identities without scoring."""
    if report.get("validation", {}).get("status") != "pass":
        raise GhostbusterError("Ghostbuster preparation report status is not pass")
    documents_per_cell = report["selection"]["documents_per_domain_condition"]
    documents = discover_main_documents(
        repository_root,
        documents_per_cell=documents_per_cell,
    )
    if _input_identity(documents) != report.get("input"):
        raise GhostbusterError("Ghostbuster selected-input identity does not match")
    output_identity = report["output"]
    if not output_path.is_file():
        raise GhostbusterError(f"Missing prepared external corpus: {output_path}")
    if output_path.stat().st_size != output_identity["bytes"]:
        raise GhostbusterError("Ghostbuster prepared-output size does not match")
    if sha256_file(output_path) != output_identity["sha256"]:
        raise GhostbusterError("Ghostbuster prepared-output hash does not match")

    rows = 0
    targets: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    conditions: Counter[str] = Counter()
    record_ids: set[str] = set()
    with gzip.open(output_path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            required = {
                "domain",
                "record_id",
                "source_condition",
                "target",
                "text",
            }
            if not required.issubset(record):
                raise GhostbusterError(f"{output_path}:{line_number} is missing required fields")
            if record["record_id"] in record_ids or not str(record["text"]).strip():
                raise GhostbusterError(f"{output_path}:{line_number} is duplicate or blank")
            record_ids.add(record["record_id"])
            targets[str(record["target"])] += 1
            domains[str(record["domain"])] += 1
            conditions[str(record["source_condition"])] += 1
            rows += 1
    profile = report["profile"]
    if rows != output_identity["rows"] or rows != len(record_ids):
        raise GhostbusterError("Ghostbuster prepared-output rows do not reconcile")
    if _sorted_counts(targets) != profile["target_counts"]:
        raise GhostbusterError("Ghostbuster prepared target counts do not match")
    if _sorted_counts(domains) != profile["domain_counts"]:
        raise GhostbusterError("Ghostbuster prepared domain counts do not match")
    if _sorted_counts(conditions) != profile["source_condition_counts"]:
        raise GhostbusterError("Ghostbuster prepared condition counts do not match")


def render_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
