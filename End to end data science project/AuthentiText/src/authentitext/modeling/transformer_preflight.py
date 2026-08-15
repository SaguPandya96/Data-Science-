"""Assess whether the pinned transformer experiment can run without touching test data."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORT_SCHEMA_VERSION = 1
MODEL_ID = "google/bert_uncased_L-2_H-128_A-2"
MODEL_REVISION = "30b0a37ccaaa32f332884b96992754e246e48c5f"
MODEL_LICENSE = "Apache-2.0"
MODEL_CACHE_DIRECTORY = "models--google--bert_uncased_L-2_H-128_A-2"
REQUIRED_DEPENDENCIES = ("torch", "transformers", "tokenizers", "accelerate")
MINIMUM_LOGICAL_CPUS = 4
MINIMUM_MEMORY_BYTES = 8 * 1024**3
MINIMUM_FREE_DISK_BYTES = 3 * 1024**3
SUPPORTED_WINDOWS_PYTHON = ((3, 9), (3, 12))


class TransformerPreflightError(RuntimeError):
    """Raised when preflight inputs or evidence are invalid."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _total_memory_bytes() -> int | None:
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical)
        return None
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in REQUIRED_DEPENDENCIES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _model_revision_cached() -> bool:
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return (cache_root / MODEL_CACHE_DIRECTORY / "snapshots" / MODEL_REVISION).is_dir()


def _load_training_identity(repo_root: Path) -> dict[str, Any]:
    report_path = repo_root / "data" / "metadata" / "mage_id_split_report.json"
    try:
        split_report = json.loads(report_path.read_text(encoding="utf-8"))
        train = next(item for item in split_report["outputs"] if item.get("partition") == "train")
        relative_path = train["relative_path"]
        expected_sha256 = train["sha256"]
        expected_bytes = train["bytes"]
        expected_rows = train.get("rows_written", train.get("rows"))
    except (OSError, json.JSONDecodeError, KeyError, StopIteration, TypeError) as error:
        raise TransformerPreflightError("Invalid MAGE split report") from error

    if not isinstance(expected_rows, int) or expected_rows <= 0:
        raise TransformerPreflightError("Sanitized training row count is invalid")

    train_path = repo_root / "data" / "processed" / "mage_splits" / "id" / relative_path
    if not train_path.is_file():
        raise TransformerPreflightError(f"Missing sanitized training partition: {train_path}")
    if train_path.stat().st_size != expected_bytes:
        raise TransformerPreflightError("Sanitized training partition size does not match report")
    if sha256_file(train_path) != expected_sha256:
        raise TransformerPreflightError(
            "Sanitized training partition SHA-256 does not match report"
        )
    return {
        "dataset_id": split_report["dataset_id"],
        "revision": split_report["revision"],
        "partition": "train",
        "relative_path": str(train_path.relative_to(repo_root)).replace("\\", "/"),
        "rows": expected_rows,
        "bytes": expected_bytes,
        "sha256": expected_sha256,
        "test_data_read": False,
    }


def build_preflight_report(
    repo_root: Path,
    *,
    python_version: tuple[int, int, int] | None = None,
    logical_cpus: int | None = None,
    total_memory_bytes: int | None = None,
    free_disk_bytes: int | None = None,
    dependency_versions: dict[str, str | None] | None = None,
    model_revision_cached: bool | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a text-free, evidence-backed readiness report for the experiment."""
    observed_python = python_version or sys.version_info[:3]
    observed_cpus = logical_cpus if logical_cpus is not None else (os.cpu_count() or 0)
    observed_memory = (
        total_memory_bytes if total_memory_bytes is not None else _total_memory_bytes()
    )
    observed_disk = (
        free_disk_bytes if free_disk_bytes is not None else shutil.disk_usage(repo_root).free
    )
    observed_dependencies = dependency_versions or _dependency_versions()
    observed_cache = (
        model_revision_cached if model_revision_cached is not None else _model_revision_cached()
    )
    training_input = _load_training_identity(repo_root)

    python_pair = observed_python[:2]
    python_supported = SUPPORTED_WINDOWS_PYTHON[0] <= python_pair <= SUPPORTED_WINDOWS_PYTHON[1]
    dependencies_ready = all(observed_dependencies.get(name) for name in REQUIRED_DEPENDENCIES)
    memory_ready = observed_memory is not None and observed_memory >= MINIMUM_MEMORY_BYTES
    gates = {
        "python_supported": python_supported,
        "dependencies_installed": dependencies_ready,
        "model_revision_cached": observed_cache,
        "training_input_verified": True,
        "logical_cpu_minimum_met": observed_cpus >= MINIMUM_LOGICAL_CPUS,
        "memory_minimum_met": memory_ready,
        "free_disk_minimum_met": observed_disk >= MINIMUM_FREE_DISK_BYTES,
    }
    blockers = [name for name, passed in gates.items() if not passed]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": generated_at or datetime.now(UTC).isoformat(),
        "status": "ready" if not blockers else "not_ready",
        "candidate": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
            "architecture": {
                "family": "BERT-Tiny",
                "layers": 2,
                "hidden_size": 128,
                "attention_heads": 2,
                "sequence_classifier_parameters": 4_386_178,
            },
        },
        "training_protocol": {
            "random_seed": 1729,
            "maximum_sequence_length": 128,
            "training_partition": "sanitized MAGE train",
            "training_rows": training_input["rows"],
            "full_training_partition_required": True,
            "test_data_allowed_before_freeze": False,
            "throughput_probe_is_model_evidence": False,
            "replacement_requires_full_model_selection_gate": True,
        },
        "training_input": training_input,
        "environment": {
            "operating_system": platform.system(),
            "python": ".".join(str(item) for item in observed_python),
            "logical_cpus": observed_cpus,
            "total_memory_bytes": observed_memory,
            "free_disk_bytes": observed_disk,
            "dependencies": observed_dependencies,
            "model_revision_cached": observed_cache,
        },
        "minimums": {
            "logical_cpus": MINIMUM_LOGICAL_CPUS,
            "memory_bytes": MINIMUM_MEMORY_BYTES,
            "free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
            "windows_python_minimum": "3.9",
            "windows_python_maximum": "3.12",
        },
        "gates": gates,
        "blockers": blockers,
        "validation": {
            "source_text_in_report": False,
            "training_metrics_reported": False,
            "transformer_trained": False,
            "test_data_read": False,
            "status": "pass",
        },
    }


def verify_preflight_report(report: dict[str, Any]) -> None:
    """Reject reports that overstate progress or weaken the frozen-data rules."""
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise TransformerPreflightError("Unexpected preflight report schema")
    if report.get("candidate", {}).get("revision") != MODEL_REVISION:
        raise TransformerPreflightError("Transformer revision is not pinned")
    protocol = report.get("training_protocol", {})
    if protocol.get("full_training_partition_required") is not True:
        raise TransformerPreflightError("Full sanitized training data must remain required")
    if protocol.get("test_data_allowed_before_freeze") is not False:
        raise TransformerPreflightError("Test data must remain sealed")
    validation = report.get("validation", {})
    for key in ("source_text_in_report", "training_metrics_reported", "transformer_trained"):
        if validation.get(key) is not False:
            raise TransformerPreflightError(f"Preflight validation.{key} must be false")
    if report.get("status") == "ready" and report.get("blockers"):
        raise TransformerPreflightError("Ready preflight report cannot contain blockers")


def render_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
