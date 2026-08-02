"""Run repository-level reproducibility, safety, and artifact consistency checks."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import nbformat

from supplylens.config import load_config, resolve_path
from supplylens.validation import file_sha256

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".csv",
    ".html",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    "pyproject.toml",
    "Makefile",
    "configs/config.yaml",
    "data/README.md",
    "notebooks/SupplyLens_End_to_End_Project.ipynb",
    "scripts/download_data.py",
    "scripts/validate_data.py",
    "scripts/train.py",
    "scripts/score.py",
    "scripts/build_reports.py",
    "scripts/validate_project.py",
    "app/app.py",
    "reports/metrics/final_metrics.json",
    "reports/tables/shipment_intervention_queue.csv",
    "reports/tables/supplier_scorecard.csv",
    "reports/tables/replenishment_risk_indicators.csv",
    "reports/html/SupplyLens_End_to_End_Project.html",
    ".github/workflows/tests.yml",
    "docs/DATA_SOURCE_REVIEW.md",
    "docs/DATA_PROVENANCE.md",
    "docs/DATA_DICTIONARY.md",
    "docs/FEATURE_AVAILABILITY.md",
    "docs/BUSINESS_ASSUMPTIONS.md",
    "docs/MODEL_CARD.md",
    "docs/DEPLOYMENT_AND_MONITORING.md",
    "docs/SCORING_CONTRACT.md",
]


def repository_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / line for line in completed.stdout.splitlines() if line]


def check_required_files(errors: list[str]) -> None:
    for relative in REQUIRED_FILES:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Required nonempty file missing: {relative}")


def check_source_and_artifacts(errors: list[str]) -> None:
    config = load_config()
    raw_path = resolve_path(config["data"]["raw_path"])
    if not raw_path.is_file():
        errors.append("Raw source is missing; run the data acquisition command")
    elif file_sha256(raw_path) != config["data"]["sha256"]:
        errors.append("Raw source checksum does not match configuration")

    model_path = resolve_path(config["outputs"]["model_path"])
    if not model_path.is_file() or model_path.stat().st_size == 0:
        errors.append("Serialized model artifact is missing")

    try:
        metrics = json.loads(
            resolve_path(config["outputs"]["final_metrics_path"]).read_text(encoding="utf-8")
        )
        if metrics["splits"]["test"]["rows"] != 1479:
            errors.append("Final metrics contain an unexpected test row count")
        if metrics["report_artifacts"]["figure_count"] < 15:
            errors.append("Expected at least 15 report figures")
        if metrics["report_artifacts"]["table_count"] < 26:
            errors.append("Expected at least 26 report tables")
    except (KeyError, ValueError, OSError) as exc:
        errors.append(f"Final metrics artifact is invalid: {exc}")


def check_readme_is_generated(errors: list[str]) -> None:
    readme = ROOT / "README.md"
    before = readme.read_bytes()
    completed = subprocess.run(
        [sys.executable, "scripts/build_readme.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        errors.append(f"README regeneration failed: {completed.stderr.strip()}")
    elif readme.read_bytes() != before:
        errors.append("README did not match the metrics-generated version")


def check_notebook(errors: list[str]) -> None:
    path = ROOT / "notebooks" / "SupplyLens_End_to_End_Project.ipynb"
    if not path.exists():
        return
    notebook = nbformat.read(path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not code_cells or any(cell.execution_count is None for cell in code_cells):
        errors.append("Canonical notebook is not fully executed")
    for cell in code_cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append("Canonical notebook contains an error output")
                return


def check_repository_text(errors: list[str], warnings: list[str]) -> None:
    absolute_patterns = [
        re.compile(r"[A-Za-z]:[\\/]Users[\\/]", re.IGNORECASE),
        re.compile(r"/(?:Users|home)/[^/\s]+/"),
    ]
    control_pattern = re.compile(r"[\x00-\x09\x0b\x0c\x0e-\x1f]")
    secret_patterns = [
        re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(
            r"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*['\"][^'\"]{8,}"
        ),
    ]
    for path in repository_files():
        relative = path.relative_to(ROOT).as_posix()
        if not path.is_file():
            continue
        if path.stat().st_size > 25 * 1024 * 1024:
            errors.append(f"Repository file exceeds 25 MiB: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            warnings.append(f"Skipped non-UTF-8 text scan: {relative}")
            continue
        if control_pattern.search(content):
            errors.append(f"Unexpected control character found in {relative}")
        if any(pattern.search(content) for pattern in absolute_patterns):
            errors.append(f"Machine-specific absolute path found in {relative}")
        if any(pattern.search(content) for pattern in secret_patterns):
            errors.append(f"Possible credential found in {relative}")


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    check_required_files(errors)
    check_source_and_artifacts(errors)
    check_readme_is_generated(errors)
    check_notebook(errors)
    check_repository_text(errors, warnings)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        print("Project validation failed:")
        for error in sorted(set(errors)):
            print(f"- {error}")
        return 1
    print("Project validation passed: structure, source, artifacts, notebook, README, paths, and repository text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
