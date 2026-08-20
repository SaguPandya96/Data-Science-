"""Check local documentation links and README evidence against committed reports."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
GENERATOR_EVALUATION = REPO_ROOT / "docs" / "evaluation" / "mage_generator_holdouts.md"
METADATA_DIR = REPO_ROOT / "data" / "metadata"
LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _load(name: str) -> dict[str, Any]:
    with (METADATA_DIR / name).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _output(split_report: dict[str, Any], partition: str) -> dict[str, Any]:
    return next(item for item in split_report["outputs"] if item["partition"] == partition)


def _artifact(training_report: dict[str, Any], model_type: str) -> dict[str, Any]:
    return next(item for item in training_report["artifacts"] if item["model_type"] == model_type)


def _count(value: int) -> str:
    return f"{value:,}"


def _metric(value: float) -> str:
    return f"{value:.6f}"


def _percent(value: float) -> str:
    return f"{value * 100:.4f}%"


def expected_readme_evidence() -> list[str]:
    split = _load("mage_id_split_report.json")
    training = _load("mage_baseline_training_report.json")
    calibration = _load("mage_calibration_report.json")
    frozen = _load("mage_frozen_test_report.json")
    ood = _load("mage_ood_evaluation_report.json")
    domain_holdouts = _load("mage_domain_holdout_report.json")
    generator_holdouts = _load("mage_generator_holdout_report.json")
    external = _load("ghostbuster_evaluation_report.json")
    truncation = _load("mage_truncation_robustness_report.json")

    train = _output(split, "train")
    validation = _output(split, "validation")
    test = _output(split, "test")
    model = _artifact(training, "word_tfidf_logistic")
    raw = frozen["metrics"]["raw_score"]["overall"]
    policy = frozen["metrics"]["policy"]
    combined = ood["metrics"]["content_deduplicated_combined"]
    ood_ranking = combined["raw_ranking"]
    ood_policy = combined["policy"]
    external_raw = external["metrics"]["raw_score"]["overall"]
    external_policy = external["metrics"]["policy"]

    lines = [
        "| Sanitized train | "
        f"{_count(train['rows_written'])} | {_count(train['target_counts']['0'])} | "
        f"{_count(train['target_counts']['1'])} | Fit baseline |",
        "| Sanitized validation | "
        f"{_count(validation['rows_written'])} | "
        f"{_count(validation['target_counts']['0'])} | "
        f"{_count(validation['target_counts']['1'])} | "
        "EDA, candidate evaluation, calibration roles, drift reference |",
        "| Sanitized test | "
        f"{_count(test['rows_written'])} | {_count(test['target_counts']['0'])} | "
        f"{_count(test['target_counts']['1'])} | One frozen in-distribution evaluation |",
        f"| Base model | {_count(model['bytes'])} bytes; SHA-256 `{model['sha256']}` |",
        "| Calibrator | "
        f"{_count(calibration['artifact']['bytes'])} bytes; "
        f"SHA-256 `{calibration['artifact']['sha256']}` |",
        f"| Likely-human maximum | `{calibration['thresholds']['human_threshold']:.12f}` |",
        f"| Likely-machine minimum | `{calibration['thresholds']['machine_threshold']:.12f}` |",
        "| Frozen sanitized MAGE test | "
        f"{_count(raw['rows'])} | {_metric(raw['roc_auc'])} | "
        f"{_metric(raw['average_precision'])} | {_percent(policy['uncertain_rate'])} | "
        f"{_percent(policy['human_false_machine_rate'])} | "
        f"{_percent(policy['machine_false_human_rate'])} |",
        "| Deduplicated MAGE development OOD | "
        f"{_count(combined['rows'])} | {_metric(ood_ranking['roc_auc'])} | "
        f"{_metric(ood_ranking['average_precision'])} | "
        f"{_percent(ood_policy['uncertain_rate'])} | "
        f"{_percent(ood_policy['human_false_machine_rate'])} | "
        f"{_percent(ood_policy['machine_false_human_rate'])} |",
        "| Frozen Ghostbuster external | "
        f"{_count(external_raw['rows'])} | {_metric(external_raw['roc_auc'])} | "
        f"{_metric(external_raw['average_precision'])} | "
        f"{_percent(external_policy['uncertain_rate'])} | "
        f"{_percent(external_policy['human_false_machine_rate'])} | "
        f"{_percent(external_policy['machine_false_human_rate'])} |",
    ]
    for condition in truncation["conditions"]:
        original = condition["original"]
        truncated = condition["truncated"]
        changed = condition["paired_effects"]["category_changes"]
        lines.append(
            f"| {condition['budget_whitespace_tokens']}-token prefix | "
            f"{_count(condition['selection']['rows'])} | "
            f"{_metric(original['raw_score']['roc_auc'])} | "
            f"{_metric(truncated['raw_score']['roc_auc'])} | "
            f"{_percent(original['policy']['uncertain_rate'])} | "
            f"{_percent(truncated['policy']['uncertain_rate'])} | "
            f"{_percent(changed['changed_rate'])} |"
        )
    domain_summary_specs = (
        ("ROC AUC", "roc_auc", _metric),
        ("Average precision", "average_precision", _metric),
        ("Brier score", "brier_score", _metric),
        ("Expected calibration error", "expected_calibration_error", _metric),
        ("Coverage", "coverage", _percent),
        ("Uncertain rate", "uncertain_rate", _percent),
        ("Human false-machine rate", "human_false_machine_rate", _percent),
        ("Machine false-human rate", "machine_false_human_rate", _percent),
    )
    for label, key, formatter in domain_summary_specs:
        metric = domain_holdouts["summary"][key]
        lines.append(
            f"| {label} | {formatter(metric['minimum'])} | "
            f"{formatter(metric['median'])} | {formatter(metric['maximum'])} |"
        )
    generator_summary_specs = (
        ("ROC AUC", "roc_auc", _metric),
        ("Average precision", "average_precision", _metric),
        ("Brier score", "brier_score", _metric),
        ("Expected calibration error", "expected_calibration_error", _metric),
        ("Coverage", "coverage", _percent),
        ("Uncertain rate", "uncertain_rate", _percent),
        ("Human false-machine rate", "human_false_machine_rate", _percent),
        ("Machine false-human rate", "machine_false_human_rate", _percent),
    )
    for label, key, formatter in generator_summary_specs:
        metric = generator_holdouts["summary"][key]
        lines.append(
            f"| {label} | {formatter(metric['minimum'])} | "
            f"{formatter(metric['median'])} | {formatter(metric['maximum'])} |"
        )
    return lines


def expected_generator_evaluation_evidence() -> list[str]:
    report = _load("mage_generator_holdout_report.json")
    lines = []
    for fold in report["folds"]:
        metrics = fold["test_evaluation"]["metrics"]
        ranking = metrics["raw_score"]["overall"]
        calibrated = metrics["calibrated_probability"]
        policy = metrics["policy"]
        lines.append(
            f"| `{fold['held_out_generator']}` | {_count(ranking['rows'])} | "
            f"{_metric(ranking['roc_auc'])} | {_metric(ranking['average_precision'])} | "
            f"{_metric(calibrated['brier_score'])} | "
            f"{_metric(calibrated['expected_calibration_error'])} | "
            f"{_percent(policy['uncertain_rate'])} | "
            f"{_percent(policy['human_false_machine_rate'])} | "
            f"{_percent(policy['machine_false_human_rate'])} |"
        )
    return lines


def _markdown_files() -> list[Path]:
    return sorted([*REPO_ROOT.glob("*.md"), *(REPO_ROOT / "docs").rglob("*.md")])


def _check_links(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for match in LINK_PATTERN.finditer(text):
            destination = match.group(1).strip().strip("<>")
            if destination.startswith(("#", "http://", "https://", "mailto:")):
                continue
            destination = destination.split(maxsplit=1)[0].split("#", maxsplit=1)[0]
            if not destination:
                continue
            resolved = (path.parent / destination).resolve()
            if not resolved.exists():
                relative_source = path.relative_to(REPO_ROOT).as_posix()
                errors.append(f"{relative_source}: missing link target {destination}")
    return errors


def check_documentation() -> tuple[list[str], int]:
    files = _markdown_files()
    errors = _check_links(files)
    readme = README.read_text(encoding="utf-8")
    normalized = " ".join(readme.split())

    for line in expected_readme_evidence():
        occurrences = readme.count(line)
        if occurrences != 1:
            errors.append(f"README expected one occurrence, found {occurrences}: {line}")

    generator_evaluation = GENERATOR_EVALUATION.read_text(encoding="utf-8")
    for line in expected_generator_evaluation_evidence():
        occurrences = generator_evaluation.count(line)
        if occurrences != 1:
            errors.append(
                f"generator evaluation expected one occurrence, found {occurrences}: {line}"
            )

    required_sections = (
        "## Current status",
        "## System architecture",
        "## Data and evaluation design",
        "## Model and calibration",
        "## Measured results",
        "## Environment and setup",
        "## Run locally",
        "## Reproduce data and experiments",
        "## API, monitoring, and drift",
        "## Testing and CI",
        "## Responsible use and limitations",
        "## Documentation map",
    )
    for section in required_sections:
        if section not in readme:
            errors.append(f"README missing section: {section}")

    required_statements = (
        "This system provides a statistical estimate and should not be treated as proof of "
        "authorship.",
        "the evaluated BERT-Tiny candidate regresses severely on MAGE OOD",
        "The Ghostbuster external evaluation does not establish production safety.",
        "Hosted CI, both container builds, full transformer training, and the frozen "
        "transformer evaluation have passed",
        "no deployment acceptance test, production-user validation, or production rollback test",
    )
    normalized_lower = normalized.lower()
    for statement in required_statements:
        if statement.lower() not in normalized_lower:
            errors.append(f"README missing required limitation: {statement}")

    drift = _load("mage_drift_reference.json")
    minimum = drift["thresholds"]["minimum_observations"]
    false_alert = drift["backtest"]["false_alert_audit"]
    domain = drift["backtest"]["domain_shift"]
    drift_evidence = (
        f"below {_count(minimum)} successful items. Its validation-only audit flagged "
        f"{false_alert['flagged']} of {false_alert['windows']} same-distribution windows and "
        f"all {domain['detected']} of {domain['domains']} held-out domain groups."
    )
    if drift_evidence not in normalized:
        errors.append(f"README drift evidence does not match reports: {drift_evidence}")

    registry = _load("experiment_registry.json")
    registry_evidence = (
        f"bind {_count(len(registry['experiments']))} completed runs to validated source-report "
        f"hashes and milestone commits. {_count(len(registry['not_run']))} unrun candidates"
    )
    if registry_evidence not in normalized:
        errors.append(f"README registry evidence does not match registry: {registry_evidence}")
    return errors, len(files)


def main() -> int:
    errors, file_count = check_documentation()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        f"Checked {file_count} Markdown files, all local links, and "
        f"{len(expected_readme_evidence())} README evidence rows plus "
        f"{len(expected_generator_evaluation_evidence())} generator evidence rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
