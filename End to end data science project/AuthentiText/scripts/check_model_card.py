"""Verify model-card evidence against committed experiment reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = REPO_ROOT / "data" / "metadata"
MODEL_CARD = REPO_ROOT / "docs" / "MODEL_CARD.md"


def _load(name: str) -> dict[str, Any]:
    with (METADATA_DIR / name).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _model(report: dict[str, Any], model_type: str) -> dict[str, Any]:
    return next(item for item in report["models"] if item["model_type"] == model_type)


def _artifact(report: dict[str, Any], model_type: str) -> dict[str, Any]:
    return next(item for item in report["artifacts"] if item["model_type"] == model_type)


def _count(value: int) -> str:
    return f"{value:,}"


def _metric(value: float) -> str:
    return f"{value:.6f}"


def _percent(value: float) -> str:
    return f"{value * 100:.4f}%"


def expected_evidence_lines() -> list[str]:
    training = _load("mage_baseline_training_report.json")
    validation = _load("mage_baseline_validation_report.json")
    calibration = _load("mage_calibration_report.json")
    frozen = _load("mage_frozen_test_report.json")
    ood = _load("mage_ood_evaluation_report.json")
    domain_holdouts = _load("mage_domain_holdout_report.json")
    generator_holdouts = _load("mage_generator_holdout_report.json")
    external = _load("ghostbuster_evaluation_report.json")
    truncation = _load("mage_truncation_robustness_report.json")

    word_artifact = _artifact(training, "word_tfidf_logistic")
    word_validation = _model(validation, "word_tfidf_logistic")
    validation_overall = word_validation["metrics"]["overall"]
    validation_calibration = word_validation["metrics"]["calibration"]
    frozen_raw = frozen["metrics"]["raw_score"]["overall"]
    frozen_calibration = frozen["metrics"]["calibrated_probability"]
    frozen_policy = frozen["metrics"]["policy"]
    ood_combined = ood["metrics"]["content_deduplicated_combined"]
    ood_policy = ood_combined["policy"]
    ood_calibration = ood_combined["calibrated_probability"]
    ood_ranking = ood_combined["raw_ranking"]
    external_raw = external["metrics"]["raw_score"]["overall"]
    external_calibration = external["metrics"]["calibrated_probability"]
    external_policy = external["metrics"]["policy"]
    training_targets = training["input"]["target_counts"]
    roles = calibration["roles"]
    config = word_artifact["configuration"]

    lines = [
        f"| Dataset | `{training['dataset_id']}` at `{training['revision']}` |",
        "| Sanitized training rows | "
        f"{_count(training['input']['rows'])} "
        f"({_count(training_targets['0'])} human; "
        f"{_count(training_targets['1'])} machine) |",
        "| Base artifact | "
        f"{_count(word_artifact['bytes'])} bytes; SHA-256 `{word_artifact['sha256']}` |",
        "| Calibrator artifact | "
        f"{_count(calibration['artifact']['bytes'])} bytes; "
        f"SHA-256 `{calibration['artifact']['sha256']}` |",
        "| Feature space | Lowercased word 1–2 grams; "
        f"{_count(config['vectorizer']['max_features'])} maximum features; "
        f"`min_df={config['vectorizer']['min_df']}`; "
        f"`max_df={config['vectorizer']['max_df']}` |",
        "| Classifier | Balanced logistic regression; "
        f"`C={config['classifier']['C']}`; SAGA; seed "
        f"{config['classifier']['random_state']} |",
        "| Calibration | "
        f"{calibration['method_selection']['selected_method'].title()} regression |",
        f"| Likely-human maximum | `{calibration['thresholds']['human_threshold']:.12f}` |",
        f"| Likely-machine minimum | `{calibration['thresholds']['machine_threshold']:.12f}` |",
        f"| Fit isotonic or sigmoid calibration | {_count(roles['calibration_fit']['rows'])} |",
        "| Select calibration method and thresholds | "
        f"{_count(roles['policy_selection']['rows'])} |",
        f"| Audit the selected policy | {_count(roles['calibration_audit']['rows'])} |",
        "| Sanitized MAGE validation, raw model | "
        f"{_count(validation_overall['rows'])} | {_metric(validation_overall['roc_auc'])} | "
        f"{_metric(validation_overall['average_precision'])} | "
        f"{_metric(validation_overall['brier_score'])} | "
        f"{_metric(validation_calibration['expected_calibration_error'])} | n/a |",
        "| Frozen sanitized MAGE test | "
        f"{_count(frozen_raw['rows'])} | {_metric(frozen_raw['roc_auc'])} | "
        f"{_metric(frozen_raw['average_precision'])} | "
        f"{_metric(frozen_calibration['brier_score'])} | "
        f"{_metric(frozen_calibration['expected_calibration_error'])} | "
        f"{_percent(frozen_policy['uncertain_rate'])} |",
        "| Deduplicated MAGE development OOD | "
        f"{_count(ood_combined['rows'])} | {_metric(ood_ranking['roc_auc'])} | "
        f"{_metric(ood_ranking['average_precision'])} | "
        f"{_metric(ood_calibration['brier_score'])} | "
        f"{_metric(ood_calibration['expected_calibration_error'])} | "
        f"{_percent(ood_policy['uncertain_rate'])} |",
        "| Frozen Ghostbuster external | "
        f"{_count(external_raw['rows'])} | {_metric(external_raw['roc_auc'])} | "
        f"{_metric(external_raw['average_precision'])} | "
        f"{_metric(external_calibration['brier_score'])} | "
        f"{_metric(external_calibration['expected_calibration_error'])} | "
        f"{_percent(external_policy['uncertain_rate'])} |",
        "| Sanitized MAGE test | "
        f"{_percent(frozen_policy['coverage'])} | "
        f"{_percent(frozen_policy['human_false_machine_rate'])} | "
        f"{_percent(frozen_policy['machine_false_human_rate'])} | "
        f"{_percent(frozen_policy['decisive_accuracy'])} |",
        "| Deduplicated MAGE development OOD | "
        f"{_percent(ood_policy['coverage'])} | "
        f"{_percent(ood_policy['human_false_machine_rate'])} | "
        f"{_percent(ood_policy['machine_false_human_rate'])} | "
        f"{_percent(ood_policy['decisive_accuracy'])} |",
        "| Frozen Ghostbuster external | "
        f"{_percent(external_policy['coverage'])} | "
        f"{_percent(external_policy['human_false_machine_rate'])} | "
        f"{_percent(external_policy['machine_false_human_rate'])} | "
        f"{_percent(external_policy['decisive_accuracy'])} |",
    ]

    subgroup_specs = (
        ("Under 50 whitespace tokens", "policy_by_whitespace_token_band", "under_50"),
        ("HellaSwag domain", "policy_by_domain", "hswag"),
        ("Yelp domain", "policy_by_domain", "yelp"),
    )
    for label, group, key in subgroup_specs:
        metrics = frozen["metrics"][group][key]
        lines.append(
            f"| {label} | {_count(metrics['rows'])} | "
            f"{_percent(metrics['uncertain_rate'])} | "
            f"{_percent(metrics['human_false_machine_rate'])} | "
            f"{_percent(metrics['machine_false_human_rate'])} | "
            f"{_percent(metrics['decisive_accuracy'])} |"
        )

    external_student = external["metrics"]["policy_by_domain"]["student_essay"]
    lines.append(
        f"| Ghostbuster student essays | {_count(external_student['rows'])} | "
        f"{_percent(external_student['uncertain_rate'])} | "
        f"{_percent(external_student['human_false_machine_rate'])} | "
        f"{_percent(external_student['machine_false_human_rate'])} | "
        f"{_percent(external_student['decisive_accuracy'])} |"
    )

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

    performance_specs = (
        ("Validation", word_validation["scoring"]),
        ("Frozen test", frozen["scoring"]),
        ("Raw OOD files, including repeated controls", ood["scoring"]),
        ("Ghostbuster external", external["scoring"]),
    )
    for label, scoring in performance_specs:
        lines.append(
            f"| {label} | {_count(scoring['batch_size'])} | "
            f"{scoring['seconds']:.3f} | {_count(scoring['records_per_second'])} |"
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


def check_model_card(path: Path = MODEL_CARD) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    for line in expected_evidence_lines():
        occurrences = text.count(line)
        if occurrences != 1:
            errors.append(f"expected one occurrence, found {occurrences}: {line}")

    required_statements = (
        "This system provides a statistical estimate and should not be treated as proof",
        "No transformer was trained or evaluated.",
        "Ghostbuster supplies cross-dataset evidence",
        "must not be described as production-ready",
    )
    for statement in required_statements:
        if statement not in text:
            errors.append(f"missing required limitation: {statement}")
    return errors


def main() -> int:
    errors = check_model_card()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        f"Checked {len(expected_evidence_lines())} model-card evidence rows "
        "against committed experiment reports."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
