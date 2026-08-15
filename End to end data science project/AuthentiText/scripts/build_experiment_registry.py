"""Build a deterministic registry from completed AuthentiText experiment reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = REPO_ROOT / "data" / "metadata"
DEFAULT_OUTPUT = METADATA_DIR / "experiment_registry.json"

REPORTS = {
    "baseline_training_v1": (
        "training",
        "23fe60c",
        "mage_baseline_training_report.json",
        "Retain all three fitted controls for evaluation.",
    ),
    "baseline_validation_v1": (
        "evaluation",
        "5782059",
        "mage_baseline_validation_report.json",
        "Advance word TF-IDF; reject threshold 0.5 for product decisions.",
    ),
    "calibration_policy_v1": (
        "calibration",
        "614c59c",
        "mage_calibration_report.json",
        "Select isotonic calibration and freeze a three-way abstaining policy.",
    ),
    "frozen_id_test_v1": (
        "final_in_distribution_evaluation",
        "668d870",
        "mage_frozen_test_report.json",
        "Preserve the test misses without threshold retuning.",
    ),
    "validation_drift_backtest_v1": (
        "operational_backtest",
        "8fb484d",
        "mage_drift_reference.json",
        "Use aggregate flags for investigation only.",
    ),
    "mage_development_ood_v1": (
        "development_ood_evaluation",
        "c8dcec2",
        "mage_ood_evaluation_report.json",
        "Retain the degraded OOD result and prohibit retuning from it.",
    ),
    "mage_domain_holdout_v1": (
        "domain_generalization_evaluation",
        "49c7fd8",
        "mage_domain_holdout_report.json",
        "Retain measured domain dependence and prohibit test-driven fold retuning.",
    ),
    "mage_generator_holdout_v1": (
        "exact_generator_generalization_evaluation",
        "1f5ae54",
        "mage_generator_holdout_report.json",
        "Retain measured exact-generator dependence without family-level claims.",
    ),
    "ghostbuster_external_evaluation": (
        "external_generalization_evaluation",
        "deae0f0",
        "ghostbuster_evaluation_report.json",
        "Retain external calibration and human false-machine failures without retuning.",
    ),
    "mage_truncation_robustness_v1": (
        "paired_robustness_evaluation",
        "54649bd",
        "mage_truncation_robustness_report.json",
        "Retain measured prefix sensitivity without model, calibration, or threshold retuning.",
    ),
}


def _load_report(name: str) -> dict[str, Any]:
    path = METADATA_DIR / name
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    if payload.get("validation", {}).get("status") != "pass":
        raise ValueError(f"{name} does not have validation.status='pass'")
    return payload


def _canonical_text_sha256(path: Path) -> str:
    """Hash UTF-8 report content independently of checkout line endings."""
    text = path.read_bytes().decode("utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _artifact(report: dict[str, Any], model_type: str) -> dict[str, Any]:
    return next(item for item in report["artifacts"] if item["model_type"] == model_type)


def _model(report: dict[str, Any], model_type: str) -> dict[str, Any]:
    return next(item for item in report["models"] if item["model_type"] == model_type)


def _report_identity(name: str) -> dict[str, Any]:
    path = METADATA_DIR / name
    return {
        "relative_path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": _canonical_text_sha256(path),
    }


def _training_evidence(report: dict[str, Any]) -> dict[str, Any]:
    word = _artifact(report, "word_tfidf_logistic")
    return {
        "artifact_reload_verified": report["validation"]["artifacts_reload_and_score"],
        "candidate_types": [item["model_type"] for item in report["artifacts"]],
        "test_data_used": report["configuration"]["test_data_used"],
        "train_rows": report["input"]["rows"],
        "validation_data_used": report["configuration"]["validation_data_used"],
        "word_artifact_bytes": word["bytes"],
        "word_artifact_sha256": word["sha256"],
    }


def _validation_evidence(report: dict[str, Any]) -> dict[str, Any]:
    word = _model(report, "word_tfidf_logistic")
    overall = word["metrics"]["overall"]
    return {
        "average_precision": overall["average_precision"],
        "candidates_evaluated": report["validation"]["models_evaluated"],
        "human_false_machine_rate_at_0_5": overall["false_positive_rate"],
        "records_per_second": word["scoring"]["records_per_second"],
        "roc_auc": overall["roc_auc"],
        "rows": overall["rows"],
    }


def _calibration_evidence(report: dict[str, Any]) -> dict[str, Any]:
    audit = report["calibration_audit"]["policy_metrics"]
    return {
        "audit_coverage": audit["coverage"],
        "audit_human_false_machine_rate": audit["human_false_machine_rate"],
        "audit_machine_false_human_rate": audit["machine_false_human_rate"],
        "audit_rows": audit["rows"],
        "fit_rows": report["roles"]["calibration_fit"]["rows"],
        "human_threshold": report["thresholds"]["human_threshold"],
        "machine_threshold": report["thresholds"]["machine_threshold"],
        "selected_method": report["method_selection"]["selected_method"],
        "selection_rows": report["roles"]["policy_selection"]["rows"],
    }


def _frozen_test_evidence(report: dict[str, Any]) -> dict[str, Any]:
    raw = report["metrics"]["raw_score"]["overall"]
    calibrated = report["metrics"]["calibrated_probability"]
    policy = report["metrics"]["policy"]
    return {
        "average_precision": raw["average_precision"],
        "brier_score": calibrated["brier_score"],
        "coverage": policy["coverage"],
        "expected_calibration_error": calibrated["expected_calibration_error"],
        "human_false_machine_rate": policy["human_false_machine_rate"],
        "machine_false_human_rate": policy["machine_false_human_rate"],
        "retuning_allowed": report["configuration"]["retuning_after_test_allowed"],
        "roc_auc": raw["roc_auc"],
        "rows": raw["rows"],
        "uncertain_rate": policy["uncertain_rate"],
    }


def _drift_evidence(report: dict[str, Any]) -> dict[str, Any]:
    false_alert = report["backtest"]["false_alert_audit"]
    domain = report["backtest"]["domain_shift"]
    return {
        "automatic_action": report["configuration"]["automatic_action"],
        "audit_rows": sum(item["rows"] for item in false_alert["results"]),
        "domain_shifts_detected": domain["detected"],
        "domain_shifts_tested": domain["domains"],
        "false_alert_rate": false_alert["false_alert_rate"],
        "false_alert_windows": false_alert["windows"],
        "false_alerts": false_alert["flagged"],
        "reference_rows": report["identity"]["reference"]["rows"],
        "test_data_used": report["configuration"]["test_data_used"],
        "validation_rows": report["identity"]["validation"]["rows"],
    }


def _ood_evidence(report: dict[str, Any]) -> dict[str, Any]:
    combined = report["metrics"]["content_deduplicated_combined"]
    ranking = combined["raw_ranking"]
    calibrated = combined["calibrated_probability"]
    policy = combined["policy"]
    return {
        "average_precision": ranking["average_precision"],
        "brier_score": calibrated["brier_score"],
        "duplicate_extra_rows": report["content_accounting"]["duplicate_extra_rows"],
        "expected_calibration_error": calibrated["expected_calibration_error"],
        "human_false_machine_rate": policy["human_false_machine_rate"],
        "machine_false_human_rate": policy["machine_false_human_rate"],
        "published_test_data_used": report["validation"]["published_test_data_used"],
        "raw_rows": report["content_accounting"]["rows"],
        "roc_auc": ranking["roc_auc"],
        "uncertain_rate": policy["uncertain_rate"],
        "unique_rows": combined["rows"],
    }


def _domain_holdout_evidence(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    return {
        "artifacts_and_predictions_verified": report["validation"][
            "all_artifacts_and_predictions_verified"
        ],
        "folds": report["validation"]["folds_completed"],
        "human_false_machine_rate_maximum": summary["human_false_machine_rate"]["maximum"],
        "human_false_machine_rate_median": summary["human_false_machine_rate"]["median"],
        "roc_auc_maximum": summary["roc_auc"]["maximum"],
        "roc_auc_median": summary["roc_auc"]["median"],
        "roc_auc_minimum": summary["roc_auc"]["minimum"],
        "test_data_used_for_training_calibration_or_thresholds": report["configuration"][
            "test_data_used_for_training_calibration_or_thresholds"
        ],
        "uncertain_rate_median": summary["uncertain_rate"]["median"],
    }


def _generator_holdout_evidence(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    prevalences = [
        fold["test_evaluation"]["input"]["target_counts"]["1"]
        / fold["test_evaluation"]["input"]["rows"]
        for fold in report["folds"]
    ]
    return {
        "artifacts_and_predictions_verified": report["validation"][
            "all_artifacts_and_predictions_verified"
        ],
        "expected_calibration_error_median": summary["expected_calibration_error"]["median"],
        "folds": report["validation"]["folds_completed"],
        "human_false_machine_rate_maximum": summary["human_false_machine_rate"]["maximum"],
        "machine_false_human_rate_maximum": summary["machine_false_human_rate"]["maximum"],
        "machine_prevalence_maximum": round(max(prevalences), 6),
        "machine_prevalence_minimum": round(min(prevalences), 6),
        "roc_auc_maximum": summary["roc_auc"]["maximum"],
        "roc_auc_median": summary["roc_auc"]["median"],
        "roc_auc_minimum": summary["roc_auc"]["minimum"],
        "test_data_used_for_training_calibration_or_thresholds": report["configuration"][
            "test_data_used_for_training_calibration_or_thresholds"
        ],
        "uncertain_rate_median": summary["uncertain_rate"]["median"],
    }


def _external_evidence(report: dict[str, Any]) -> dict[str, Any]:
    raw = report["metrics"]["raw_score"]["overall"]
    calibrated = report["metrics"]["calibrated_probability"]
    policy = report["metrics"]["policy"]
    student = report["metrics"]["policy_by_domain"]["student_essay"]
    claude = report["metrics"]["machine_outcomes_by_generator"]["claude"]
    return {
        "average_precision": raw["average_precision"],
        "brier_score": calibrated["brier_score"],
        "coverage": policy["coverage"],
        "expected_calibration_error": calibrated["expected_calibration_error"],
        "external_records_excluded": report["selection"]["excluded_rows"],
        "external_results_may_change_policy": not report["configuration"][
            "external_results_may_not_change_model_calibration_or_thresholds"
        ],
        "human_false_machine_rate": policy["human_false_machine_rate"],
        "machine_false_human_rate": policy["machine_false_human_rate"],
        "machine_prevalence": round(raw["target_counts"]["1"] / raw["rows"], 6),
        "prediction_file_verified": report["validation"]["prediction_file_verified"],
        "roc_auc": raw["roc_auc"],
        "rows": raw["rows"],
        "student_essay_human_false_machine_rate": student["human_false_machine_rate"],
        "uncertain_rate": policy["uncertain_rate"],
        "claude_likely_machine_rate": claude["likely_machine_rate"],
    }


def _truncation_evidence(report: dict[str, Any]) -> dict[str, Any]:
    conditions = {}
    for condition in report["conditions"]:
        original = condition["original"]
        truncated = condition["truncated"]
        paired = condition["paired_effects"]
        conditions[str(condition["budget_whitespace_tokens"])] = {
            "category_changed_rate": paired["category_changes"]["changed_rate"],
            "original_human_false_machine_rate": original["policy"]["human_false_machine_rate"],
            "original_machine_false_human_rate": original["policy"]["machine_false_human_rate"],
            "original_roc_auc": original["raw_score"]["roc_auc"],
            "original_uncertain_rate": original["policy"]["uncertain_rate"],
            "paired_rows": condition["selection"]["rows"],
            "prefix_human_false_machine_rate": truncated["policy"]["human_false_machine_rate"],
            "prefix_machine_false_human_rate": truncated["policy"]["machine_false_human_rate"],
            "prefix_roc_auc": truncated["raw_score"]["roc_auc"],
            "prefix_uncertain_rate": truncated["policy"]["uncertain_rate"],
        }
    return {
        "budgets_whitespace_tokens": report["configuration"]["budgets_whitespace_tokens"],
        "conditions": conditions,
        "model_calibration_and_thresholds_frozen": report["configuration"][
            "model_calibration_and_thresholds_frozen"
        ],
        "prediction_file_verified": report["validation"]["prediction_file_verified"],
        "retuning_allowed": report["configuration"]["retuning_after_robustness_evaluation_allowed"],
        "total_paired_rows": report["predictions"]["rows"],
    }


EVIDENCE_BUILDERS = {
    "baseline_training_v1": _training_evidence,
    "baseline_validation_v1": _validation_evidence,
    "calibration_policy_v1": _calibration_evidence,
    "frozen_id_test_v1": _frozen_test_evidence,
    "validation_drift_backtest_v1": _drift_evidence,
    "mage_development_ood_v1": _ood_evidence,
    "mage_domain_holdout_v1": _domain_holdout_evidence,
    "mage_generator_holdout_v1": _generator_holdout_evidence,
    "ghostbuster_external_evaluation": _external_evidence,
    "mage_truncation_robustness_v1": _truncation_evidence,
}


def build_registry() -> dict[str, Any]:
    experiments = []
    for experiment_id, (kind, commit, report_name, decision) in REPORTS.items():
        report = _load_report(report_name)
        experiments.append(
            {
                "decision": decision,
                "evidence": EVIDENCE_BUILDERS[experiment_id](report),
                "experiment_id": experiment_id,
                "kind": kind,
                "milestone_commit": commit,
                "report": _report_identity(report_name),
                "status": "complete",
            }
        )

    return {
        "experiments": experiments,
        "not_run": [
            {
                "experiment_id": "transformer_candidate",
                "reason": "Deferred on the audited CPU-only workstation; no model was trained.",
            },
            {
                "experiment_id": "raid_robustness_evaluation",
                "reason": "No storage-safe source-group acquisition plan has been executed.",
            },
            {
                "experiment_id": "multilingual_evaluation",
                "reason": "The first research cycle is English-only.",
            },
        ],
        "registry_version": 1,
        "schema_version": 1,
        "scope": "completed_local_experiments_and_explicitly_unrun_candidates",
        "validation": {
            "completed_experiments": len(experiments),
            "source_report_hashes_recorded": True,
            "source_reports_validation_status_pass": True,
            "status": "pass",
        },
    }


def render_registry() -> str:
    return json.dumps(build_registry(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--stdout", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rendered = render_registry()
    if args.stdout:
        print(rendered, end="")
        return 0
    if args.check:
        try:
            existing = args.output.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"ERROR: cannot read {args.output}: {exc}")
            return 1
        if existing != rendered:
            print(f"ERROR: {args.output} is not the deterministic current registry")
            return 1
        print(f"Verified {len(REPORTS)} hash-linked completed experiments in {args.output}.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote {len(REPORTS)} completed experiments to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
