"""Run leakage-aware leave-one-domain-out experiments over sanitized MAGE data."""

from __future__ import annotations

import gzip
import json
import re
import shutil
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from statistics import median
from typing import Any

from authentitext.data.cleaning import open_deterministic_gzip
from authentitext.data.regimes import RegimeError, parse_source_id
from authentitext.modeling.baselines import (
    BaselineError,
    sha256_file,
    train_baselines,
    verify_baselines,
)
from authentitext.modeling.calibration import (
    CalibrationError,
    fit_calibration_policy,
    verify_calibration_policy,
)
from authentitext.modeling.evaluation import (
    EvaluationError,
    evaluate_baselines,
    verify_evaluation,
)
from authentitext.modeling.frozen_test import (
    FrozenTestError,
    evaluate_frozen_test,
    verify_frozen_test,
)

DOMAIN_HOLDOUT_VERSION = 1
REPORT_SCHEMA_VERSION = 1
SAFE_DOMAIN = re.compile(r"^[A-Za-z0-9_.-]+$")


class DomainHoldoutError(RuntimeError):
    """Raised when a domain-holdout run cannot be executed or verified."""


def _safe_child(root: Path, name: str) -> Path:
    if not SAFE_DOMAIN.fullmatch(name):
        raise DomainHoldoutError(f"Unsafe domain identifier: {name!r}")
    resolved_root = root.resolve()
    child = (root / name).resolve()
    if not child.is_relative_to(resolved_root):
        raise DomainHoldoutError(f"Domain path escapes work root: {name!r}")
    return child


def _role_selector(role: str, held_out_domain: str) -> Callable[[str], bool]:
    if role in {"train", "validation"}:
        return lambda domain: domain != held_out_domain
    if role == "test":
        return lambda domain: domain == held_out_domain
    raise DomainHoldoutError(f"Unsupported domain-holdout role: {role!r}")


def materialize_domain_role(
    *,
    input_path: Path,
    output_path: Path,
    partition: str,
    role: str,
    held_out_domain: str,
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Create one deterministic, metadata-selected fold input and verify its counts."""
    selector = _role_selector(role, held_out_domain)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    rows_seen = 0
    rows_selected = 0
    targets: Counter[str] = Counter()
    sources: set[str] = set()
    record_ids: set[str] = set()
    try:
        with open_deterministic_gzip(temporary) as output_handle:
            with gzip.open(input_path, "rt", encoding="utf-8", newline="") as input_handle:
                for line_number, line in enumerate(input_handle, start=1):
                    rows_seen += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise DomainHoldoutError(
                            f"{input_path}:{line_number} is not valid JSON"
                        ) from error
                    source = record.get("source") if isinstance(record, dict) else None
                    target = record.get("target") if isinstance(record, dict) else None
                    record_id = record.get("record_id") if isinstance(record, dict) else None
                    if (
                        not isinstance(record, dict)
                        or record.get("partition") != partition
                        or not isinstance(source, str)
                        or target not in (0, 1)
                        or not isinstance(record_id, str)
                    ):
                        raise DomainHoldoutError(
                            f"{input_path}:{line_number} has invalid selector metadata"
                        )
                    try:
                        source_domain = parse_source_id(source).domain
                    except RegimeError as error:
                        raise DomainHoldoutError(
                            f"{input_path}:{line_number} has invalid source metadata"
                        ) from error
                    if not selector(source_domain):
                        continue
                    if record_id in record_ids:
                        raise DomainHoldoutError(
                            f"{input_path}:{line_number} has a duplicate selected record ID"
                        )
                    record_ids.add(record_id)
                    output_handle.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    rows_selected += 1
                    targets[str(target)] += 1
                    sources.add(source)
        temporary.replace(output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    observed_counts = {key: targets[key] for key in ("0", "1")}
    if rows_selected != expected["rows"]:
        output_path.unlink(missing_ok=True)
        raise DomainHoldoutError(
            f"{held_out_domain} {role} selected {rows_selected} rows; expected {expected['rows']}"
        )
    if observed_counts != expected["target_counts"]:
        output_path.unlink(missing_ok=True)
        raise DomainHoldoutError(
            f"{held_out_domain} {role} target counts {observed_counts!r}; "
            f"expected {expected['target_counts']!r}"
        )
    if len(sources) != expected["distinct_sources"]:
        output_path.unlink(missing_ok=True)
        raise DomainHoldoutError(
            f"{held_out_domain} {role} selected {len(sources)} sources; "
            f"expected {expected['distinct_sources']}"
        )
    return {
        "source_relative_path": input_path.name,
        "relative_path": output_path.name,
        "partition": partition,
        "selector_role": role,
        "held_out_domain": held_out_domain,
        "source_rows_scanned": rows_seen,
        "rows": rows_selected,
        "target_counts": observed_counts,
        "distinct_sources": len(sources),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
    }


def _word_model(report: dict[str, Any]) -> dict[str, Any]:
    return next(
        artifact
        for artifact in report["artifacts"]
        if artifact["model_type"] == "word_tfidf_logistic"
    )


def _word_validation(report: dict[str, Any]) -> dict[str, Any]:
    return next(model for model in report["models"] if model["model_type"] == "word_tfidf_logistic")


def _mark_verified(
    training: dict[str, Any],
    validation: dict[str, Any],
    calibration: dict[str, Any],
    test: dict[str, Any],
) -> None:
    training["validation"]["artifacts_reload_and_score"] = True
    validation["validation"]["prediction_files_verified"] = True
    validation["validation"]["metrics_recomputed_from_predictions"] = True
    calibration["validation"]["artifact_reload_and_score"] = True
    calibration["validation"]["source_prediction_verified"] = True
    calibration["validation"]["audit_metrics_recomputed"] = True
    test["validation"]["prediction_file_verified"] = True
    test["validation"]["metrics_recomputed_from_predictions"] = True


def _run_fold(
    *,
    fold_spec: dict[str, Any],
    input_root: Path,
    artifact_root: Path,
    work_root: Path,
    dataset_id: str,
    revision: str,
    tfidf_config: dict[str, Any] | None,
    logistic_config: dict[str, Any] | None,
) -> dict[str, Any]:
    domain = fold_spec["held_out_domain"]
    fold_artifacts = _safe_child(artifact_root, domain)
    fold_work = _safe_child(work_root, domain)
    if fold_work.exists():
        shutil.rmtree(fold_work)
    fold_work.mkdir(parents=True)
    model_root = fold_artifacts / "models"
    validation_prediction_root = fold_artifacts / "validation_predictions"
    test_prediction_path = fold_artifacts / "test_predictions.jsonl.gz"

    try:
        train_identity = materialize_domain_role(
            input_path=input_root / "train.jsonl.gz",
            output_path=fold_work / "train.jsonl.gz",
            partition="train",
            role="train",
            held_out_domain=domain,
            expected=fold_spec["roles"]["train"],
        )
        validation_identity = materialize_domain_role(
            input_path=input_root / "validation.jsonl.gz",
            output_path=fold_work / "validation.jsonl.gz",
            partition="validation",
            role="validation",
            held_out_domain=domain,
            expected=fold_spec["roles"]["validation"],
        )
        training = train_baselines(
            train_path=fold_work / "train.jsonl.gz",
            artifact_root=model_root,
            dataset_id=dataset_id,
            revision=revision,
            input_sha256=train_identity["sha256"],
            tfidf_config=tfidf_config,
            logistic_config=logistic_config,
        )
        training["input"]["selection"] = train_identity
        verify_baselines(training, model_root)
        validation = evaluate_baselines(
            evaluation_path=fold_work / "validation.jsonl.gz",
            evaluation_partition="validation",
            evaluation_identity=validation_identity,
            training_report=training,
            artifact_root=model_root,
            prediction_root=validation_prediction_root,
        )
        verify_evaluation(validation, validation_prediction_root)
        validation_word = _word_validation(validation)
        base_model = _word_model(training)
        calibration_path = model_root / "calibration_policy.joblib"
        calibration = fit_calibration_policy(
            prediction_path=(
                validation_prediction_root / validation_word["predictions"]["relative_path"]
            ),
            prediction_identity=validation_word["predictions"],
            base_model_identity=base_model,
            artifact_path=calibration_path,
            dataset_id=dataset_id,
            revision=revision,
        )
        verify_calibration_policy(
            calibration,
            validation_prediction_root / validation_word["predictions"]["relative_path"],
            calibration_path,
        )

        # Test text is not materialized or read by a model until training,
        # calibration-method selection, and threshold selection are complete.
        test_identity = materialize_domain_role(
            input_path=input_root / "test.jsonl.gz",
            output_path=fold_work / "test.jsonl.gz",
            partition="test",
            role="test",
            held_out_domain=domain,
            expected=fold_spec["roles"]["test"],
        )
        test = evaluate_frozen_test(
            test_path=fold_work / "test.jsonl.gz",
            test_identity=test_identity,
            base_model_path=model_root / base_model["relative_path"],
            base_model_identity=base_model,
            calibration_path=calibration_path,
            calibration_identity=calibration["artifact"],
            prediction_path=test_prediction_path,
            dataset_id=dataset_id,
            revision=revision,
        )
        verify_frozen_test(test, test_prediction_path, calibration_path)
        _mark_verified(training, validation, calibration, test)
        return {
            "regime_id": fold_spec["regime_id"],
            "held_out_domain": domain,
            "artifact_subdirectory": domain,
            "configuration": {
                "train_selector": "source domain != held_out_domain",
                "validation_selector": "source domain != held_out_domain",
                "test_selector": "source domain == held_out_domain",
                "test_materialized_after_policy_frozen": True,
                "test_retuning_allowed": False,
            },
            "training": training,
            "validation_evaluation": validation,
            "calibration": calibration,
            "test_evaluation": test,
            "validation": {
                "status": "pass",
                "selector_counts_match_regime_report": True,
                "artifacts_and_predictions_verified": True,
            },
        }
    finally:
        if fold_work.exists():
            shutil.rmtree(fold_work)


def _summary(folds: list[dict[str, Any]]) -> dict[str, Any]:
    metric_paths = {
        "roc_auc": ("raw_score", "overall", "roc_auc"),
        "average_precision": ("raw_score", "overall", "average_precision"),
        "brier_score": ("calibrated_probability", "brier_score"),
        "expected_calibration_error": (
            "calibrated_probability",
            "expected_calibration_error",
        ),
        "coverage": ("policy", "coverage"),
        "uncertain_rate": ("policy", "uncertain_rate"),
        "human_false_machine_rate": ("policy", "human_false_machine_rate"),
        "machine_false_human_rate": ("policy", "machine_false_human_rate"),
    }
    summary: dict[str, Any] = {}
    for name, path in metric_paths.items():
        values = []
        by_domain = {}
        for fold in folds:
            value: Any = fold["test_evaluation"]["metrics"]
            for key in path:
                value = value[key]
            value = float(value)
            values.append(value)
            by_domain[fold["held_out_domain"]] = value
        summary[name] = {
            "minimum": round(min(values), 6),
            "median": round(median(values), 6),
            "maximum": round(max(values), 6),
            "by_held_out_domain": by_domain,
        }
    return summary


def run_domain_holdouts(
    *,
    regime_report: dict[str, Any],
    input_root: Path,
    artifact_root: Path,
    work_root: Path,
    progress: Callable[[str], None] | None = None,
    tfidf_config: dict[str, Any] | None = None,
    logistic_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train, calibrate, and evaluate every declared leave-one-domain-out fold."""
    folds = []
    for fold_spec in regime_report["domain_holdouts"]:
        domain = fold_spec["held_out_domain"]
        if progress is not None:
            progress(f"starting domain holdout: {domain}")
        fold = _run_fold(
            fold_spec=fold_spec,
            input_root=input_root,
            artifact_root=artifact_root,
            work_root=work_root,
            dataset_id=regime_report["dataset_id"],
            revision=regime_report["revision"],
            tfidf_config=tfidf_config,
            logistic_config=logistic_config,
        )
        folds.append(fold)
        if progress is not None:
            auc = fold["test_evaluation"]["metrics"]["raw_score"]["overall"]["roc_auc"]
            progress(f"completed domain holdout: {domain} (ROC AUC {auc:.6f})")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "domain_holdout_version": DOMAIN_HOLDOUT_VERSION,
        "dataset_id": regime_report["dataset_id"],
        "revision": regime_report["revision"],
        "configuration": {
            "regime_report": "mage_source_regimes.json",
            "candidate_model": "word_tfidf_logistic",
            "controls_also_fitted": ["majority", "length_logistic"],
            "calibration_cycle_per_fold": True,
            "metadata_features_used": [],
            "test_data_used_for_training_calibration_or_thresholds": False,
            "test_retuning_allowed": False,
            "selected_inputs_removed_after_each_fold": True,
        },
        "folds": folds,
        "summary": _summary(folds),
        "validation": {
            "status": "pass",
            "folds_expected": len(regime_report["domain_holdouts"]),
            "folds_completed": len(folds),
            "all_selector_counts_match": True,
            "all_artifacts_and_predictions_verified": True,
        },
    }


def verify_domain_holdouts(report: dict[str, Any], artifact_root: Path) -> None:
    """Verify all ignored fold artifacts and recompute stored prediction metrics."""
    folds = report.get("folds")
    if not isinstance(folds, list) or not folds:
        raise DomainHoldoutError("Domain-holdout report has no folds")
    domains = [fold.get("held_out_domain") for fold in folds]
    if len(set(domains)) != len(domains):
        raise DomainHoldoutError("Domain-holdout report has duplicate domains")
    for fold in folds:
        domain = fold["held_out_domain"]
        fold_root = _safe_child(artifact_root, domain)
        model_root = fold_root / "models"
        prediction_root = fold_root / "validation_predictions"
        training = fold["training"]
        validation = fold["validation_evaluation"]
        calibration = fold["calibration"]
        test = fold["test_evaluation"]
        verify_baselines(training, model_root)
        verify_evaluation(validation, prediction_root)
        word = _word_validation(validation)
        calibration_path = model_root / calibration["artifact"]["relative_path"]
        verify_calibration_policy(
            calibration,
            prediction_root / word["predictions"]["relative_path"],
            calibration_path,
        )
        verify_frozen_test(test, fold_root / "test_predictions.jsonl.gz", calibration_path)
        if fold.get("validation", {}).get("status") != "pass":
            raise DomainHoldoutError(f"{domain} fold validation status is not pass")
    if _summary(folds) != report.get("summary"):
        raise DomainHoldoutError("Domain-holdout summary does not match fold metrics")
    validation = report.get("validation", {})
    if validation.get("folds_completed") != len(folds):
        raise DomainHoldoutError("Domain-holdout fold count does not reconcile")


def render_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


__all__ = [
    "BaselineError",
    "CalibrationError",
    "DomainHoldoutError",
    "EvaluationError",
    "FrozenTestError",
    "materialize_domain_role",
    "render_report",
    "run_domain_holdouts",
    "verify_domain_holdouts",
]
