"""One-time frozen evaluation on the overlap-gated Ghostbuster corpus."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from authentitext.data.cleaning import sha256_file
from authentitext.modeling.frozen_test import (
    FrozenTestError,
    evaluate_frozen_test,
    verify_frozen_test,
)

EXTERNAL_EVALUATION_VERSION = 1


class ExternalEvaluationError(RuntimeError):
    """Raised when sealed external scoring cannot be performed or verified."""


def _exclusion_ids(overlap_report: dict[str, Any]) -> set[str]:
    if overlap_report.get("validation", {}).get("status") != "pass":
        raise ExternalEvaluationError("External overlap gate status is not pass")
    validation = overlap_report["validation"]
    if validation.get("model_scoring_performed") is not False:
        raise ExternalEvaluationError("Overlap gate must precede model scoring")
    if validation.get("outcome_data_used_to_define_exclusions") is not False:
        raise ExternalEvaluationError("Overlap exclusions must be outcome-blind")
    if validation.get("all_confirmed_overlap_resolved_by_external_exclusion") is not True:
        raise ExternalEvaluationError("Overlap gate findings are not resolved")
    records = overlap_report.get("exclusions", {}).get("records")
    if not isinstance(records, list):
        raise ExternalEvaluationError("Overlap report has no exclusion records")
    identifiers = []
    for record in records:
        record_id = record.get("external_record_id") if isinstance(record, dict) else None
        reasons = record.get("reasons") if isinstance(record, dict) else None
        if not isinstance(record_id, str) or not record_id or not isinstance(reasons, list):
            raise ExternalEvaluationError("Overlap report has an invalid exclusion")
        identifiers.append(record_id)
    if len(identifiers) != len(set(identifiers)):
        raise ExternalEvaluationError("Overlap report repeats an exclusion ID")
    if len(identifiers) != overlap_report["exclusions"]["external_records"]:
        raise ExternalEvaluationError("Overlap exclusion count does not reconcile")
    return set(identifiers)


def _record_ids(path: Path, *, prediction: bool) -> set[str]:
    record_ids = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ExternalEvaluationError(f"{path}:{line_number} is invalid JSON") from error
            if not isinstance(record, dict):
                raise ExternalEvaluationError(f"{path}:{line_number} is not an object")
            record_id = record.get("record_id")
            if not isinstance(record_id, str) or record_id in record_ids:
                raise ExternalEvaluationError(f"{path}:{line_number} has an invalid record ID")
            if not prediction and record.get("partition") != "test":
                raise ExternalEvaluationError(f"{path}:{line_number} is not external test data")
            record_ids.add(record_id)
    return record_ids


def evaluate_ghostbuster_external(
    *,
    test_path: Path,
    test_identity: dict[str, Any],
    overlap_report: dict[str, Any],
    overlap_identity: dict[str, Any],
    base_model_path: Path,
    base_model_identity: dict[str, Any],
    calibration_path: Path,
    calibration_identity: dict[str, Any],
    prediction_path: Path,
    dataset_id: str,
    revision: str,
) -> dict[str, Any]:
    """Apply the frozen policy once after enforcing the overlap gate."""
    external_identity = overlap_report.get("datasets", {}).get("external", {})
    if external_identity.get("sha256") != test_identity.get("sha256"):
        raise ExternalEvaluationError("Overlap gate external hash does not match test input")
    if external_identity.get("rows") != test_identity.get("rows"):
        raise ExternalEvaluationError("Overlap gate external rows do not match test input")
    excluded_record_ids = _exclusion_ids(overlap_report)
    report = evaluate_frozen_test(
        test_path=test_path,
        test_identity=test_identity,
        base_model_path=base_model_path,
        base_model_identity=base_model_identity,
        calibration_path=calibration_path,
        calibration_identity=calibration_identity,
        prediction_path=prediction_path,
        dataset_id=dataset_id,
        revision=revision,
        excluded_record_ids=excluded_record_ids,
    )
    expected_scored_rows = overlap_report["exclusions"]["external_rows_after_exclusion"]
    if report["predictions"]["rows"] != expected_scored_rows:
        raise ExternalEvaluationError("Scored rows do not match the overlap gate")
    report["external_evaluation_version"] = EXTERNAL_EVALUATION_VERSION
    report["overlap_gate"] = {
        **overlap_identity,
        "cross_dataset_exact_pairs": overlap_report["cross_dataset_exact"]["pairs"],
        "cross_dataset_normalized_pairs": overlap_report["cross_dataset_normalized"]["pairs"],
        "cross_dataset_confirmed_near_pairs": overlap_report["cross_dataset_near"][
            "confirmed_pairs"
        ],
        "external_records_excluded": len(excluded_record_ids),
        "external_rows_scored": expected_scored_rows,
    }
    report["configuration"].update(
        {
            "one_time_external_scoring": True,
            "external_results_may_not_change_model_calibration_or_thresholds": True,
            "exclusion_policy": "stable IDs declared by the outcome-blind overlap gate",
        }
    )
    report["validation"].update(
        {
            "overlap_gate_verified": True,
            "external_exclusions_applied": True,
            "external_outcomes_used_for_retuning": False,
        }
    )
    return report


def verify_ghostbuster_external(
    report: dict[str, Any],
    *,
    prediction_path: Path,
    calibration_path: Path,
    external_path: Path,
    overlap_report: dict[str, Any],
    overlap_path: Path,
) -> None:
    """Recompute metrics and prove the exact external record selection."""
    try:
        verify_frozen_test(report, prediction_path, calibration_path)
    except FrozenTestError as error:
        raise ExternalEvaluationError(str(error)) from error
    overlap_identity = report.get("overlap_gate", {})
    if overlap_path.stat().st_size != overlap_identity.get("bytes"):
        raise ExternalEvaluationError("Overlap report size does not match evaluation")
    if sha256_file(overlap_path) != overlap_identity.get("sha256"):
        raise ExternalEvaluationError("Overlap report hash does not match evaluation")
    exclusions = _exclusion_ids(overlap_report)
    external_ids = _record_ids(external_path, prediction=False)
    if not exclusions <= external_ids:
        raise ExternalEvaluationError("An overlap exclusion is absent from external data")
    prediction_ids = _record_ids(prediction_path, prediction=True)
    if prediction_ids != external_ids - exclusions:
        raise ExternalEvaluationError("Predictions do not match overlap-gated external IDs")
    selection = report.get("selection", {})
    expected_fingerprint = hashlib.sha256("\n".join(sorted(exclusions)).encode("utf-8")).hexdigest()
    if selection.get("excluded_record_ids_sha256") != expected_fingerprint:
        raise ExternalEvaluationError("External exclusion fingerprint does not match")
    if selection.get("input_rows") != len(external_ids):
        raise ExternalEvaluationError("External selection input rows do not match")
    if selection.get("excluded_rows") != len(exclusions):
        raise ExternalEvaluationError("External selection exclusion rows do not match")
    if selection.get("scored_rows") != len(prediction_ids):
        raise ExternalEvaluationError("External selection scored rows do not match")
