"""Fit and verify validation-only calibration and abstention policy."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from authentitext.data.regimes import parse_source_id
from authentitext.modeling.baselines import sha256_file

CALIBRATION_VERSION = 1
REPORT_SCHEMA_VERSION = 1
RANDOM_SEED = 1729
CALIBRATION_BINS = 15
MACHINE_FALSE_POSITIVE_GOAL = 0.05
HUMAN_FALSE_NEGATIVE_GOAL = 0.05


class CalibrationError(RuntimeError):
    """Raised when calibration or threshold selection is invalid."""


def _validation_bucket(record_id: str) -> int:
    digest = hashlib.sha256(record_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10


def _split_role(record_id: str) -> str:
    bucket = _validation_bucket(record_id)
    if bucket <= 3:
        return "calibration_fit"
    if bucket <= 6:
        return "policy_selection"
    return "calibration_audit"


def _read_scores(path: Path) -> dict[str, Any]:
    record_ids = []
    scores = []
    targets = []
    sources = []
    seen_ids = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise CalibrationError(f"{path}:{line_number} is not valid JSON") from error
            record_id = record.get("record_id")
            score = record.get("score")
            target = record.get("target")
            source = record.get("source")
            if (
                not isinstance(record_id, str)
                or record_id in seen_ids
                or not isinstance(score, int | float)
                or target not in (0, 1)
                or not isinstance(source, str)
            ):
                raise CalibrationError(f"{path}:{line_number} has invalid calibration fields")
            seen_ids.add(record_id)
            record_ids.append(record_id)
            scores.append(score)
            targets.append(target)
            sources.append(source)
    score_array = np.asarray(scores, dtype=np.float64)
    if (
        not len(score_array)
        or not np.isfinite(score_array).all()
        or ((score_array < 0) | (score_array > 1)).any()
        or set(targets) != {0, 1}
    ):
        raise CalibrationError("Prediction scores must be finite probabilities with both targets")
    return {
        "record_ids": record_ids,
        "scores": score_array,
        "targets": np.asarray(targets, dtype=np.int8),
        "sources": sources,
    }


def _role_masks(record_ids: list[str]) -> dict[str, np.ndarray]:
    roles = np.asarray([_split_role(record_id) for record_id in record_ids], dtype=object)
    masks = {
        role: roles == role for role in ("calibration_fit", "policy_selection", "calibration_audit")
    }
    if any(mask.sum() == 0 for mask in masks.values()):
        raise CalibrationError("A deterministic validation role is empty")
    return masks


def _logits(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def _fit_methods(scores: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    if set(targets.tolist()) != {0, 1}:
        raise CalibrationError("Calibration-fit role must contain both targets")
    sigmoid = LogisticRegression(
        C=1_000_000.0,
        max_iter=1000,
        random_state=RANDOM_SEED,
        solver="lbfgs",
        tol=1e-8,
    ).fit(_logits(scores), targets)
    isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(scores, targets)
    return {"raw": None, "sigmoid": sigmoid, "isotonic": isotonic}


def calibrate_scores(method: str, calibrator: Any, scores: np.ndarray) -> np.ndarray:
    """Map raw scores to calibrated probabilities using the declared method."""
    if method == "raw":
        calibrated = scores
    elif method == "sigmoid":
        calibrated = calibrator.predict_proba(_logits(scores))[:, 1]
    elif method == "isotonic":
        calibrated = calibrator.predict(scores)
    else:
        raise CalibrationError(f"Unsupported calibration method: {method!r}")
    calibrated = np.asarray(calibrated, dtype=np.float64)
    if not np.isfinite(calibrated).all() or ((calibrated < 0) | (calibrated > 1)).any():
        raise CalibrationError(f"{method} calibration returned invalid probabilities")
    return calibrated


def calibration_metrics(targets: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    indexes = np.minimum((scores * CALIBRATION_BINS).astype(int), CALIBRATION_BINS - 1)
    ece = 0.0
    nonempty_bins = 0
    for index in range(CALIBRATION_BINS):
        mask = indexes == index
        count = int(mask.sum())
        if count:
            nonempty_bins += 1
            ece += count / len(targets) * abs(float(scores[mask].mean() - targets[mask].mean()))
    return {
        "rows": len(targets),
        "brier_score": round(float(brier_score_loss(targets, scores)), 6),
        "log_loss": round(float(log_loss(targets, scores, labels=[0, 1])), 6),
        "expected_calibration_error": round(ece, 6),
        "equal_width_bins": CALIBRATION_BINS,
        "nonempty_bins": nonempty_bins,
    }


def _select_method(
    methods: dict[str, Any],
    scores: np.ndarray,
    targets: np.ndarray,
) -> tuple[str, list[dict[str, Any]]]:
    comparisons = []
    for method in sorted(methods):
        calibrated = calibrate_scores(method, methods[method], scores)
        metrics = calibration_metrics(targets, calibrated)
        comparisons.append({"method": method, "metrics": metrics})
    selected = min(
        comparisons,
        key=lambda item: (
            item["metrics"]["brier_score"],
            item["metrics"]["expected_calibration_error"],
            item["method"],
        ),
    )["method"]
    return selected, comparisons


def _machine_threshold(human_scores: np.ndarray, goal: float) -> float:
    allowed = math.floor(goal * len(human_scores))
    ordered = np.sort(human_scores)[::-1]
    if allowed == 0:
        return 1.0
    boundary = ordered[allowed - 1]
    if int((human_scores >= boundary).sum()) > allowed:
        boundary = np.nextafter(boundary, 1.0)
    return float(min(1.0, boundary))


def _human_threshold(machine_scores: np.ndarray, goal: float) -> float:
    allowed = math.floor(goal * len(machine_scores))
    ordered = np.sort(machine_scores)
    if allowed == 0:
        return 0.0
    boundary = ordered[allowed - 1]
    if int((machine_scores <= boundary).sum()) > allowed:
        boundary = np.nextafter(boundary, 0.0)
    return float(max(0.0, boundary))


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials == 0:
        return [0.0, 0.0]
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return [round(max(0.0, centre - margin), 6), round(min(1.0, centre + margin), 6)]


def policy_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    human_threshold: float,
    machine_threshold: float,
) -> dict[str, Any]:
    categories = np.full(len(scores), "uncertain", dtype=object)
    categories[scores <= human_threshold] = "likely_human"
    categories[scores >= machine_threshold] = "likely_machine"
    human_mask = targets == 0
    machine_mask = targets == 1
    false_machine = int(((categories == "likely_machine") & human_mask).sum())
    false_human = int(((categories == "likely_human") & machine_mask).sum())
    likely_machine = int((categories == "likely_machine").sum())
    likely_human = int((categories == "likely_human").sum())
    uncertain = int((categories == "uncertain").sum())
    correct_machine = int(((categories == "likely_machine") & machine_mask).sum())
    correct_human = int(((categories == "likely_human") & human_mask).sum())
    human_rows = int(human_mask.sum())
    machine_rows = int(machine_mask.sum())
    decisive = likely_machine + likely_human
    return {
        "rows": len(targets),
        "target_counts": {"0": human_rows, "1": machine_rows},
        "category_counts": {
            "likely_human": likely_human,
            "uncertain": uncertain,
            "likely_machine": likely_machine,
        },
        "coverage": round(decisive / len(targets), 6),
        "uncertain_rate": round(uncertain / len(targets), 6),
        "human_false_machine_count": false_machine,
        "human_false_machine_rate": (round(false_machine / human_rows, 6) if human_rows else None),
        "human_false_machine_wilson_95": _wilson_interval(false_machine, human_rows),
        "machine_false_human_count": false_human,
        "machine_false_human_rate": (
            round(false_human / machine_rows, 6) if machine_rows else None
        ),
        "machine_false_human_wilson_95": _wilson_interval(false_human, machine_rows),
        "likely_machine_precision": round(
            correct_machine / likely_machine if likely_machine else 0.0, 6
        ),
        "likely_human_negative_predictive_value": round(
            correct_human / likely_human if likely_human else 0.0, 6
        ),
        "decisive_accuracy": round(
            (correct_machine + correct_human) / decisive if decisive else 0.0, 6
        ),
    }


def domain_policy_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    sources: list[str],
    mask: np.ndarray,
    human_threshold: float,
    machine_threshold: float,
) -> dict[str, Any]:
    masked_sources = np.asarray(sources, dtype=object)[mask]
    domains = np.asarray(
        [parse_source_id(source).domain for source in masked_sources], dtype=object
    )
    masked_targets = targets[mask]
    masked_scores = scores[mask]
    return {
        domain: policy_metrics(
            masked_targets[domains == domain],
            masked_scores[domains == domain],
            human_threshold,
            machine_threshold,
        )
        for domain in sorted(set(domains))
    }


def _role_summary(targets: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        role: {
            "rows": int(mask.sum()),
            "target_counts": {
                "0": int(((targets == 0) & mask).sum()),
                "1": int(((targets == 1) & mask).sum()),
            },
        }
        for role, mask in masks.items()
    }


def _write_artifact(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        joblib.dump(payload, temporary, compress=3)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def fit_calibration_policy(
    *,
    prediction_path: Path,
    prediction_identity: dict[str, Any],
    base_model_identity: dict[str, Any],
    artifact_path: Path,
    dataset_id: str,
    revision: str,
) -> dict[str, Any]:
    """Select a calibrator and thresholds with three deterministic validation roles."""
    data = _read_scores(prediction_path)
    record_ids = data["record_ids"]
    raw_scores = data["scores"]
    targets = data["targets"]
    sources = data["sources"]
    masks = _role_masks(record_ids)
    for role, mask in masks.items():
        if set(targets[mask].tolist()) != {0, 1}:
            raise CalibrationError(f"{role} must contain both targets")

    fit_mask = masks["calibration_fit"]
    selection_mask = masks["policy_selection"]
    audit_mask = masks["calibration_audit"]
    methods = _fit_methods(raw_scores[fit_mask], targets[fit_mask])
    selected_method, selection_comparison = _select_method(
        methods, raw_scores[selection_mask], targets[selection_mask]
    )
    selected_calibrator = methods[selected_method]
    selection_scores = calibrate_scores(
        selected_method, selected_calibrator, raw_scores[selection_mask]
    )
    selection_targets = targets[selection_mask]
    machine_threshold = _machine_threshold(
        selection_scores[selection_targets == 0], MACHINE_FALSE_POSITIVE_GOAL
    )
    human_threshold = _human_threshold(
        selection_scores[selection_targets == 1], HUMAN_FALSE_NEGATIVE_GOAL
    )
    if human_threshold >= machine_threshold:
        raise CalibrationError("Selected thresholds do not leave an uncertain interval")

    audit_raw_metrics = calibration_metrics(targets[audit_mask], raw_scores[audit_mask])
    audit_scores = calibrate_scores(selected_method, selected_calibrator, raw_scores[audit_mask])
    audit_calibrated_metrics = calibration_metrics(targets[audit_mask], audit_scores)
    selection_policy = policy_metrics(
        selection_targets, selection_scores, human_threshold, machine_threshold
    )
    audit_policy = policy_metrics(
        targets[audit_mask], audit_scores, human_threshold, machine_threshold
    )

    artifact_payload = {
        "artifact_version": CALIBRATION_VERSION,
        "model_type": "calibration_policy",
        "base_model_type": "word_tfidf_logistic",
        "base_model_sha256": base_model_identity["sha256"],
        "calibration_method": selected_method,
        "calibrator": selected_calibrator,
        "human_threshold": human_threshold,
        "machine_threshold": machine_threshold,
    }
    _write_artifact(artifact_payload, artifact_path)
    artifact_identity = {
        "relative_path": artifact_path.name,
        "bytes": artifact_path.stat().st_size,
        "sha256": sha256_file(artifact_path),
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "dataset_id": dataset_id,
        "revision": revision,
        "base_model": base_model_identity,
        "input_predictions": prediction_identity,
        "configuration": {
            "validation_role_assignment": "sha256(record_id) first 64 bits modulo 10",
            "calibration_fit_buckets": [0, 1, 2, 3],
            "policy_selection_buckets": [4, 5, 6],
            "calibration_audit_buckets": [7, 8, 9],
            "candidate_methods": ["raw", "sigmoid", "isotonic"],
            "selection_metric": "lowest policy-selection Brier; then ECE; then method name",
            "machine_false_positive_goal": MACHINE_FALSE_POSITIVE_GOAL,
            "human_false_negative_goal": HUMAN_FALSE_NEGATIVE_GOAL,
            "threshold_inclusivity": {
                "likely_human": "score <= human_threshold",
                "likely_machine": "score >= machine_threshold",
                "uncertain": "strictly between thresholds",
            },
            "test_data_used": False,
        },
        "roles": _role_summary(targets, masks),
        "method_selection": {
            "selected_method": selected_method,
            "policy_selection_comparison": selection_comparison,
        },
        "thresholds": {
            "human_threshold": round(human_threshold, 12),
            "machine_threshold": round(machine_threshold, 12),
        },
        "policy_selection_metrics": selection_policy,
        "calibration_audit": {
            "raw_metrics": audit_raw_metrics,
            "selected_calibrated_metrics": audit_calibrated_metrics,
            "policy_metrics": audit_policy,
            "policy_by_domain": domain_policy_metrics(
                targets,
                calibrate_scores(selected_method, selected_calibrator, raw_scores),
                sources,
                audit_mask,
                human_threshold,
                machine_threshold,
            ),
        },
        "artifact": artifact_identity,
        "validation": {
            "status": "pass",
            "artifact_reload_and_score": False,
            "source_prediction_verified": False,
            "audit_metrics_recomputed": False,
        },
    }


def verify_calibration_policy(
    report: dict[str, Any], prediction_path: Path, artifact_path: Path
) -> None:
    """Verify artifact linkage and recompute audit evidence from source predictions."""
    prediction_identity = report["input_predictions"]
    if prediction_path.stat().st_size != prediction_identity["bytes"]:
        raise CalibrationError("Calibration source prediction size mismatch")
    if sha256_file(prediction_path) != prediction_identity["sha256"]:
        raise CalibrationError("Calibration source prediction SHA-256 mismatch")
    artifact_identity = report["artifact"]
    if artifact_path.stat().st_size != artifact_identity["bytes"]:
        raise CalibrationError("Calibration artifact size mismatch")
    if sha256_file(artifact_path) != artifact_identity["sha256"]:
        raise CalibrationError("Calibration artifact SHA-256 mismatch")
    payload = joblib.load(artifact_path)
    if payload.get("model_type") != "calibration_policy":
        raise CalibrationError("Calibration artifact type mismatch")
    if payload.get("base_model_sha256") != report["base_model"]["sha256"]:
        raise CalibrationError("Calibration artifact base model mismatch")
    human_threshold = payload["human_threshold"]
    machine_threshold = payload["machine_threshold"]
    if not 0 <= human_threshold < machine_threshold <= 1:
        raise CalibrationError("Calibration artifact thresholds are invalid")

    data = _read_scores(prediction_path)
    masks = _role_masks(data["record_ids"])
    audit_mask = masks["calibration_audit"]
    calibrated = calibrate_scores(
        payload["calibration_method"], payload["calibrator"], data["scores"]
    )
    audit_metrics = calibration_metrics(data["targets"][audit_mask], calibrated[audit_mask])
    audit_policy_metrics = policy_metrics(
        data["targets"][audit_mask],
        calibrated[audit_mask],
        human_threshold,
        machine_threshold,
    )
    by_domain = domain_policy_metrics(
        data["targets"],
        calibrated,
        data["sources"],
        audit_mask,
        human_threshold,
        machine_threshold,
    )
    if audit_metrics != report["calibration_audit"]["selected_calibrated_metrics"]:
        raise CalibrationError("Calibration audit metrics do not match report")
    if audit_policy_metrics != report["calibration_audit"]["policy_metrics"]:
        raise CalibrationError("Calibration audit policy metrics do not match report")
    if by_domain != report["calibration_audit"]["policy_by_domain"]:
        raise CalibrationError("Calibration audit domain metrics do not match report")


def render_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
