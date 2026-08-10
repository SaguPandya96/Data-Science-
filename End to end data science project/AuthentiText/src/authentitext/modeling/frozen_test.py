"""Evaluate the frozen calibrated baseline on the published test partition."""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from authentitext.data.cleaning import open_deterministic_gzip
from authentitext.data.regimes import parse_source_id
from authentitext.modeling.baselines import positive_scores, sha256_file
from authentitext.modeling.calibration import (
    calibrate_scores,
    calibration_metrics,
    domain_policy_metrics,
    policy_metrics,
)
from authentitext.modeling.evaluation import evaluate_scores, load_evaluation_records

FROZEN_TEST_VERSION = 1
REPORT_SCHEMA_VERSION = 1


class FrozenTestError(RuntimeError):
    """Raised when the frozen test evaluation cannot be verified."""


def _categories(scores: np.ndarray, human_threshold: float, machine_threshold: float) -> np.ndarray:
    categories = np.full(len(scores), "uncertain", dtype=object)
    categories[scores <= human_threshold] = "likely_human"
    categories[scores >= machine_threshold] = "likely_machine"
    return categories


def _policy_by_length(
    targets: np.ndarray,
    scores: np.ndarray,
    whitespace_tokens: np.ndarray,
    human_threshold: float,
    machine_threshold: float,
) -> dict[str, Any]:
    bands = {
        "under_50": whitespace_tokens < 50,
        "50_to_128": (whitespace_tokens >= 50) & (whitespace_tokens <= 128),
        "129_to_512": (whitespace_tokens >= 129) & (whitespace_tokens <= 512),
        "over_512": whitespace_tokens > 512,
    }
    return {
        name: policy_metrics(targets[mask], scores[mask], human_threshold, machine_threshold)
        for name, mask in bands.items()
        if mask.any()
    }


def _machine_group_outcomes(
    targets: np.ndarray,
    scores: np.ndarray,
    sources: list[str],
    human_threshold: float,
    machine_threshold: float,
    attribute: str,
) -> dict[str, Any]:
    categories = _categories(scores, human_threshold, machine_threshold)
    groups: dict[str, Counter[str]] = {}
    for target, source, category in zip(targets, sources, categories, strict=True):
        if target != 1:
            continue
        identity = parse_source_id(source)
        value = getattr(identity, attribute)
        if value is None:
            continue
        groups.setdefault(value, Counter())[str(category)] += 1
    return {
        group: {
            "rows": sum(counts.values()),
            "likely_human": counts["likely_human"],
            "uncertain": counts["uncertain"],
            "likely_machine": counts["likely_machine"],
            "likely_machine_rate": round(counts["likely_machine"] / sum(counts.values()), 6),
        }
        for group, counts in sorted(groups.items())
    }


def _metrics(
    *,
    targets: np.ndarray,
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray,
    sources: list[str],
    whitespace_tokens: np.ndarray,
    human_threshold: float,
    machine_threshold: float,
) -> dict[str, Any]:
    all_mask = np.ones(len(targets), dtype=bool)
    return {
        "raw_score": evaluate_scores(targets, raw_scores, sources, whitespace_tokens),
        "calibrated_probability": calibration_metrics(targets, calibrated_scores),
        "policy": policy_metrics(targets, calibrated_scores, human_threshold, machine_threshold),
        "policy_by_domain": domain_policy_metrics(
            targets,
            calibrated_scores,
            sources,
            all_mask,
            human_threshold,
            machine_threshold,
        ),
        "policy_by_whitespace_token_band": _policy_by_length(
            targets,
            calibrated_scores,
            whitespace_tokens,
            human_threshold,
            machine_threshold,
        ),
        "machine_outcomes_by_generator": _machine_group_outcomes(
            targets,
            calibrated_scores,
            sources,
            human_threshold,
            machine_threshold,
            "generator",
        ),
        "machine_outcomes_by_strategy": _machine_group_outcomes(
            targets,
            calibrated_scores,
            sources,
            human_threshold,
            machine_threshold,
            "strategy",
        ),
    }


def _write_predictions(
    *,
    path: Path,
    records,
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray,
    categories: np.ndarray,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with open_deterministic_gzip(temporary) as handle:
            for record, raw_score, calibrated_score, category in zip(
                records, raw_scores, calibrated_scores, categories, strict=True
            ):
                handle.write(
                    json.dumps(
                        {
                            "calibrated_score": float(calibrated_score),
                            "category": str(category),
                            "raw_score": float(raw_score),
                            "record_id": record.record_id,
                            "source": record.source,
                            "target": record.target,
                            "whitespace_tokens": record.whitespace_tokens,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "relative_path": path.name,
        "rows": len(records),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def evaluate_frozen_test(
    *,
    test_path: Path,
    test_identity: dict[str, Any],
    base_model_path: Path,
    base_model_identity: dict[str, Any],
    calibration_path: Path,
    calibration_identity: dict[str, Any],
    prediction_path: Path,
    dataset_id: str,
    revision: str,
    excluded_record_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Score test once with frozen artifacts and persist a text-free audit trail."""
    all_records = load_evaluation_records(test_path, "test")
    if "rows" in test_identity and test_identity["rows"] != len(all_records):
        raise FrozenTestError("Frozen test input row count does not match its identity")
    excluded_record_ids = set(excluded_record_ids or ())
    observed_record_ids = {record.record_id for record in all_records}
    unknown_exclusions = excluded_record_ids - observed_record_ids
    if unknown_exclusions:
        raise FrozenTestError("Frozen test exclusions reference unknown record IDs")
    records = [record for record in all_records if record.record_id not in excluded_record_ids]
    if not records or {record.target for record in records} != {0, 1}:
        raise FrozenTestError("Selected frozen test data must contain both targets")
    texts = [record.text for record in records]
    targets = np.asarray([record.target for record in records], dtype=np.int8)
    sources = [record.source for record in records]
    whitespace_tokens = np.asarray([record.whitespace_tokens for record in records], dtype=np.int32)
    base_model = joblib.load(base_model_path)
    calibration = joblib.load(calibration_path)
    if base_model.get("model_type") != "word_tfidf_logistic":
        raise FrozenTestError("Frozen base artifact has the wrong model type")
    if calibration.get("base_model_sha256") != base_model_identity["sha256"]:
        raise FrozenTestError("Calibration artifact is not linked to the frozen base model")
    scoring_started = time.perf_counter()
    raw_scores = np.round(np.asarray(positive_scores(base_model, texts), dtype=np.float64), 12)
    calibrated_scores = np.round(
        calibrate_scores(calibration["calibration_method"], calibration["calibrator"], raw_scores),
        12,
    )
    categories = _categories(
        calibrated_scores,
        calibration["human_threshold"],
        calibration["machine_threshold"],
    )
    scoring_seconds = time.perf_counter() - scoring_started
    prediction_identity = _write_predictions(
        path=prediction_path,
        records=records,
        raw_scores=raw_scores,
        calibrated_scores=calibrated_scores,
        categories=categories,
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "frozen_test_version": FROZEN_TEST_VERSION,
        "dataset_id": dataset_id,
        "revision": revision,
        "partition": "test",
        "input": test_identity,
        "base_model": base_model_identity,
        "calibration_artifact": calibration_identity,
        "configuration": {
            "calibration_method": calibration["calibration_method"],
            "human_threshold": calibration["human_threshold"],
            "machine_threshold": calibration["machine_threshold"],
            "score_rounding_decimal_places": 12,
            "artifacts_frozen_before_test": True,
            "retuning_after_test_allowed": False,
            "prediction_format": "deterministic gzip JSON Lines with no text",
        },
        "predictions": prediction_identity,
        "scoring": {
            "seconds": round(scoring_seconds, 3),
            "records_per_second": round(len(records) / scoring_seconds, 3),
            "batch_size": len(records),
        },
        "metrics": _metrics(
            targets=targets,
            raw_scores=raw_scores,
            calibrated_scores=calibrated_scores,
            sources=sources,
            whitespace_tokens=whitespace_tokens,
            human_threshold=calibration["human_threshold"],
            machine_threshold=calibration["machine_threshold"],
        ),
        "validation": {
            "status": "pass",
            "prediction_file_verified": False,
            "metrics_recomputed_from_predictions": False,
            "thresholds_match_calibration_artifact": True,
        },
    }
    if excluded_record_ids:
        report["selection"] = {
            "input_rows": len(all_records),
            "excluded_rows": len(excluded_record_ids),
            "scored_rows": len(records),
            "excluded_record_ids_sha256": hashlib.sha256(
                "\n".join(sorted(excluded_record_ids)).encode("utf-8")
            ).hexdigest(),
        }
    return report


def _read_predictions(path: Path) -> dict[str, Any]:
    raw_scores = []
    calibrated_scores = []
    categories = []
    targets = []
    sources = []
    whitespace_tokens = []
    record_ids = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise FrozenTestError(f"{path}:{line_number} is not valid JSON") from error
            record_id = record.get("record_id")
            if not isinstance(record_id, str) or record_id in record_ids:
                raise FrozenTestError(f"{path}:{line_number} has an invalid record ID")
            record_ids.add(record_id)
            raw_scores.append(record.get("raw_score"))
            calibrated_scores.append(record.get("calibrated_score"))
            categories.append(record.get("category"))
            targets.append(record.get("target"))
            sources.append(record.get("source"))
            whitespace_tokens.append(record.get("whitespace_tokens"))
    if (
        any(not isinstance(score, int | float) for score in raw_scores + calibrated_scores)
        or any(
            category not in {"likely_human", "uncertain", "likely_machine"}
            for category in categories
        )
        or any(target not in (0, 1) for target in targets)
        or any(not isinstance(source, str) for source in sources)
        or any(not isinstance(value, int) or value < 0 for value in whitespace_tokens)
    ):
        raise FrozenTestError(f"Prediction fields are invalid: {path}")
    return {
        "raw_scores": np.asarray(raw_scores, dtype=np.float64),
        "calibrated_scores": np.asarray(calibrated_scores, dtype=np.float64),
        "categories": np.asarray(categories, dtype=object),
        "targets": np.asarray(targets, dtype=np.int8),
        "sources": sources,
        "whitespace_tokens": np.asarray(whitespace_tokens, dtype=np.int32),
    }


def verify_frozen_test(
    report: dict[str, Any], prediction_path: Path, calibration_path: Path
) -> None:
    """Verify saved test predictions and recompute all frozen metrics."""
    identity = report["predictions"]
    if prediction_path.stat().st_size != identity["bytes"]:
        raise FrozenTestError("Frozen test prediction size mismatch")
    if sha256_file(prediction_path) != identity["sha256"]:
        raise FrozenTestError("Frozen test prediction SHA-256 mismatch")
    calibration = joblib.load(calibration_path)
    if calibration.get("base_model_sha256") != report["base_model"]["sha256"]:
        raise FrozenTestError("Frozen test calibration linkage mismatch")
    data = _read_predictions(prediction_path)
    if len(data["targets"]) != identity["rows"]:
        raise FrozenTestError("Frozen test prediction row count mismatch")
    recalibrated = np.round(
        calibrate_scores(
            calibration["calibration_method"],
            calibration["calibrator"],
            data["raw_scores"],
        ),
        12,
    )
    if not np.array_equal(recalibrated, data["calibrated_scores"]):
        raise FrozenTestError("Saved calibrated scores do not match artifact")
    expected_categories = _categories(
        recalibrated,
        calibration["human_threshold"],
        calibration["machine_threshold"],
    )
    if not np.array_equal(expected_categories, data["categories"]):
        raise FrozenTestError("Saved categories do not match frozen thresholds")
    recomputed = _metrics(
        targets=data["targets"],
        raw_scores=data["raw_scores"],
        calibrated_scores=data["calibrated_scores"],
        sources=data["sources"],
        whitespace_tokens=data["whitespace_tokens"],
        human_threshold=calibration["human_threshold"],
        machine_threshold=calibration["machine_threshold"],
    )
    if recomputed != report["metrics"]:
        raise FrozenTestError("Frozen test metrics do not match predictions")


def render_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
