"""Measure prefix-truncation sensitivity with the frozen baseline policy."""

from __future__ import annotations

import gzip
import json
import re
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from authentitext.data.cleaning import open_deterministic_gzip
from authentitext.modeling.baselines import positive_scores, sha256_file
from authentitext.modeling.calibration import calibrate_scores, calibration_metrics, policy_metrics
from authentitext.modeling.evaluation import evaluate_scores, load_evaluation_records

TRUNCATION_ROBUSTNESS_VERSION = 1
REPORT_SCHEMA_VERSION = 1
TRUNCATION_BUDGETS = (50, 100, 200)
SCORE_DECIMAL_PLACES = 12
CATEGORY_NAMES = ("likely_human", "uncertain", "likely_machine")


class TruncationRobustnessError(RuntimeError):
    """Raised when a truncation evaluation or its audit trail is invalid."""


def truncate_prefix(text: str, budget: int) -> str:
    """Return the prefix ending at the `budget`th non-whitespace token."""
    if budget <= 0:
        raise ValueError("Truncation budget must be positive")
    for index, match in enumerate(re.finditer(r"\S+", text), start=1):
        if index == budget:
            return text[: match.end()]
    return text


def _validate_budgets(budgets: tuple[int, ...]) -> tuple[int, ...]:
    if not budgets or any(not isinstance(value, int) or value <= 0 for value in budgets):
        raise TruncationRobustnessError("Truncation budgets must be positive integers")
    if tuple(sorted(set(budgets))) != budgets:
        raise TruncationRobustnessError("Truncation budgets must be unique and ascending")
    return budgets


def _categories(scores: np.ndarray, human_threshold: float, machine_threshold: float) -> np.ndarray:
    categories = np.full(len(scores), "uncertain", dtype=object)
    categories[scores <= human_threshold] = "likely_human"
    categories[scores >= machine_threshold] = "likely_machine"
    return categories


def _score_delta(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": round(float(np.min(values)), 6),
        "p05": round(float(np.quantile(values, 0.05)), 6),
        "median": round(float(np.median(values)), 6),
        "mean": round(float(np.mean(values)), 6),
        "p95": round(float(np.quantile(values, 0.95)), 6),
        "maximum": round(float(np.max(values)), 6),
        "mean_absolute": round(float(np.mean(np.abs(values))), 6),
    }


def _category_changes(original: np.ndarray, truncated: np.ndarray) -> dict[str, Any]:
    transitions = {
        source: {destination: 0 for destination in CATEGORY_NAMES} for source in CATEGORY_NAMES
    }
    for source, destination in zip(original, truncated, strict=True):
        transitions[str(source)][str(destination)] += 1
    changed = int((original != truncated).sum())
    return {
        "changed_count": changed,
        "changed_rate": round(changed / len(original), 6),
        "transitions": transitions,
    }


def _paired_metrics(
    *,
    targets: np.ndarray,
    original_raw: np.ndarray,
    truncated_raw: np.ndarray,
    original_calibrated: np.ndarray,
    truncated_calibrated: np.ndarray,
    original_categories: np.ndarray,
    truncated_categories: np.ndarray,
) -> dict[str, Any]:
    by_target = {}
    for target in (0, 1):
        mask = targets == target
        changed = int((original_categories[mask] != truncated_categories[mask]).sum())
        rows = int(mask.sum())
        by_target[str(target)] = {
            "rows": rows,
            "raw_score_delta": _score_delta(truncated_raw[mask] - original_raw[mask]),
            "calibrated_score_delta": _score_delta(
                truncated_calibrated[mask] - original_calibrated[mask]
            ),
            "category_changed_count": changed,
            "category_changed_rate": round(changed / rows, 6),
        }
    return {
        "delta_definition": "truncated_minus_original",
        "raw_score_delta": _score_delta(truncated_raw - original_raw),
        "calibrated_score_delta": _score_delta(truncated_calibrated - original_calibrated),
        "category_changes": _category_changes(original_categories, truncated_categories),
        "by_target": by_target,
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
    raw = evaluate_scores(targets, raw_scores, sources, whitespace_tokens)
    return {
        "raw_score": raw["overall"],
        "calibrated_probability": calibration_metrics(targets, calibrated_scores),
        "policy": policy_metrics(targets, calibrated_scores, human_threshold, machine_threshold),
    }


def _condition(
    *,
    budget: int,
    targets: np.ndarray,
    sources: list[str],
    original_tokens: np.ndarray,
    original_raw: np.ndarray,
    truncated_raw: np.ndarray,
    original_calibrated: np.ndarray,
    truncated_calibrated: np.ndarray,
    original_categories: np.ndarray,
    truncated_categories: np.ndarray,
    human_threshold: float,
    machine_threshold: float,
) -> dict[str, Any]:
    return {
        "budget_whitespace_tokens": budget,
        "selection": {
            "rule": "original_whitespace_tokens > budget_whitespace_tokens",
            "rows": len(targets),
            "target_counts": {
                "0": int((targets == 0).sum()),
                "1": int((targets == 1).sum()),
            },
            "original_whitespace_tokens": {
                "minimum": int(np.min(original_tokens)),
                "median": round(float(np.median(original_tokens)), 3),
                "maximum": int(np.max(original_tokens)),
            },
            "truncated_whitespace_tokens": budget,
        },
        "original": _metrics(
            targets=targets,
            raw_scores=original_raw,
            calibrated_scores=original_calibrated,
            sources=sources,
            whitespace_tokens=original_tokens,
            human_threshold=human_threshold,
            machine_threshold=machine_threshold,
        ),
        "truncated": _metrics(
            targets=targets,
            raw_scores=truncated_raw,
            calibrated_scores=truncated_calibrated,
            sources=sources,
            whitespace_tokens=np.full(len(targets), budget, dtype=np.int32),
            human_threshold=human_threshold,
            machine_threshold=machine_threshold,
        ),
        "paired_effects": _paired_metrics(
            targets=targets,
            original_raw=original_raw,
            truncated_raw=truncated_raw,
            original_calibrated=original_calibrated,
            truncated_calibrated=truncated_calibrated,
            original_categories=original_categories,
            truncated_categories=truncated_categories,
        ),
    }


def _write_predictions(path: Path, conditions: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    rows = 0
    try:
        with open_deterministic_gzip(temporary) as handle:
            for condition in conditions:
                budget = condition["budget"]
                for index, record in enumerate(condition["records"]):
                    handle.write(
                        json.dumps(
                            {
                                "budget_whitespace_tokens": budget,
                                "original_calibrated_score": float(
                                    condition["original_calibrated"][index]
                                ),
                                "original_category": str(condition["original_categories"][index]),
                                "original_raw_score": float(condition["original_raw"][index]),
                                "original_whitespace_tokens": record.whitespace_tokens,
                                "record_id": record.record_id,
                                "source": record.source,
                                "target": record.target,
                                "truncated_calibrated_score": float(
                                    condition["truncated_calibrated"][index]
                                ),
                                "truncated_category": str(condition["truncated_categories"][index]),
                                "truncated_raw_score": float(condition["truncated_raw"][index]),
                                "truncated_whitespace_tokens": budget,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    rows += 1
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "relative_path": path.name,
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _check_identity(path: Path, identity: dict[str, Any], label: str) -> None:
    if "bytes" in identity and path.stat().st_size != identity["bytes"]:
        raise TruncationRobustnessError(f"{label} size does not match its identity")
    if "sha256" in identity and sha256_file(path) != identity["sha256"]:
        raise TruncationRobustnessError(f"{label} SHA-256 does not match its identity")


def evaluate_truncation_robustness(
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
    budgets: tuple[int, ...] = TRUNCATION_BUDGETS,
) -> dict[str, Any]:
    """Score paired original and prefix-truncated test records without retuning."""
    budgets = _validate_budgets(budgets)
    _check_identity(test_path, test_identity, "Test input")
    _check_identity(base_model_path, base_model_identity, "Base model")
    _check_identity(calibration_path, calibration_identity, "Calibration artifact")
    records = load_evaluation_records(test_path, "test")
    if "rows" in test_identity and test_identity["rows"] != len(records):
        raise TruncationRobustnessError("Test input row count does not match its identity")

    base_model = joblib.load(base_model_path)
    calibration = joblib.load(calibration_path)
    if base_model.get("model_type") != "word_tfidf_logistic":
        raise TruncationRobustnessError("Frozen base artifact has the wrong model type")
    if calibration.get("base_model_sha256") != base_model_identity.get("sha256"):
        raise TruncationRobustnessError("Calibration artifact is not linked to the base model")

    scoring_started = time.perf_counter()
    original_raw_all = np.round(
        np.asarray(
            positive_scores(base_model, [record.text for record in records]), dtype=np.float64
        ),
        SCORE_DECIMAL_PLACES,
    )
    condition_data = []
    for budget in budgets:
        indexes = np.asarray(
            [index for index, record in enumerate(records) if record.whitespace_tokens > budget],
            dtype=np.int64,
        )
        selected = [records[index] for index in indexes]
        if not selected or {record.target for record in selected} != {0, 1}:
            raise TruncationRobustnessError(
                f"Budget {budget} does not select non-empty data with both targets"
            )
        truncated_texts = [truncate_prefix(record.text, budget) for record in selected]
        if any(len(text.split()) != budget for text in truncated_texts):
            raise TruncationRobustnessError(f"Budget {budget} produced an invalid token count")
        original_raw = original_raw_all[indexes]
        truncated_raw = np.round(
            np.asarray(positive_scores(base_model, truncated_texts), dtype=np.float64),
            SCORE_DECIMAL_PLACES,
        )
        original_calibrated = np.round(
            calibrate_scores(
                calibration["calibration_method"], calibration["calibrator"], original_raw
            ),
            SCORE_DECIMAL_PLACES,
        )
        truncated_calibrated = np.round(
            calibrate_scores(
                calibration["calibration_method"], calibration["calibrator"], truncated_raw
            ),
            SCORE_DECIMAL_PLACES,
        )
        condition_data.append(
            {
                "budget": budget,
                "records": selected,
                "original_raw": original_raw,
                "truncated_raw": truncated_raw,
                "original_calibrated": original_calibrated,
                "truncated_calibrated": truncated_calibrated,
                "original_categories": _categories(
                    original_calibrated,
                    calibration["human_threshold"],
                    calibration["machine_threshold"],
                ),
                "truncated_categories": _categories(
                    truncated_calibrated,
                    calibration["human_threshold"],
                    calibration["machine_threshold"],
                ),
            }
        )
    scoring_seconds = time.perf_counter() - scoring_started
    prediction_identity = _write_predictions(prediction_path, condition_data)
    conditions = []
    for data in condition_data:
        selected = data["records"]
        conditions.append(
            _condition(
                budget=data["budget"],
                targets=np.asarray([record.target for record in selected], dtype=np.int8),
                sources=[record.source for record in selected],
                original_tokens=np.asarray(
                    [record.whitespace_tokens for record in selected], dtype=np.int32
                ),
                original_raw=data["original_raw"],
                truncated_raw=data["truncated_raw"],
                original_calibrated=data["original_calibrated"],
                truncated_calibrated=data["truncated_calibrated"],
                original_categories=data["original_categories"],
                truncated_categories=data["truncated_categories"],
                human_threshold=calibration["human_threshold"],
                machine_threshold=calibration["machine_threshold"],
            )
        )
    total_scored_texts = len(records) + prediction_identity["rows"]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "truncation_robustness_version": TRUNCATION_ROBUSTNESS_VERSION,
        "dataset_id": dataset_id,
        "revision": revision,
        "partition": "test",
        "input": test_identity,
        "base_model": base_model_identity,
        "calibration_artifact": calibration_identity,
        "configuration": {
            "budgets_whitespace_tokens": list(budgets),
            "eligibility_rule": "original_whitespace_tokens > budget_whitespace_tokens",
            "transformation": "prefix_ending_at_nth_non_whitespace_token",
            "prefix_bytes_before_cut_preserved": True,
            "score_rounding_decimal_places": SCORE_DECIMAL_PLACES,
            "model_calibration_and_thresholds_frozen": True,
            "test_outcomes_used_for_budget_selection": False,
            "retuning_after_robustness_evaluation_allowed": False,
            "prediction_format": "deterministic gzip JSON Lines with no text",
        },
        "predictions": prediction_identity,
        "scoring": {
            "seconds": round(scoring_seconds, 3),
            "original_rows_scored": len(records),
            "truncated_rows_scored": prediction_identity["rows"],
            "records_per_second": round(total_scored_texts / scoring_seconds, 3),
        },
        "conditions": conditions,
        "validation": {
            "status": "pass",
            "prediction_file_verified": False,
            "metrics_recomputed_from_predictions": False,
            "all_selected_records_strictly_exceed_budget": True,
            "all_transformed_records_match_budget": True,
            "source_text_in_report_or_predictions": False,
            "model_calibration_or_threshold_retuning_performed": False,
        },
    }


def _read_predictions(path: Path) -> dict[int, dict[str, Any]]:
    grouped: dict[int, dict[str, list[Any]]] = {}
    seen = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise TruncationRobustnessError(
                    f"{path}:{line_number} is not valid JSON"
                ) from error
            budget = record.get("budget_whitespace_tokens")
            record_id = record.get("record_id")
            key = (budget, record_id)
            numeric_fields = (
                "original_raw_score",
                "truncated_raw_score",
                "original_calibrated_score",
                "truncated_calibrated_score",
            )
            if (
                not isinstance(budget, int)
                or budget <= 0
                or not isinstance(record_id, str)
                or key in seen
                or record.get("target") not in (0, 1)
                or not isinstance(record.get("source"), str)
                or not isinstance(record.get("original_whitespace_tokens"), int)
                or record["original_whitespace_tokens"] <= budget
                or record.get("truncated_whitespace_tokens") != budget
                or any(not isinstance(record.get(field), int | float) for field in numeric_fields)
                or record.get("original_category") not in CATEGORY_NAMES
                or record.get("truncated_category") not in CATEGORY_NAMES
            ):
                raise TruncationRobustnessError(f"{path}:{line_number} has invalid fields")
            values = np.asarray([record[field] for field in numeric_fields], dtype=np.float64)
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise TruncationRobustnessError(f"{path}:{line_number} has invalid scores")
            seen.add(key)
            group = grouped.setdefault(
                budget,
                {
                    "record_ids": [],
                    "targets": [],
                    "sources": [],
                    "original_tokens": [],
                    "original_raw": [],
                    "truncated_raw": [],
                    "original_calibrated": [],
                    "truncated_calibrated": [],
                    "original_categories": [],
                    "truncated_categories": [],
                },
            )
            for field, key_name in (
                ("record_id", "record_ids"),
                ("target", "targets"),
                ("source", "sources"),
                ("original_whitespace_tokens", "original_tokens"),
                ("original_raw_score", "original_raw"),
                ("truncated_raw_score", "truncated_raw"),
                ("original_calibrated_score", "original_calibrated"),
                ("truncated_calibrated_score", "truncated_calibrated"),
                ("original_category", "original_categories"),
                ("truncated_category", "truncated_categories"),
            ):
                group[key_name].append(record[field])
    if not grouped:
        raise TruncationRobustnessError("Truncation prediction file is empty")
    return grouped


def verify_truncation_robustness(
    report: dict[str, Any], prediction_path: Path, calibration_path: Path
) -> None:
    """Verify paired predictions and recompute every condition's metrics."""
    identity = report["predictions"]
    if prediction_path.stat().st_size != identity["bytes"]:
        raise TruncationRobustnessError("Truncation prediction size mismatch")
    if sha256_file(prediction_path) != identity["sha256"]:
        raise TruncationRobustnessError("Truncation prediction SHA-256 mismatch")
    _check_identity(calibration_path, report["calibration_artifact"], "Calibration artifact")
    calibration = joblib.load(calibration_path)
    if calibration.get("base_model_sha256") != report["base_model"]["sha256"]:
        raise TruncationRobustnessError("Calibration linkage does not match the report")
    grouped = _read_predictions(prediction_path)
    budgets = tuple(report["configuration"]["budgets_whitespace_tokens"])
    _validate_budgets(budgets)
    if tuple(sorted(grouped)) != budgets:
        raise TruncationRobustnessError("Prediction budgets do not match the report")
    if sum(len(group["targets"]) for group in grouped.values()) != identity["rows"]:
        raise TruncationRobustnessError("Truncation prediction row count mismatch")

    recomputed = []
    for budget in budgets:
        data = grouped[budget]
        targets = np.asarray(data["targets"], dtype=np.int8)
        if set(targets.tolist()) != {0, 1}:
            raise TruncationRobustnessError(f"Budget {budget} predictions lack both targets")
        original_raw = np.asarray(data["original_raw"], dtype=np.float64)
        truncated_raw = np.asarray(data["truncated_raw"], dtype=np.float64)
        original_calibrated = np.asarray(data["original_calibrated"], dtype=np.float64)
        truncated_calibrated = np.asarray(data["truncated_calibrated"], dtype=np.float64)
        expected_original = np.round(
            calibrate_scores(
                calibration["calibration_method"], calibration["calibrator"], original_raw
            ),
            SCORE_DECIMAL_PLACES,
        )
        expected_truncated = np.round(
            calibrate_scores(
                calibration["calibration_method"], calibration["calibrator"], truncated_raw
            ),
            SCORE_DECIMAL_PLACES,
        )
        if not np.array_equal(expected_original, original_calibrated) or not np.array_equal(
            expected_truncated, truncated_calibrated
        ):
            raise TruncationRobustnessError("Saved calibrated scores do not match the artifact")
        original_categories = np.asarray(data["original_categories"], dtype=object)
        truncated_categories = np.asarray(data["truncated_categories"], dtype=object)
        expected_original_categories = _categories(
            expected_original,
            calibration["human_threshold"],
            calibration["machine_threshold"],
        )
        expected_truncated_categories = _categories(
            expected_truncated,
            calibration["human_threshold"],
            calibration["machine_threshold"],
        )
        if not np.array_equal(
            expected_original_categories, original_categories
        ) or not np.array_equal(expected_truncated_categories, truncated_categories):
            raise TruncationRobustnessError("Saved categories do not match frozen thresholds")
        recomputed.append(
            _condition(
                budget=budget,
                targets=targets,
                sources=data["sources"],
                original_tokens=np.asarray(data["original_tokens"], dtype=np.int32),
                original_raw=original_raw,
                truncated_raw=truncated_raw,
                original_calibrated=original_calibrated,
                truncated_calibrated=truncated_calibrated,
                original_categories=original_categories,
                truncated_categories=truncated_categories,
                human_threshold=calibration["human_threshold"],
                machine_threshold=calibration["machine_threshold"],
            )
        )
    if recomputed != report["conditions"]:
        raise TruncationRobustnessError("Truncation metrics do not match predictions")


def render_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
