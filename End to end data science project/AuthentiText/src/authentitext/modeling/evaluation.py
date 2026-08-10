"""Evaluate baseline scores without exposing held-out text."""

from __future__ import annotations

import gzip
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from authentitext.data.cleaning import open_deterministic_gzip
from authentitext.data.regimes import parse_source_id
from authentitext.modeling.baselines import positive_scores, sha256_file

EVALUATION_VERSION = 1
REPORT_SCHEMA_VERSION = 1
FIXED_THRESHOLD = 0.5
CALIBRATION_BINS = 15


class EvaluationError(RuntimeError):
    """Raised when model predictions cannot support a valid evaluation."""


@dataclass(frozen=True)
class EvaluationRecord:
    record_id: str
    source: str
    target: int
    text: str
    whitespace_tokens: int


def load_evaluation_records(path: Path, expected_partition: str) -> list[EvaluationRecord]:
    records = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvaluationError(f"{path}:{line_number} is not valid JSON") from error
            if not isinstance(record, dict) or record.get("partition") != expected_partition:
                raise EvaluationError(f"{path}:{line_number} has invalid partition metadata")
            record_id = record.get("record_id")
            source = record.get("source")
            target = record.get("target")
            text = record.get("text")
            if (
                not isinstance(record_id, str)
                or not isinstance(source, str)
                or target not in (0, 1)
                or not isinstance(text, str)
            ):
                raise EvaluationError(f"{path}:{line_number} has invalid evaluation fields")
            records.append(EvaluationRecord(record_id, source, target, text, len(text.split())))
    if not records or {record.target for record in records} != {0, 1}:
        raise EvaluationError("Evaluation data must be non-empty and contain both targets")
    return records


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


def _classification_metrics(targets: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    predictions = (scores >= FIXED_THRESHOLD).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    human_rows = int(tn + fp)
    machine_rows = int(tp + fn)
    false_positive_rate = fp / human_rows if human_rows else 0.0
    has_both_targets = len(np.unique(targets)) == 2
    return {
        "threshold": FIXED_THRESHOLD,
        "rows": len(targets),
        "target_counts": {
            "0": int((targets == 0).sum()),
            "1": int((targets == 1).sum()),
        },
        "confusion": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "accuracy": round(float(accuracy_score(targets, predictions)), 6),
        "balanced_accuracy": (
            round(float(balanced_accuracy_score(targets, predictions)), 6)
            if has_both_targets
            else None
        ),
        "precision": round(float(precision_score(targets, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(targets, predictions, zero_division=0)), 6),
        "specificity": round(float(tn / human_rows if human_rows else 0.0), 6),
        "f1": round(float(f1_score(targets, predictions, zero_division=0)), 6),
        "false_positive_rate": round(float(false_positive_rate), 6),
        "false_positive_rate_wilson_95": _wilson_interval(int(fp), human_rows),
        "true_positive_rate_wilson_95": _wilson_interval(int(tp), machine_rows),
        "roc_auc": round(float(roc_auc_score(targets, scores)), 6) if has_both_targets else None,
        "average_precision": (
            round(float(average_precision_score(targets, scores)), 6) if has_both_targets else None
        ),
        "brier_score": round(float(brier_score_loss(targets, scores)), 6),
        "log_loss": round(float(log_loss(targets, scores, labels=[0, 1])), 6),
    }


def _calibration_summary(targets: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    bin_indexes = np.minimum((scores * CALIBRATION_BINS).astype(int), CALIBRATION_BINS - 1)
    bins = []
    expected_calibration_error = 0.0
    for index in range(CALIBRATION_BINS):
        mask = bin_indexes == index
        count = int(mask.sum())
        mean_score = float(scores[mask].mean()) if count else None
        observed_rate = float(targets[mask].mean()) if count else None
        if count:
            expected_calibration_error += count / len(targets) * abs(mean_score - observed_rate)
        bins.append(
            {
                "lower_inclusive": round(index / CALIBRATION_BINS, 6),
                "upper_exclusive": round((index + 1) / CALIBRATION_BINS, 6),
                "rows": count,
                "mean_score": round(mean_score, 6) if mean_score is not None else None,
                "observed_machine_rate": (
                    round(observed_rate, 6) if observed_rate is not None else None
                ),
            }
        )
    return {
        "equal_width_bins": CALIBRATION_BINS,
        "expected_calibration_error": round(expected_calibration_error, 6),
        "bins": bins,
    }


def _score_summary(scores: np.ndarray) -> dict[str, float]:
    return {
        "minimum": round(float(np.min(scores)), 6),
        "p05": round(float(np.quantile(scores, 0.05)), 6),
        "median": round(float(np.median(scores)), 6),
        "p95": round(float(np.quantile(scores, 0.95)), 6),
        "maximum": round(float(np.max(scores)), 6),
        "mean": round(float(np.mean(scores)), 6),
    }


def _compact_group_metrics(targets: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    if not len(targets):
        return {
            "rows": 0,
            "target_counts": {"0": 0, "1": 0},
            "confusion": {
                "true_negative": 0,
                "false_positive": 0,
                "false_negative": 0,
                "true_positive": 0,
            },
            "balanced_accuracy": None,
            "false_positive_rate": None,
            "recall": None,
            "roc_auc": None,
            "average_precision": None,
        }
    metrics = _classification_metrics(targets, scores)
    return {
        key: metrics[key]
        for key in (
            "rows",
            "target_counts",
            "confusion",
            "balanced_accuracy",
            "false_positive_rate",
            "recall",
            "roc_auc",
            "average_precision",
        )
    }


def evaluate_scores(
    targets: np.ndarray,
    scores: np.ndarray,
    sources: list[str],
    whitespace_tokens: np.ndarray,
) -> dict[str, Any]:
    """Compute fixed-threshold, ranking, calibration, and subgroup metrics."""
    if len(targets) != len(scores) or len(targets) != len(sources):
        raise EvaluationError("Targets, scores, and sources have different lengths")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise EvaluationError("Scores must be finite probabilities in [0, 1]")
    identities = [parse_source_id(source) for source in sources]
    domains = np.asarray([identity.domain for identity in identities], dtype=object)
    generators = np.asarray(
        [identity.generator or "human" for identity in identities], dtype=object
    )
    strategies = np.asarray([identity.strategy or "human" for identity in identities], dtype=object)

    by_domain = {}
    for domain in sorted(set(domains)):
        mask = domains == domain
        by_domain[domain] = _compact_group_metrics(targets[mask], scores[mask])

    by_generator_machine = {}
    for generator in sorted(set(generators) - {"human"}):
        mask = generators == generator
        predictions = scores[mask] >= FIXED_THRESHOLD
        by_generator_machine[generator] = {
            "rows": int(mask.sum()),
            "mean_score": round(float(scores[mask].mean()), 6),
            "recall": round(float(predictions.mean()), 6),
        }

    by_strategy_machine = {}
    for strategy in sorted(set(strategies) - {"human"}):
        mask = strategies == strategy
        predictions = scores[mask] >= FIXED_THRESHOLD
        by_strategy_machine[strategy] = {
            "rows": int(mask.sum()),
            "mean_score": round(float(scores[mask].mean()), 6),
            "recall": round(float(predictions.mean()), 6),
        }

    length_bands = {
        "under_50": whitespace_tokens < 50,
        "50_to_128": (whitespace_tokens >= 50) & (whitespace_tokens <= 128),
        "129_to_512": (whitespace_tokens >= 129) & (whitespace_tokens <= 512),
        "over_512": whitespace_tokens > 512,
    }
    by_length = {
        name: _compact_group_metrics(targets[mask], scores[mask])
        for name, mask in length_bands.items()
    }
    return {
        "overall": _classification_metrics(targets, scores),
        "calibration": _calibration_summary(targets, scores),
        "score_distribution": {
            "human_target_0": _score_summary(scores[targets == 0]),
            "machine_target_1": _score_summary(scores[targets == 1]),
        },
        "by_domain": by_domain,
        "by_generator_machine": by_generator_machine,
        "by_strategy_machine": by_strategy_machine,
        "by_whitespace_token_band": by_length,
    }


def _write_predictions(
    path: Path, records: list[EvaluationRecord], scores: np.ndarray
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with open_deterministic_gzip(temporary) as handle:
            for record, score in zip(records, scores, strict=True):
                handle.write(
                    json.dumps(
                        {
                            "record_id": record.record_id,
                            "score": float(score),
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


def evaluate_baselines(
    *,
    evaluation_path: Path,
    evaluation_partition: str,
    evaluation_identity: dict[str, Any],
    training_report: dict[str, Any],
    artifact_root: Path,
    prediction_root: Path,
) -> dict[str, Any]:
    """Score all baseline artifacts and save reproducible text-free predictions."""
    records = load_evaluation_records(evaluation_path, evaluation_partition)
    texts = [record.text for record in records]
    targets = np.asarray([record.target for record in records], dtype=np.int8)
    sources = [record.source for record in records]
    whitespace_tokens = np.asarray([record.whitespace_tokens for record in records], dtype=np.int32)
    model_reports = []
    for artifact in training_report["artifacts"]:
        payload = joblib.load(artifact_root / artifact["relative_path"])
        scoring_started = time.perf_counter()
        scores = np.round(
            np.asarray(positive_scores(payload, texts), dtype=np.float64),
            12,
        )
        scoring_seconds = time.perf_counter() - scoring_started
        prediction_path = prediction_root / f"{artifact['model_type']}.jsonl.gz"
        prediction_identity = _write_predictions(prediction_path, records, scores)
        model_reports.append(
            {
                "model_type": artifact["model_type"],
                "model_artifact": {
                    "relative_path": artifact["relative_path"],
                    "bytes": artifact["bytes"],
                    "sha256": artifact["sha256"],
                },
                "predictions": prediction_identity,
                "scoring": {
                    "seconds": round(scoring_seconds, 3),
                    "records_per_second": round(len(records) / scoring_seconds, 3),
                    "batch_size": len(records),
                },
                "metrics": evaluate_scores(targets, scores, sources, whitespace_tokens),
            }
        )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "dataset_id": training_report["dataset_id"],
        "revision": training_report["revision"],
        "partition": evaluation_partition,
        "input": evaluation_identity,
        "configuration": {
            "fixed_threshold": FIXED_THRESHOLD,
            "machine_positive_target": 1,
            "calibration_bins": CALIBRATION_BINS,
            "score_rounding_decimal_places": 12,
            "test_data_used": False,
            "prediction_format": "deterministic gzip JSON Lines with no text",
        },
        "models": model_reports,
        "validation": {
            "status": "pass",
            "models_evaluated": len(model_reports),
            "prediction_files_verified": False,
            "metrics_recomputed_from_predictions": False,
        },
    }


def _read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    targets = []
    scores = []
    sources = []
    whitespace_tokens = []
    record_ids = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvaluationError(f"{path}:{line_number} is not valid JSON") from error
            record_id = record.get("record_id")
            if not isinstance(record_id, str) or record_id in record_ids:
                raise EvaluationError(f"{path}:{line_number} has an invalid record ID")
            record_ids.add(record_id)
            targets.append(record.get("target"))
            scores.append(record.get("score"))
            sources.append(record.get("source"))
            whitespace_tokens.append(record.get("whitespace_tokens"))
    if (
        any(target not in (0, 1) for target in targets)
        or any(not isinstance(score, int | float) for score in scores)
        or any(not isinstance(source, str) for source in sources)
        or any(not isinstance(value, int) or value < 0 for value in whitespace_tokens)
    ):
        raise EvaluationError(f"Prediction fields are invalid: {path}")
    return (
        np.asarray(targets, dtype=np.int8),
        np.asarray(scores, dtype=np.float64),
        sources,
        np.asarray(whitespace_tokens, dtype=np.int32),
    )


def verify_evaluation(report: dict[str, Any], prediction_root: Path) -> None:
    """Verify prediction identities and recompute every metric from them."""
    for model in report["models"]:
        identity = model["predictions"]
        path = prediction_root / identity["relative_path"]
        if not path.is_file():
            raise EvaluationError(f"Missing prediction file: {path}")
        if path.stat().st_size != identity["bytes"]:
            raise EvaluationError(f"Prediction size mismatch: {path}")
        if sha256_file(path) != identity["sha256"]:
            raise EvaluationError(f"Prediction SHA-256 mismatch: {path}")
        targets, scores, sources, whitespace_tokens = _read_predictions(path)
        if len(targets) != identity["rows"]:
            raise EvaluationError(f"Prediction row count mismatch: {path}")
        recomputed = evaluate_scores(targets, scores, sources, whitespace_tokens)
        if recomputed != model["metrics"]:
            raise EvaluationError(f"Prediction metrics do not match report: {path}")


def render_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
