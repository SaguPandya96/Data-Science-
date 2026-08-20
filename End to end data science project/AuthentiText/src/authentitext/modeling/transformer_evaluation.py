"""Evaluate a hash-frozen transformer on test and development OOD roles."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from authentitext.data.cleaning import sha256_file
from authentitext.data.transformer_train import sha256_gzip_content
from authentitext.modeling.calibration import calibrate_scores
from authentitext.modeling.evaluation import load_evaluation_records
from authentitext.modeling.frozen_test import (
    _categories as test_categories,
)
from authentitext.modeling.frozen_test import (
    _metrics as test_metrics,
)
from authentitext.modeling.frozen_test import (
    _write_predictions as write_test_predictions,
)
from authentitext.modeling.ood_evaluation import (
    _categories as ood_categories,
)
from authentitext.modeling.ood_evaluation import (
    _content_accounting,
    load_ood_records,
)
from authentitext.modeling.ood_evaluation import (
    _metrics as ood_metrics,
)
from authentitext.modeling.ood_evaluation import (
    _write_predictions as write_ood_predictions,
)
from authentitext.modeling.transformer_probe import MAX_LENGTH
from authentitext.modeling.transformer_training import _artifact_files

REPORT_SCHEMA_VERSION = 1
EVALUATION_BATCH_SIZE = 64


class TransformerEvaluationError(RuntimeError):
    """Raised when frozen transformer evaluation violates its contract."""


def verify_frozen_artifacts(
    artifact_root: Path, freeze: dict[str, Any]
) -> tuple[Path, Path, dict[str, Any]]:
    """Verify every model file plus the calibrator identity and linkage."""
    model_identity = freeze["model_artifact"]
    calibration_identity = freeze["calibration_artifact"]
    model_dir = artifact_root / model_identity["relative_path"]
    calibration_path = artifact_root / calibration_identity["relative_path"]
    files, aggregate_sha256, total_bytes = _artifact_files(model_dir)
    if (
        files != model_identity["files"]
        or aggregate_sha256 != model_identity["sha256"]
        or total_bytes != model_identity["bytes"]
    ):
        raise TransformerEvaluationError("Frozen transformer model identity mismatch")
    if (
        not calibration_path.is_file()
        or calibration_path.stat().st_size != calibration_identity["bytes"]
        or sha256_file(calibration_path) != calibration_identity["sha256"]
    ):
        raise TransformerEvaluationError("Frozen calibration artifact identity mismatch")
    calibration = joblib.load(calibration_path)
    expected = (
        calibration.get("model_type") == "calibration_policy"
        and calibration.get("base_model_sha256") == model_identity["sha256"]
        and calibration.get("calibration_method") == calibration_identity["method"]
        and round(calibration.get("human_threshold"), 12) == calibration_identity["human_threshold"]
        and round(calibration.get("machine_threshold"), 12)
        == calibration_identity["machine_threshold"]
    )
    if not expected:
        raise TransformerEvaluationError("Frozen calibration policy does not match its declaration")
    return model_dir, calibration_path, calibration


def score_texts(model_dir: Path, texts: list[str]) -> tuple[np.ndarray, float]:
    """Score text in bounded CPU batches with the saved local-only checkpoint."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise TransformerEvaluationError(f"Transformer dependency is missing: {error}") from error

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    chunks = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(texts), EVALUATION_BATCH_SIZE):
            encoded = tokenizer(
                texts[start : start + EVALUATION_BATCH_SIZE],
                max_length=MAX_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            chunks.append(torch.softmax(model(**encoded).logits, dim=1)[:, 1].cpu())
    seconds = time.perf_counter() - started
    return np.round(torch.cat(chunks).numpy().astype(np.float64), 12), seconds


def evaluate_frozen_transformer(
    *,
    test_path: Path,
    test_materialization: dict[str, Any],
    raw_ood_root: Path,
    ood_manifest: dict[str, Any],
    artifact_root: Path,
    freeze: dict[str, Any],
    test_prediction_path: Path,
    ood_prediction_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the one-time test and fixed-policy OOD evaluation without retuning."""
    model_dir, _, calibration = verify_frozen_artifacts(artifact_root, freeze)
    if (
        test_materialization.get("partition") != "test"
        or not test_materialization.get("test_data_read")
        or test_materialization.get("status") != "pass"
        or test_materialization.get("output_content_sha256") != sha256_gzip_content(test_path)
    ):
        raise TransformerEvaluationError("Materialized test identity is invalid")

    test_records = load_evaluation_records(test_path, "test")
    if len(test_records) != test_materialization["rows_written"]:
        raise TransformerEvaluationError("Materialized test row count is invalid")
    test_raw, test_seconds = score_texts(model_dir, [record.text for record in test_records])
    test_calibrated = np.round(
        calibrate_scores(calibration["calibration_method"], calibration["calibrator"], test_raw),
        12,
    )
    test_category = test_categories(
        test_calibrated, calibration["human_threshold"], calibration["machine_threshold"]
    )
    test_prediction_identity = write_test_predictions(
        path=test_prediction_path,
        records=test_records,
        raw_scores=test_raw,
        calibrated_scores=test_calibrated,
        categories=test_category,
    )
    targets = np.asarray([record.target for record in test_records], dtype=np.int8)
    sources = [record.source for record in test_records]
    lengths = np.asarray([record.whitespace_tokens for record in test_records], dtype=np.int32)
    test_report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluation": "frozen_transformer_test",
        "dataset_id": freeze.get("dataset_id", "yaful/MAGE"),
        "revision": "342663f0a2b775455c023f5d36a1341ff0ec5402",
        "partition": "test",
        "input": test_materialization,
        "base_model": freeze["model_artifact"],
        "calibration_artifact": freeze["calibration_artifact"],
        "configuration": {
            "max_length": MAX_LENGTH,
            "evaluation_batch_size": EVALUATION_BATCH_SIZE,
            "artifacts_frozen_before_test": True,
            "retuning_after_test_allowed": False,
            "prediction_format": "deterministic gzip JSON Lines with no text",
        },
        "predictions": test_prediction_identity,
        "scoring": {
            "seconds": round(test_seconds, 3),
            "records_per_second": round(len(test_records) / test_seconds, 3),
        },
        "metrics": test_metrics(
            targets=targets,
            raw_scores=test_raw,
            calibrated_scores=test_calibrated,
            sources=sources,
            whitespace_tokens=lengths,
            human_threshold=calibration["human_threshold"],
            machine_threshold=calibration["machine_threshold"],
        ),
        "validation": {"status": "pass", "artifacts_and_inputs_verified": True},
    }

    ood_records, input_identities = load_ood_records(raw_ood_root, ood_manifest)
    ood_raw, ood_seconds = score_texts(model_dir, [record.text for record in ood_records])
    ood_calibrated = np.round(
        calibrate_scores(calibration["calibration_method"], calibration["calibrator"], ood_raw),
        12,
    )
    ood_category = ood_categories(
        ood_calibrated, calibration["human_threshold"], calibration["machine_threshold"]
    )
    ood_prediction_identity = write_ood_predictions(
        ood_prediction_path, ood_records, ood_raw, ood_calibrated, ood_category
    )
    ood_report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluation": "frozen_transformer_mage_ood",
        "dataset_id": ood_manifest["dataset_id"],
        "revision": ood_manifest["revision"],
        "role": "development_only_ood_stress_test",
        "inputs": input_identities,
        "base_model": freeze["model_artifact"],
        "calibration_artifact": freeze["calibration_artifact"],
        "configuration": {
            "max_length": MAX_LENGTH,
            "evaluation_batch_size": EVALUATION_BATCH_SIZE,
            "policy_frozen_before_ood_evaluation": True,
            "retuning_from_ood_results_allowed": False,
            "published_test_data_used_for_tuning": False,
            "prediction_format": "deterministic gzip JSON Lines with no text",
            "combined_metric_policy": "deduplicate exact content_id across files",
        },
        "predictions": ood_prediction_identity,
        "content_accounting": _content_accounting(ood_records),
        "scoring": {
            "seconds": round(ood_seconds, 3),
            "records_per_second": round(len(ood_records) / ood_seconds, 3),
        },
        "metrics": ood_metrics(
            ood_records,
            ood_raw,
            ood_calibrated,
            calibration["human_threshold"],
            calibration["machine_threshold"],
        ),
        "validation": {"status": "pass", "artifacts_and_inputs_verified": True},
    }
    return test_report, ood_report


def render_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
