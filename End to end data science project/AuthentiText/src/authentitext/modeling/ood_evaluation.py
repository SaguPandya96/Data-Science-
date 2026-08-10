"""Evaluate the frozen baseline on development-only MAGE OOD stress sets."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from authentitext.data.cleaning import open_deterministic_gzip
from authentitext.modeling.baselines import positive_scores, sha256_file
from authentitext.modeling.calibration import calibrate_scores, calibration_metrics

OOD_EVALUATION_VERSION = 1
REPORT_SCHEMA_VERSION = 1
FAMILIES = ("human", "gpt4", "gpt4_para", "human_para")


class OodEvaluationError(RuntimeError):
    """Raised when OOD inputs, predictions, or reports are invalid."""


@dataclass(frozen=True)
class OodRecord:
    record_id: str
    content_id: str
    file_name: str
    source: str
    domain: str
    family: str
    target: int
    text: str
    whitespace_tokens: int


def _source_identity(source: str, target: int) -> tuple[str, str]:
    domain, separator, suffix = source.partition("_")
    if not separator or not domain or suffix not in FAMILIES:
        raise OodEvaluationError(f"Unsupported OOD source: {source!r}")
    expected_target = 0 if suffix == "human" else 1
    if target != expected_target:
        raise OodEvaluationError(f"OOD source/target mismatch: {source!r}")
    return domain, suffix


def _record_id(
    dataset_id: str,
    revision: str,
    file_name: str,
    content_id: str,
    raw_label: str,
    source: str,
) -> str:
    value = "\0".join((dataset_id, revision, file_name, content_id, raw_label, source))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_ood_records(
    raw_root: Path, manifest: dict[str, Any]
) -> tuple[list[OodRecord], list[dict[str, Any]]]:
    """Load manifest-verified CSV inputs with deterministic text-free IDs."""
    dataset_id = manifest.get("dataset_id")
    revision = manifest.get("revision")
    if not isinstance(dataset_id, str) or not isinstance(revision, str):
        raise OodEvaluationError("OOD manifest identity is invalid")
    records = []
    record_ids = set()
    content_targets: dict[str, int] = {}
    identities = []
    for file_entry in manifest.get("files", []):
        name = file_entry.get("name")
        relative_path = file_entry.get("relative_path")
        expected_columns = file_entry.get("expected_columns")
        if (
            not isinstance(name, str)
            or not isinstance(relative_path, str)
            or not isinstance(expected_columns, list)
        ):
            raise OodEvaluationError("OOD manifest file entry is invalid")
        path = raw_root / relative_path
        if (
            not path.is_file()
            or path.stat().st_size != file_entry.get("size_bytes")
            or sha256_file(path) != file_entry.get("sha256")
        ):
            raise OodEvaluationError(f"OOD file identity mismatch: {path}")
        file_rows = 0
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != expected_columns:
                raise OodEvaluationError(f"OOD columns do not match manifest: {path}")
            for row_number, row in enumerate(reader, start=2):
                text = row.get("text")
                raw_label = row.get("label")
                source = row.get("src")
                if not isinstance(text, str) or not text.strip() or raw_label not in {"0", "1"}:
                    raise OodEvaluationError(f"{path}:{row_number} has invalid fields")
                if not isinstance(source, str):
                    raise OodEvaluationError(f"{path}:{row_number} has invalid source")
                target = 1 if raw_label == "0" else 0
                domain, family = _source_identity(source, target)
                content_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
                record_id = _record_id(
                    dataset_id,
                    revision,
                    name,
                    content_id,
                    raw_label,
                    source,
                )
                if record_id in record_ids:
                    raise OodEvaluationError(f"{path}:{row_number} repeats a stable record ID")
                if content_id in content_targets and content_targets[content_id] != target:
                    raise OodEvaluationError(f"{path}:{row_number} conflicts on an exact text")
                record_ids.add(record_id)
                content_targets[content_id] = target
                records.append(
                    OodRecord(
                        record_id=record_id,
                        content_id=content_id,
                        file_name=name,
                        source=source,
                        domain=domain,
                        family=family,
                        target=target,
                        text=text,
                        whitespace_tokens=len(text.split()),
                    )
                )
                file_rows += 1
        identities.append(
            {
                "name": name,
                "relative_path": relative_path,
                "rows": file_rows,
                "bytes": path.stat().st_size,
                "sha256": file_entry["sha256"],
            }
        )
    if not records or {record.target for record in records} != {0, 1}:
        raise OodEvaluationError("OOD inputs must contain both targets")
    return records, identities


def _categories(scores: np.ndarray, human_threshold: float, machine_threshold: float) -> np.ndarray:
    categories = np.full(len(scores), "uncertain", dtype=object)
    categories[scores <= human_threshold] = "likely_human"
    categories[scores >= machine_threshold] = "likely_machine"
    return categories


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float] | None:
    if not trials:
        return None
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return [round(max(0.0, centre - margin), 6), round(min(1.0, centre + margin), 6)]


def _policy_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    human_threshold: float,
    machine_threshold: float,
) -> dict[str, Any]:
    categories = _categories(scores, human_threshold, machine_threshold)
    human = targets == 0
    machine = targets == 1
    human_rows = int(human.sum())
    machine_rows = int(machine.sum())
    false_machine = int(((categories == "likely_machine") & human).sum())
    false_human = int(((categories == "likely_human") & machine).sum())
    likely_human = int((categories == "likely_human").sum())
    likely_machine = int((categories == "likely_machine").sum())
    uncertain = int((categories == "uncertain").sum())
    correct = int(((categories == "likely_human") & human).sum()) + int(
        ((categories == "likely_machine") & machine).sum()
    )
    decisive = likely_human + likely_machine
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
        "decisive_accuracy": round(correct / decisive, 6) if decisive else None,
        "human_false_machine_count": false_machine,
        "human_false_machine_rate": (round(false_machine / human_rows, 6) if human_rows else None),
        "human_false_machine_wilson_95": _wilson(false_machine, human_rows),
        "machine_false_human_count": false_human,
        "machine_false_human_rate": (
            round(false_human / machine_rows, 6) if machine_rows else None
        ),
        "machine_false_human_wilson_95": _wilson(false_human, machine_rows),
    }


def _slice_metrics(
    targets: np.ndarray,
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray,
    human_threshold: float,
    machine_threshold: float,
) -> dict[str, Any]:
    has_both = set(targets.tolist()) == {0, 1}
    return {
        "rows": len(targets),
        "target_counts": {
            "0": int((targets == 0).sum()),
            "1": int((targets == 1).sum()),
        },
        "raw_ranking": {
            "roc_auc": round(float(roc_auc_score(targets, raw_scores)), 6) if has_both else None,
            "average_precision": round(float(average_precision_score(targets, raw_scores)), 6)
            if has_both
            else None,
        },
        "calibrated_probability": calibration_metrics(targets, calibrated_scores),
        "policy": _policy_metrics(targets, calibrated_scores, human_threshold, machine_threshold),
    }


def _metrics(
    records: list[OodRecord],
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray,
    human_threshold: float,
    machine_threshold: float,
) -> dict[str, Any]:
    targets = np.asarray([record.target for record in records], dtype=np.int8)
    files = np.asarray([record.file_name for record in records], dtype=object)
    domains = np.asarray([record.domain for record in records], dtype=object)
    families = np.asarray([record.family for record in records], dtype=object)
    tokens = np.asarray([record.whitespace_tokens for record in records], dtype=np.int32)
    seen_content = set()
    deduplicated = []
    for record in records:
        deduplicated.append(record.content_id not in seen_content)
        seen_content.add(record.content_id)
    deduplicated_mask = np.asarray(deduplicated, dtype=bool)

    def evaluate(mask: np.ndarray) -> dict[str, Any]:
        return _slice_metrics(
            targets[mask],
            raw_scores[mask],
            calibrated_scores[mask],
            human_threshold,
            machine_threshold,
        )

    by_file = {name: evaluate(files == name) for name in sorted(set(files))}
    by_file_domain = {
        f"{name}:{domain}": evaluate((files == name) & (domains == domain))
        for name in sorted(set(files))
        for domain in sorted(set(domains[files == name]))
    }
    by_domain_deduplicated = {
        domain: evaluate(deduplicated_mask & (domains == domain)) for domain in sorted(set(domains))
    }
    length_masks = {
        "under_50": tokens < 50,
        "50_to_128": (tokens >= 50) & (tokens <= 128),
        "129_to_512": (tokens >= 129) & (tokens <= 512),
        "over_512": tokens > 512,
    }
    by_length = {
        name: evaluate(deduplicated_mask & mask)
        for name, mask in length_masks.items()
        if (deduplicated_mask & mask).any()
    }
    categories = _categories(calibrated_scores, human_threshold, machine_threshold)
    family_outcomes = {}
    for family in FAMILIES:
        mask = deduplicated_mask & (families == family)
        counts = Counter(str(category) for category in categories[mask])
        rows = int(mask.sum())
        family_outcomes[family] = {
            "rows": rows,
            "target": int(targets[mask][0]) if rows else None,
            "category_counts": {
                "likely_human": counts["likely_human"],
                "uncertain": counts["uncertain"],
                "likely_machine": counts["likely_machine"],
            },
            "likely_machine_rate": round(counts["likely_machine"] / rows, 6) if rows else None,
            "likely_human_rate": round(counts["likely_human"] / rows, 6) if rows else None,
        }
    return {
        "by_file": by_file,
        "content_deduplicated_combined": evaluate(deduplicated_mask),
        "by_file_and_domain": by_file_domain,
        "by_domain_content_deduplicated": by_domain_deduplicated,
        "by_whitespace_token_band_content_deduplicated": by_length,
        "source_family_outcomes_content_deduplicated": family_outcomes,
    }


def _content_accounting(records: list[OodRecord]) -> dict[str, int]:
    unique = len({record.content_id for record in records})
    return {
        "rows": len(records),
        "unique_content_ids": unique,
        "duplicate_extra_rows": len(records) - unique,
    }


def _write_predictions(
    path: Path,
    records: list[OodRecord],
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
                            "content_id": record.content_id,
                            "domain": record.domain,
                            "family": record.family,
                            "file": record.file_name,
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


def evaluate_ood(
    *,
    raw_root: Path,
    manifest: dict[str, Any],
    base_model_path: Path,
    base_model_identity: dict[str, Any],
    calibration_path: Path,
    calibration_identity: dict[str, Any],
    prediction_path: Path,
) -> dict[str, Any]:
    """Score the two MAGE development OOD files without changing the policy."""
    records, input_identities = load_ood_records(raw_root, manifest)
    if (
        base_model_path.stat().st_size != base_model_identity.get("bytes")
        or sha256_file(base_model_path) != base_model_identity.get("sha256")
        or calibration_path.stat().st_size != calibration_identity.get("bytes")
        or sha256_file(calibration_path) != calibration_identity.get("sha256")
    ):
        raise OodEvaluationError("Frozen artifact identity mismatch")
    base_model = joblib.load(base_model_path)
    calibration = joblib.load(calibration_path)
    if (
        base_model.get("model_type") != "word_tfidf_logistic"
        or calibration.get("model_type") != "calibration_policy"
        or calibration.get("base_model_sha256") != base_model_identity["sha256"]
    ):
        raise OodEvaluationError("Frozen artifact type or linkage mismatch")

    started = time.perf_counter()
    raw_scores = np.round(
        np.asarray(
            positive_scores(base_model, [record.text for record in records]), dtype=np.float64
        ),
        12,
    )
    calibrated_scores = np.round(
        calibrate_scores(calibration["calibration_method"], calibration["calibrator"], raw_scores),
        12,
    )
    categories = _categories(
        calibrated_scores, calibration["human_threshold"], calibration["machine_threshold"]
    )
    scoring_seconds = time.perf_counter() - started
    prediction_identity = _write_predictions(
        prediction_path, records, raw_scores, calibrated_scores, categories
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "ood_evaluation_version": OOD_EVALUATION_VERSION,
        "dataset_id": manifest["dataset_id"],
        "revision": manifest["revision"],
        "role": "development_only_ood_stress_test",
        "inputs": input_identities,
        "base_model": base_model_identity,
        "calibration_artifact": calibration_identity,
        "configuration": {
            "calibration_method": calibration["calibration_method"],
            "human_threshold": calibration["human_threshold"],
            "machine_threshold": calibration["machine_threshold"],
            "score_rounding_decimal_places": 12,
            "policy_frozen_before_ood_evaluation": True,
            "retuning_from_ood_results_allowed": False,
            "published_test_data_used": False,
            "prediction_format": "deterministic gzip JSON Lines with no text",
            "combined_metric_policy": "deduplicate exact content_id across files",
        },
        "predictions": prediction_identity,
        "content_accounting": _content_accounting(records),
        "scoring": {
            "seconds": round(scoring_seconds, 3),
            "records_per_second": round(len(records) / scoring_seconds, 3),
            "batch_size": len(records),
        },
        "metrics": _metrics(
            records,
            raw_scores,
            calibrated_scores,
            calibration["human_threshold"],
            calibration["machine_threshold"],
        ),
        "validation": {
            "status": "pass",
            "input_hashes_verified": True,
            "artifact_hashes_and_linkage_verified": True,
            "prediction_file_verified": False,
            "metrics_recomputed_from_predictions": False,
            "published_test_data_used": False,
        },
    }


def _read_predictions(path: Path) -> tuple[list[OodRecord], np.ndarray, np.ndarray, np.ndarray]:
    records = []
    raw_scores = []
    calibrated_scores = []
    categories = []
    seen_ids = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise OodEvaluationError(f"{path}:{line_number} is invalid JSON") from error
            record_id = row.get("record_id")
            if (
                not isinstance(record_id, str)
                or record_id in seen_ids
                or not isinstance(row.get("content_id"), str)
                or not isinstance(row.get("file"), str)
                or not isinstance(row.get("source"), str)
                or not isinstance(row.get("domain"), str)
                or row.get("family") not in FAMILIES
                or row.get("target") not in (0, 1)
                or not isinstance(row.get("whitespace_tokens"), int)
                or not isinstance(row.get("raw_score"), int | float)
                or not isinstance(row.get("calibrated_score"), int | float)
                or row.get("category") not in {"likely_human", "uncertain", "likely_machine"}
            ):
                raise OodEvaluationError(f"{path}:{line_number} has invalid fields")
            seen_ids.add(record_id)
            records.append(
                OodRecord(
                    record_id=record_id,
                    content_id=row["content_id"],
                    file_name=row["file"],
                    source=row["source"],
                    domain=row["domain"],
                    family=row["family"],
                    target=row["target"],
                    text="",
                    whitespace_tokens=row["whitespace_tokens"],
                )
            )
            raw_scores.append(row["raw_score"])
            calibrated_scores.append(row["calibrated_score"])
            categories.append(row["category"])
    return (
        records,
        np.asarray(raw_scores, dtype=np.float64),
        np.asarray(calibrated_scores, dtype=np.float64),
        np.asarray(categories, dtype=object),
    )


def verify_ood(report: dict[str, Any], prediction_path: Path, calibration_path: Path) -> None:
    """Verify text-free predictions, calibration, categories, and all metrics."""
    identity = report["predictions"]
    if (
        prediction_path.stat().st_size != identity["bytes"]
        or sha256_file(prediction_path) != identity["sha256"]
    ):
        raise OodEvaluationError("OOD prediction identity mismatch")
    calibration = joblib.load(calibration_path)
    if calibration.get("base_model_sha256") != report["base_model"]["sha256"]:
        raise OodEvaluationError("OOD calibration linkage mismatch")
    records, raw_scores, calibrated_scores, categories = _read_predictions(prediction_path)
    if len(records) != identity["rows"]:
        raise OodEvaluationError("OOD prediction row count mismatch")
    if _content_accounting(records) != report.get("content_accounting"):
        raise OodEvaluationError("OOD content accounting mismatch")
    recalibrated = np.round(
        calibrate_scores(calibration["calibration_method"], calibration["calibrator"], raw_scores),
        12,
    )
    if not np.array_equal(recalibrated, calibrated_scores):
        raise OodEvaluationError("OOD calibrated scores do not match artifact")
    expected_categories = _categories(
        calibrated_scores, calibration["human_threshold"], calibration["machine_threshold"]
    )
    if not np.array_equal(expected_categories, categories):
        raise OodEvaluationError("OOD categories do not match frozen thresholds")
    recomputed = _metrics(
        records,
        raw_scores,
        calibrated_scores,
        calibration["human_threshold"],
        calibration["machine_threshold"],
    )
    if recomputed != report["metrics"]:
        raise OodEvaluationError("OOD metrics do not match predictions")


def render_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
