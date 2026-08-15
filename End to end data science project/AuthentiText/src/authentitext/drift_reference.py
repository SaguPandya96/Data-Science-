"""Build a validation-only aggregate drift reference and backtest."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from authentitext.data.regimes import parse_source_id
from authentitext.drift import DRIFT_VERSION, SIGNALS, DriftError, aggregate_distribution
from authentitext.modeling.baselines import sha256_file
from authentitext.modeling.calibration import calibrate_scores

HASH_WINDOWS = 60
THRESHOLD_SELECTION_WINDOWS = tuple(range(40))
FALSE_ALERT_AUDIT_WINDOWS = tuple(range(40, HASH_WINDOWS))


class DriftReferenceError(DriftError):
    """Raised when source artifacts cannot produce a valid drift reference."""


@dataclass(frozen=True)
class DriftRecord:
    record_id: str
    domain: str
    characters: int
    whitespace_tokens: int
    calibrated_machine_likelihood: float
    category: str

    def distribution_record(self) -> dict[str, Any]:
        return {
            "characters": self.characters,
            "whitespace_tokens": self.whitespace_tokens,
            "calibrated_machine_likelihood": self.calibrated_machine_likelihood,
            "category": self.category,
        }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DriftReferenceError(f"Cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise DriftReferenceError(f"{path} must contain a JSON object")
    return value


def _verify_identity(path: Path, identity: dict[str, Any], row_count: int | None = None) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != identity.get("bytes")
        or sha256_file(path) != identity.get("sha256")
        or row_count is not None
        and row_count != identity.get("rows", identity.get("rows_written"))
    ):
        raise DriftReferenceError(f"Identity check failed for {path}")


def _read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise DriftReferenceError(f"{path}:{line_number} is not valid JSON") from error
            record_id = row.get("record_id")
            if (
                not isinstance(record_id, str)
                or record_id in predictions
                or not isinstance(row.get("score"), int | float)
                or not 0 <= row["score"] <= 1
                or not isinstance(row.get("source"), str)
                or row.get("target") not in (0, 1)
                or not isinstance(row.get("whitespace_tokens"), int)
            ):
                raise DriftReferenceError(f"{path}:{line_number} has invalid prediction fields")
            predictions[record_id] = row
    if not predictions:
        raise DriftReferenceError("Validation predictions are empty")
    return predictions


def load_drift_records(
    *,
    validation_path: Path,
    prediction_path: Path,
    calibration_path: Path,
    split_report_path: Path,
    calibration_report_path: Path,
) -> tuple[list[DriftRecord], dict[str, Any]]:
    """Load hash-verified validation rows and produce text-free drift records."""
    split_report = _load_json(split_report_path)
    calibration_report = _load_json(calibration_report_path)
    try:
        validation_identity = next(
            output for output in split_report["outputs"] if output["partition"] == "validation"
        )
        prediction_identity = calibration_report["input_predictions"]
        calibration_identity = calibration_report["artifact"]
        base_model_identity = calibration_report["base_model"]
    except (KeyError, StopIteration, TypeError) as error:
        raise DriftReferenceError("Source reports have invalid identities") from error

    _verify_identity(validation_path, validation_identity)
    _verify_identity(prediction_path, prediction_identity)
    _verify_identity(calibration_path, calibration_identity)
    predictions = _read_predictions(prediction_path)
    if len(predictions) != validation_identity.get("rows_written"):
        raise DriftReferenceError("Prediction and validation row counts differ")

    try:
        policy = joblib.load(calibration_path)
    except Exception as error:
        raise DriftReferenceError(f"Cannot load calibration policy: {error}") from error
    if (
        not isinstance(policy, dict)
        or policy.get("model_type") != "calibration_policy"
        or policy.get("base_model_sha256") != base_model_identity.get("sha256")
    ):
        raise DriftReferenceError("Calibration policy is invalid or linked to another model")

    validation_rows = []
    raw_scores = []
    seen_ids = set()
    with gzip.open(validation_path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise DriftReferenceError(
                    f"{validation_path}:{line_number} is not valid JSON"
                ) from error
            record_id = row.get("record_id")
            source = row.get("source")
            text = row.get("text")
            target = row.get("target")
            prediction = predictions.get(record_id)
            if (
                row.get("partition") != "validation"
                or not isinstance(record_id, str)
                or record_id in seen_ids
                or not isinstance(source, str)
                or not isinstance(text, str)
                or target not in (0, 1)
                or prediction is None
                or prediction["source"] != source
                or prediction["target"] != target
                or prediction["whitespace_tokens"] != len(text.split())
            ):
                raise DriftReferenceError(
                    f"{validation_path}:{line_number} does not match predictions"
                )
            seen_ids.add(record_id)
            validation_rows.append((record_id, source, len(text), len(text.split())))
            raw_scores.append(prediction["score"])
    if seen_ids != predictions.keys():
        raise DriftReferenceError("Validation and prediction record IDs differ")

    calibrated = calibrate_scores(
        policy["calibration_method"],
        policy["calibrator"],
        np.asarray(raw_scores, dtype=np.float64),
    )
    records = []
    for (record_id, source, characters, tokens), score in zip(
        validation_rows, calibrated, strict=True
    ):
        if score <= policy["human_threshold"]:
            category = "likely_human"
        elif score >= policy["machine_threshold"]:
            category = "likely_machine"
        else:
            category = "uncertain"
        records.append(
            DriftRecord(
                record_id=record_id,
                domain=parse_source_id(source).domain,
                characters=characters,
                whitespace_tokens=tokens,
                calibrated_machine_likelihood=float(score),
                category=category,
            )
        )

    identities = {
        "dataset_id": split_report["dataset_id"],
        "revision": split_report["revision"],
        "validation": {
            "relative_path": validation_identity["relative_path"],
            "rows": len(records),
            "bytes": validation_identity["bytes"],
            "sha256": validation_identity["sha256"],
        },
        "predictions": {
            "relative_path": prediction_identity["relative_path"],
            "rows": len(predictions),
            "bytes": prediction_identity["bytes"],
            "sha256": prediction_identity["sha256"],
        },
        "model": {
            "name": "word_tfidf_logistic_isotonic",
            "base_model_sha256": base_model_identity["sha256"],
            "calibration_sha256": calibration_identity["sha256"],
        },
    }
    return records, identities


def _hash_window(record_id: str) -> int:
    digest = hashlib.sha256(record_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % HASH_WINDOWS


def _distribution(records: Iterable[DriftRecord]) -> dict[str, Any]:
    return aggregate_distribution(record.distribution_record() for record in records)


def _distance_report(records: list[DriftRecord], reference: dict[str, Any]) -> dict[str, float]:
    from authentitext.drift import signal_distances

    return signal_distances(_distribution(records), reference)


def _backtest_row(
    *,
    name: str,
    records: list[DriftRecord],
    reference: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    distances = _distance_report(records, reference)
    flags = {signal: distances[signal] > thresholds[signal] for signal in SIGNALS}
    return {
        "name": name,
        "rows": len(records),
        "investigation_required": any(flags.values()),
        "signals": {
            signal: {
                "total_variation": round(distances[signal], 6),
                "threshold": round(thresholds[signal], 6),
                "flagged": flags[signal],
            }
            for signal in SIGNALS
        },
    }


def build_drift_reference(
    *,
    validation_path: Path,
    prediction_path: Path,
    calibration_path: Path,
    split_report_path: Path,
    calibration_report_path: Path,
) -> dict[str, Any]:
    """Derive thresholds and audit them without reading published test data."""
    records, identities = load_drift_records(
        validation_path=validation_path,
        prediction_path=prediction_path,
        calibration_path=calibration_path,
        split_report_path=split_report_path,
        calibration_report_path=calibration_report_path,
    )
    windows: dict[int, list[DriftRecord]] = {index: [] for index in range(HASH_WINDOWS)}
    for record in records:
        windows[_hash_window(record.record_id)].append(record)
    if any(not window for window in windows.values()):
        raise DriftReferenceError("A deterministic reference window is empty")

    selection_records = [
        record for index in THRESHOLD_SELECTION_WINDOWS for record in windows[index]
    ]
    reference = _distribution(selection_records)
    selection_distances = {
        index: _distance_report(
            windows[index],
            _distribution(
                record
                for other_index in THRESHOLD_SELECTION_WINDOWS
                if other_index != index
                for record in windows[other_index]
            ),
        )
        for index in THRESHOLD_SELECTION_WINDOWS
    }
    thresholds = {
        signal: max(selection_distances[index][signal] for index in THRESHOLD_SELECTION_WINDOWS)
        for signal in SIGNALS
    }
    minimum_observations = min(len(window) for window in windows.values())

    audit_rows = [
        _backtest_row(
            name=f"hash_window_{index}",
            records=windows[index],
            reference=reference,
            thresholds=thresholds,
        )
        for index in FALSE_ALERT_AUDIT_WINDOWS
    ]
    domains: defaultdict[str, list[DriftRecord]] = defaultdict(list)
    for index in FALSE_ALERT_AUDIT_WINDOWS:
        for record in windows[index]:
            domains[record.domain].append(record)
    domain_rows = [
        _backtest_row(
            name=domain,
            records=domain_records,
            reference=reference,
            thresholds=thresholds,
        )
        for domain, domain_records in sorted(domains.items())
    ]
    audit_flags = sum(row["investigation_required"] for row in audit_rows)
    domain_flags = sum(row["investigation_required"] for row in domain_rows)
    identities["reference"] = {
        "rows": len(selection_records),
        "hash_windows": list(THRESHOLD_SELECTION_WINDOWS),
    }

    return {
        "schema_version": 1,
        "drift_version": DRIFT_VERSION,
        "identity": identities,
        "configuration": {
            "distance": "total variation",
            "hash_window_assignment": "sha256(record_id) first 64 bits modulo 60",
            "threshold_selection_windows": list(THRESHOLD_SELECTION_WINDOWS),
            "false_alert_audit_windows": list(FALSE_ALERT_AUDIT_WINDOWS),
            "reference_windows": list(THRESHOLD_SELECTION_WINDOWS),
            "threshold_rule": (
                "maximum leave-one-window-out threshold-selection distance per signal"
            ),
            "investigation_rule": "flag when any signal distance is strictly above threshold",
            "audit_and_domain_records_disjoint_from_reference": True,
            "automatic_action": False,
            "test_data_used": False,
        },
        "reference_distribution": reference,
        "thresholds": {
            "minimum_observations": minimum_observations,
            "total_variation": {signal: round(thresholds[signal], 12) for signal in SIGNALS},
        },
        "backtest": {
            "false_alert_audit": {
                "windows": len(audit_rows),
                "flagged": audit_flags,
                "false_alert_rate": round(audit_flags / len(audit_rows), 6),
                "results": audit_rows,
            },
            "domain_shift": {
                "domains": len(domain_rows),
                "detected": domain_flags,
                "sensitivity": round(domain_flags / len(domain_rows), 6),
                "results": domain_rows,
            },
        },
        "validation": {
            "status": "pass",
            "source_files_hash_verified": True,
            "validation_prediction_ids_matched": True,
            "calibrator_link_verified": True,
            "threshold_and_audit_windows_disjoint": True,
            "reference_and_audit_records_disjoint": True,
            "published_test_data_used": False,
        },
    }


def render_drift_reference(report: dict[str, Any]) -> str:
    """Render deterministic report JSON."""
    return json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
