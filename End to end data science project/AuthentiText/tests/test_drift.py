from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from authentitext.api import create_app
from authentitext.drift import (
    SIGNALS,
    DriftError,
    aggregate_distribution,
    evaluate_drift,
    load_drift_reference,
    total_variation,
)
from authentitext.monitoring import OperationalMetrics

REPO_ROOT = Path(__file__).resolve().parents[1]


def _record(category: str = "uncertain", score: float = 0.5) -> dict[str, object]:
    return {
        "characters": 500,
        "whitespace_tokens": 80,
        "calibrated_machine_likelihood": score,
        "category": category,
    }


def _reference(distribution: dict[str, object], minimum: int = 3) -> dict[str, object]:
    return {
        "identity": {
            "dataset_id": "fixture",
            "validation": {"rows": 3},
            "model": {
                "base_model_sha256": "a" * 64,
                "calibration_sha256": "b" * 64,
            },
        },
        "reference_distribution": distribution,
        "thresholds": {
            "minimum_observations": minimum,
            "total_variation": {signal: 0.1 for signal in SIGNALS},
        },
    }


def _snapshot(distribution: dict[str, object]) -> dict[str, object]:
    predictions = dict(distribution)
    predictions["successful_items"] = predictions.pop("rows")
    return {"predictions": predictions}


class DriftPredictor:
    def __init__(self) -> None:
        self.identity = SimpleNamespace(
            dataset_id="fixture",
            revision="abc123",
            base_model_sha256="a" * 64,
            calibration_sha256="b" * 64,
        )
        self.human_threshold = 0.2
        self.machine_threshold = 0.8


class DriftTests(unittest.TestCase):
    def test_total_variation_has_expected_bounds(self) -> None:
        self.assertEqual(total_variation([2, 2], [2, 2]), 0.0)
        self.assertEqual(total_variation([4, 0], [0, 4]), 1.0)
        with self.assertRaises(DriftError):
            total_variation([0, 0], [1, 1])

    def test_matching_distribution_and_insufficient_window_do_not_flag(self) -> None:
        distribution = aggregate_distribution([_record(), _record(), _record()])
        reference = _reference(distribution)

        insufficient = _snapshot(aggregate_distribution([_record()]))
        matching = _snapshot(distribution)

        self.assertEqual(evaluate_drift(insufficient, reference)["status"], "insufficient_data")
        result = evaluate_drift(matching, reference)
        self.assertEqual(result["status"], "within_reference")
        self.assertFalse(result["investigation_required"])
        self.assertFalse(result["automatic_action"])

    def test_shifted_distribution_requires_investigation(self) -> None:
        reference_distribution = aggregate_distribution([_record(), _record(), _record()])
        shifted = aggregate_distribution([_record("likely_machine", 0.95) for _ in range(3)])

        result = evaluate_drift(_snapshot(shifted), _reference(reference_distribution))

        self.assertEqual(result["status"], "investigate")
        self.assertTrue(result["investigation_required"])
        self.assertTrue(result["signals"]["categories"]["flagged"])
        self.assertTrue(result["signals"]["calibrated_score_histogram"]["flagged"])

    def test_generated_reference_is_valid_and_validation_only(self) -> None:
        reference = load_drift_reference(
            REPO_ROOT / "data" / "metadata" / "mage_drift_reference.json"
        )

        self.assertEqual(reference["identity"]["validation"]["rows"], 50_509)
        self.assertEqual(reference["reference_distribution"]["rows"], 33_647)
        self.assertEqual(reference["backtest"]["false_alert_audit"]["windows"], 20)
        self.assertEqual(reference["backtest"]["false_alert_audit"]["flagged"], 1)
        self.assertEqual(reference["backtest"]["domain_shift"]["detected"], 9)
        self.assertFalse(reference["configuration"]["test_data_used"])
        self.assertFalse(reference["validation"]["published_test_data_used"])

    def test_api_returns_insufficient_then_matching_aggregate_status(self) -> None:
        distribution = aggregate_distribution([_record(), _record(), _record()])
        reference = _reference(distribution)
        metrics = OperationalMetrics()
        application = create_app(
            predictor=DriftPredictor(), metrics=metrics, drift_reference=reference
        )
        with TestClient(application) as client:
            insufficient = client.get("/v1/drift")
            metrics.record_predictions(
                {
                    "category": "uncertain",
                    "calibrated_machine_likelihood": 0.5,
                    "evidence_quality": "standard",
                    "input_summary": {"characters": 500, "whitespace_tokens": 80},
                    "warnings": [],
                }
                for _ in range(3)
            )
            matching = client.get("/v1/drift")

        self.assertEqual(insufficient.status_code, 200)
        self.assertEqual(insufficient.json()["status"], "insufficient_data")
        self.assertEqual(matching.status_code, 200)
        self.assertEqual(matching.json()["status"], "within_reference")
        self.assertFalse(matching.json()["automatic_action"])


if __name__ == "__main__":
    unittest.main()
