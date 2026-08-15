from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

from authentitext.api import create_app
from authentitext.inference import PredictionError
from authentitext.monitoring import OperationalMetrics


class MetricsPredictor:
    def __init__(self) -> None:
        self.identity = SimpleNamespace(
            dataset_id="example/test",
            revision="abc123",
            base_model_sha256="a" * 64,
            calibration_sha256="b" * 64,
        )
        self.human_threshold = 0.2
        self.machine_threshold = 0.8

    def predict(self, text: object) -> dict[str, object]:
        if not isinstance(text, str) or not text.strip():
            raise PredictionError("text_blank", "Text must not be blank")
        return {
            "category": "uncertain",
            "calibrated_machine_likelihood": 0.55,
            "evidence_quality": "low",
            "input_summary": {
                "characters": len(text),
                "whitespace_tokens": len(text.split()),
            },
            "warnings": [{"code": "short_text_low_evidence", "message": "fixture"}],
        }


class OperationalMetricsTests(unittest.TestCase):
    def test_empty_snapshot_is_explicit_and_nonpersistent(self) -> None:
        snapshot = OperationalMetrics(latency_window_capacity=4).snapshot()

        self.assertEqual(snapshot["requests"]["total"], 0)
        self.assertIsNone(snapshot["requests"]["latency_ms"]["p95"])
        self.assertIsNone(snapshot["predictions"]["uncertain_rate"])
        self.assertFalse(snapshot["retention"]["persistent"])
        self.assertFalse(snapshot["retention"]["raw_text"])
        self.assertFalse(snapshot["retention"]["text_hashes"])

    def test_bounded_latency_window_and_nearest_rank_quantiles(self) -> None:
        metrics = OperationalMetrics(latency_window_capacity=3)
        for latency in (1.0, 2.0, 3.0, 10.0):
            metrics.record_http_request("/v1/predict", 200, latency)

        latency = metrics.snapshot()["requests"]["latency_ms"]
        self.assertEqual(latency["sample_size"], 3)
        self.assertEqual(latency["p50"], 3.0)
        self.assertEqual(latency["p95"], 10.0)
        self.assertEqual(latency["maximum"], 10.0)

    def test_api_metrics_aggregate_fixture_traffic_without_text(self) -> None:
        secret = "private monitoring sentence e295bdc2"
        with TestClient(create_app(predictor=MetricsPredictor())) as client:
            valid = client.post("/v1/predict", json={"text": secret})
            invalid = client.post("/v1/predict", json={"text": " "})
            batch = client.post("/v1/predict/batch", json={"texts": ["one", "two"]})
            response = client.get("/v1/metrics")

        self.assertEqual(valid.status_code, 200)
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(batch.status_code, 200)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["requests"]["total"], 3)
        self.assertEqual(payload["requests"]["by_endpoint"]["/v1/predict"], 2)
        self.assertEqual(payload["requests"]["by_status_code"], {"200": 2, "422": 1})
        self.assertEqual(payload["requests"]["errors_by_code"]["text_blank"], 1)
        self.assertEqual(payload["predictions"]["successful_items"], 3)
        self.assertEqual(payload["predictions"]["categories"]["uncertain"], 3)
        self.assertEqual(payload["predictions"]["uncertain_rate"], 1.0)
        self.assertEqual(payload["predictions"]["warnings_by_code"]["short_text_low_evidence"], 3)
        self.assertTrue(payload["model"]["ready"])
        self.assertNotIn(secret, json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
