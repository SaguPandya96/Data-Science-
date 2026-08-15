from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

from authentitext.api import MAX_BATCH_ITEMS, create_app
from authentitext.inference import PredictionError


class FakePredictor:
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
        if not isinstance(text, str):
            raise PredictionError("text_type", "Text must be a string")
        if not text.strip():
            raise PredictionError("text_blank", "Text must not be blank")
        return {
            "schema_version": 1,
            "category": "uncertain",
            "calibrated_machine_likelihood": 0.5,
            "raw_model_score": 0.5,
            "evidence_quality": "standard",
            "input_summary": {"characters": len(text), "whitespace_tokens": len(text.split())},
            "thresholds": {"likely_human_max": 0.2, "likely_machine_min": 0.8},
            "warnings": [],
            "limitations": [],
            "model": {"name": "fixture"},
        }


class ApiTests(unittest.TestCase):
    def test_health_version_and_model_metadata(self) -> None:
        with TestClient(create_app(predictor=FakePredictor())) as client:
            self.assertEqual(client.get("/health/live").status_code, 200)
            ready = client.get("/health/ready")
            version = client.get("/v1/version")
            model = client.get("/v1/model")

        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json()["status"], "ready")
        self.assertEqual(version.json()["api_version"], "v1")
        self.assertEqual(model.json()["dataset_revision"], "abc123")
        self.assertEqual(model.json()["thresholds"]["likely_machine_min"], 0.8)

    def test_prediction_does_not_echo_or_log_submitted_text(self) -> None:
        secret = "private submitted sentence 9f6e4451"
        with TestClient(create_app(predictor=FakePredictor())) as client:
            with self.assertLogs("authentitext.api", level="INFO") as captured:
                response = client.post("/v1/predict", json={"text": secret})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["input_summary"]["characters"], len(secret))
        self.assertNotIn(secret, response.text)
        self.assertNotIn(secret, "\n".join(captured.output))

    def test_structured_single_and_request_validation_errors_are_sanitized(self) -> None:
        secret = "not allowed in validation response"
        with TestClient(create_app(predictor=FakePredictor())) as client:
            blank = client.post("/v1/predict", json={"text": "   "})
            wrong_type = client.post("/v1/predict", json={"text": {"secret": secret}})
            extra = client.post("/v1/predict", json={"text": "valid", "extra": secret})

        self.assertEqual(blank.status_code, 422)
        self.assertEqual(blank.json()["error"]["code"], "text_blank")
        self.assertEqual(wrong_type.json()["error"]["code"], "request_validation")
        self.assertEqual(extra.json()["error"]["code"], "request_validation")
        self.assertNotIn(secret, wrong_type.text)
        self.assertNotIn(secret, extra.text)

    def test_batch_limits_and_indexed_errors(self) -> None:
        with TestClient(create_app(predictor=FakePredictor())) as client:
            valid = client.post("/v1/predict/batch", json={"texts": ["first", "second"]})
            blank = client.post("/v1/predict/batch", json={"texts": ["valid", " "]})
            too_many = client.post(
                "/v1/predict/batch", json={"texts": ["item"] * (MAX_BATCH_ITEMS + 1)}
            )
            too_large = client.post(
                "/v1/predict/batch", json={"texts": ["x" * 100_001, "y" * 100_000]}
            )

        self.assertEqual(valid.status_code, 200)
        self.assertEqual(valid.json()["count"], 2)
        self.assertEqual(blank.json()["error"]["code"], "batch_item_invalid")
        self.assertEqual(blank.json()["error"]["item_index"], 1)
        self.assertEqual(too_many.json()["error"]["code"], "request_validation")
        self.assertEqual(too_large.json()["error"]["code"], "batch_too_large")

    def test_not_ready_returns_503_without_internal_error(self) -> None:
        with TestClient(create_app(load_default_predictor=False)) as client:
            ready = client.get("/health/ready")
            predict = client.post("/v1/predict", json={"text": "valid text"})

        self.assertEqual(ready.status_code, 503)
        self.assertEqual(predict.status_code, 503)
        self.assertEqual(predict.json()["error"]["code"], "model_not_ready")
        self.assertNotIn("traceback", json.dumps(predict.json()).casefold())


if __name__ == "__main__":
    unittest.main()
