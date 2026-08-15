from __future__ import annotations

import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

from authentitext.api import create_app


class FrontendPredictor:
    def __init__(self) -> None:
        self.identity = SimpleNamespace(
            dataset_id="example/test",
            revision="abc123",
            base_model_sha256="a" * 64,
            calibration_sha256="b" * 64,
        )
        self.human_threshold = 0.2
        self.machine_threshold = 0.8

    def predict(self, text: str) -> dict[str, object]:
        return {
            "schema_version": 1,
            "category": "uncertain",
            "calibrated_machine_likelihood": 0.5,
            "raw_model_score": 0.5,
            "evidence_quality": "standard",
            "input_summary": {
                "characters": len(text),
                "whitespace_tokens": len(text.split()),
            },
            "thresholds": {"likely_human_max": 0.2, "likely_machine_min": 0.8},
            "warnings": [],
            "limitations": [],
            "model": {
                "name": "fixture",
                "base_model_sha256": "a" * 64,
                "calibration_sha256": "b" * 64,
            },
        }


class FrontendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app(predictor=FrontendPredictor()))
        self.client.__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)

    def test_root_serves_accessible_interface_and_cautious_copy(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/html"))
        self.assertIn("A cautious signal", response.text)
        self.assertIn('id="text-input"', response.text)
        self.assertIn('maxlength="100000"', response.text)
        self.assertIn('autocomplete="off"', response.text)
        self.assertIn('spellcheck="false"', response.text)
        self.assertIn('aria-live="polite"', response.text)
        self.assertIn('class="skip-link"', response.text)
        self.assertIn("should never be used as sole grounds", response.text)

    def test_static_assets_are_local_responsive_and_do_not_persist_text(self) -> None:
        styles = self.client.get("/static/styles.css")
        script = self.client.get("/static/app.js")

        self.assertEqual(styles.status_code, 200)
        self.assertEqual(script.status_code, 200)
        self.assertIn("@media (max-width: 560px)", styles.text)
        self.assertIn("prefers-reduced-motion", styles.text)
        self.assertIn('fetch("/v1/predict"', script.text)
        self.assertIn("JSON.stringify({ text: input.value })", script.text)
        self.assertIn("result.thresholds.likely_human_max", script.text)
        self.assertIn("result.thresholds.likely_machine_min", script.text)
        self.assertIn('input.value = ""', script.text)
        for prohibited in ("localStorage", "sessionStorage", "analytics", "http://", "https://"):
            self.assertNotIn(prohibited, script.text)

    def test_security_headers_cover_interface_and_api(self) -> None:
        for path in ("/", "/static/app.js", "/health/live"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.headers["x-content-type-options"], "nosniff")
                self.assertEqual(response.headers["x-frame-options"], "DENY")
                self.assertEqual(response.headers["referrer-policy"], "no-referrer")
                policy = response.headers["content-security-policy"]
                self.assertIn("default-src 'self'", policy)
                self.assertIn("frame-ancestors 'none'", policy)


if __name__ == "__main__":
    unittest.main()
