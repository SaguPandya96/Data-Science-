from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class OperationsRunbookTests(unittest.TestCase):
    def test_runbook_preserves_local_privacy_and_failure_boundaries(self) -> None:
        runbook = (REPO_ROOT / "docs" / "operations" / "runbook.md").read_text(encoding="utf-8")
        normalized = " ".join(runbook.split())
        required = (
            "127.0.0.1:8000",
            "/health/live",
            "/health/ready",
            "/v1/predict",
            "/v1/metrics",
            "/v1/drift",
            "Explicit local operations smoke-test fixture; not research or evaluation data.",
            "Drift remains `insufficient_data` below 760 successful prediction items.",
            "Do not serve an uncalibrated fallback.",
            "There is no hot reload or approved version 1 promotion command.",
            "no authentication or TLS",
            "This is a real local smoke result, not fabricated production traffic",
        )
        for statement in required:
            with self.subTest(statement=statement):
                self.assertIn(statement, normalized)


if __name__ == "__main__":
    unittest.main()
