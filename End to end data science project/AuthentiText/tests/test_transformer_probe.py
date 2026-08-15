from __future__ import annotations

import unittest
from pathlib import Path

from authentitext.modeling.transformer_probe import (
    BATCH_SIZE,
    HOSTED_RUNNER_LIMIT_SECONDS,
    MAX_LENGTH,
    MEASURED_STEPS,
    MODEL_ID,
    MODEL_REVISION,
    PLANNED_EPOCHS,
    PROBE_ROWS,
    SEED,
    WARMUP_STEPS,
    estimate_training_seconds,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TransformerProbeTests(unittest.TestCase):
    def test_protocol_is_prespecified(self) -> None:
        self.assertEqual(MODEL_ID, "google/bert_uncased_L-2_H-128_A-2")
        self.assertEqual(MODEL_REVISION, "30b0a37ccaaa32f332884b96992754e246e48c5f")
        self.assertEqual(SEED, 1729)
        self.assertEqual(MAX_LENGTH, 128)
        self.assertEqual(BATCH_SIZE, 32)
        self.assertEqual(PROBE_ROWS, 2048)
        self.assertEqual(WARMUP_STEPS, 4)
        self.assertEqual(MEASURED_STEPS, 60)
        self.assertEqual(PLANNED_EPOCHS, 3)
        self.assertLessEqual((WARMUP_STEPS + MEASURED_STEPS) * BATCH_SIZE, PROBE_ROWS)

    def test_estimate_applies_overhead_and_runner_limit(self) -> None:
        viable = estimate_training_seconds(100.0)
        slow = estimate_training_seconds(10.0)

        self.assertTrue(viable["full_run_viable"])
        self.assertFalse(slow["full_run_viable"])
        self.assertEqual(viable["hosted_runner_limit_seconds"], HOSTED_RUNNER_LIMIT_SECONDS)
        self.assertGreater(viable["estimated_total_seconds"], viable["estimated_training_seconds"])

    def test_workflow_never_acquires_or_names_test_data(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "transformer-probe.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("--file train", workflow)
        self.assertIn("requirements/transformer.lock", workflow)
        self.assertNotIn("--file test", workflow)
        self.assertNotIn("test.jsonl", workflow)
        self.assertNotIn("mage/test.csv", workflow)


if __name__ == "__main__":
    unittest.main()
