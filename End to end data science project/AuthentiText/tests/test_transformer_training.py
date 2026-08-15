from __future__ import annotations

import unittest
from pathlib import Path

from authentitext.modeling.transformer_training import (
    BATCH_SIZE,
    EVALUATION_BATCH_SIZE,
    EXPECTED_TRAIN_CONTENT_SHA256,
    EXPECTED_VALIDATION_CONTENT_SHA256,
    EXPECTED_VALIDATION_ROWS,
    FULL_TRAIN_ROWS,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    MAX_LENGTH,
    MODEL_REVISION,
    PLANNED_EPOCHS,
    SEED,
    WARMUP_RATIO,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TransformerTrainingTests(unittest.TestCase):
    def test_full_protocol_is_frozen_before_training(self) -> None:
        self.assertEqual(SEED, 1729)
        self.assertEqual(MAX_LENGTH, 128)
        self.assertEqual(BATCH_SIZE, 32)
        self.assertEqual(EVALUATION_BATCH_SIZE, 64)
        self.assertEqual(PLANNED_EPOCHS, 3)
        self.assertEqual(LEARNING_RATE, 2e-5)
        self.assertEqual(WARMUP_RATIO, 0.10)
        self.assertEqual(GRADIENT_CLIP_NORM, 1.0)
        self.assertEqual(FULL_TRAIN_ROWS, 287843)
        self.assertEqual(EXPECTED_VALIDATION_ROWS, 50509)
        self.assertEqual(MODEL_REVISION, "30b0a37ccaaa32f332884b96992754e246e48c5f")
        self.assertEqual(len(EXPECTED_TRAIN_CONTENT_SHA256), 64)
        self.assertEqual(len(EXPECTED_VALIDATION_CONTENT_SHA256), 64)

    def test_training_source_never_names_sealed_test_data(self) -> None:
        source = (
            REPO_ROOT / "src" / "authentitext" / "modeling" / "transformer_training.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("test.jsonl", source)
        self.assertNotIn("mage/test.csv", source)

    def test_workflow_acquires_only_modeling_roles(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "transformer-training.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("--file train", workflow)
        self.assertIn("--file validation", workflow)
        self.assertNotIn("--file test", workflow)
        self.assertNotIn("test.jsonl", workflow)


if __name__ == "__main__":
    unittest.main()
