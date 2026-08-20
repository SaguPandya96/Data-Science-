from __future__ import annotations

import json
import unittest
from pathlib import Path

from authentitext.data.transformer_train import load_partition_decisions

REPO_ROOT = Path(__file__).resolve().parents[1]


class TransformerEvaluationTests(unittest.TestCase):
    def test_candidate_is_hash_frozen_before_test(self) -> None:
        freeze = json.loads(
            (REPO_ROOT / "data/metadata/transformer_candidate_freeze.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertTrue(freeze["policy"]["frozen_before_test"])
        self.assertFalse(freeze["policy"]["retuning_after_test_allowed"])
        self.assertEqual(len(freeze["model_artifact"]["sha256"]), 64)
        self.assertEqual(len(freeze["calibration_artifact"]["sha256"]), 64)
        self.assertEqual(
            freeze["calibration_artifact"]["base_model_sha256"],
            freeze["model_artifact"]["sha256"],
        )
        self.assertEqual(freeze["calibration_artifact"]["method"], "isotonic")

    def test_test_role_is_text_free_and_matches_published_split(self) -> None:
        decisions = load_partition_decisions(
            REPO_ROOT / "data/metadata/transformer_test_decisions.json"
        )

        self.assertEqual(decisions["partition"], "test")
        self.assertEqual(decisions["expected_output"]["rows"], 50567)
        self.assertEqual(
            decisions["expected_output"]["content_sha256"],
            "9e777ff7ea722219729db482ab6308efe76cbf2fdeee46d5db0326bb7e9c0863",
        )
        self.assertNotIn("text", decisions)

    def test_evaluation_workflow_cannot_retrain_or_retune(self) -> None:
        workflow = (REPO_ROOT / ".github/workflows/transformer-evaluation.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("--partition test", workflow)
        self.assertIn("evaluate_transformer.py", workflow)
        self.assertNotIn("train_transformer.py", workflow)
        self.assertNotIn("calibrate_baseline.py", workflow)


if __name__ == "__main__":
    unittest.main()
