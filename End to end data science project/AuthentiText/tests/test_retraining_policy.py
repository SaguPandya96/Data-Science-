from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class RetrainingPolicyTests(unittest.TestCase):
    def test_policy_preserves_required_sequence_and_manual_gates(self) -> None:
        policy = (REPO_ROOT / "docs" / "operations" / "retraining.md").read_text(encoding="utf-8")
        stages = (
            "### 1. Provenance intake",
            "### 2. Data validation",
            "### 3. Leakage and contamination analysis",
            "### 4. Versioned roles and split policy",
            "### 5. Candidate training",
            "### 6. Calibration and policy freeze",
            "### 7. Development OOD and robustness evaluation",
            "### 8. Sealed external acceptance",
            "### 9. Candidate comparison and acceptance gates",
            "### 10. Manual promotion",
            "### 11. Rollback and post-promotion observation",
        )
        previous = -1
        for stage in stages:
            with self.subTest(stage=stage):
                location = policy.find(stage)
                self.assertGreater(location, previous)
                previous = location

        self.assertIn("Drift never retrains or promotes a model automatically.", policy)
        self.assertIn("do not adjust it against the external set", policy)
        self.assertIn("There is no approved promotion mechanism in version 1.", policy)
        self.assertIn(
            "it does not complete model retraining, Docker, or\ndeployment phases", policy
        )


if __name__ == "__main__":
    unittest.main()
