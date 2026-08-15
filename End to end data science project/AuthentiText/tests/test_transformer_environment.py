from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class TransformerEnvironmentWorkflowTests(unittest.TestCase):
    def test_direct_dependencies_are_pinned(self) -> None:
        requirements = (REPO_ROOT / "requirements" / "transformer.in").read_text(encoding="utf-8")
        pins = {
            line.strip()
            for line in requirements.splitlines()
            if line.strip() and not line.startswith("#")
        }

        self.assertEqual(
            pins,
            {
                "torch==2.13.0",
                "transformers==5.14.1",
                "accelerate==1.14.0",
            },
        )

    def test_workflow_resolves_without_training_or_test_data(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "transformer-environment.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("python -m pip check", workflow)
        self.assertIn("python -m pip freeze --all", workflow)
        self.assertIn("cat build/transformer-environment/transformer.lock", workflow)
        self.assertIn("actions/upload-artifact@043fb46d", workflow)
        self.assertNotIn("train.jsonl", workflow)
        self.assertNotIn("test.jsonl", workflow)


if __name__ == "__main__":
    unittest.main()
