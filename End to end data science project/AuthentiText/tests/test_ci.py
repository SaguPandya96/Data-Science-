from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
GIT_ATTRIBUTES = REPO_ROOT / ".gitattributes"


class CIConfigurationTests(unittest.TestCase):
    def test_evidence_line_endings_and_binary_artifacts_are_declared(self) -> None:
        attributes = GIT_ATTRIBUTES.read_text(encoding="utf-8")
        for declaration in (
            "*.json text eol=lf",
            "*.jsonl text eol=lf",
            "*.md text eol=lf",
            "*.py text eol=lf",
            "*.gz binary",
            "*.joblib binary",
        ):
            with self.subTest(declaration=declaration):
                self.assertIn(declaration, attributes)

    def test_workflow_is_read_only_and_runs_documented_checks(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        required = (
            "push:",
            "pull_request:",
            "permissions:\n  contents: read",
            "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1",
            "persist-credentials: false",
            "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0",
            'python-version: "3.14"',
            "requirements/runtime.lock",
            "requirements/dev.lock",
            "python -m pip check",
            "python scripts/build_experiment_registry.py --check",
            "python scripts/check_committed_metadata.py",
            "python scripts/check_model_card.py",
            "python scripts/check_documentation.py",
            "python -m ruff check .",
            "python -m ruff format --check .",
            'python -m unittest discover -s tests -p "test_*.py"',
            "python -m pip wheel --no-deps --wheel-dir build/wheels .",
            "docker build --tag authentitext:ci .",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, workflow)

        forbidden = (
            "pull_request_target:",
            "write-all",
            "secrets.",
            "download_data.py",
            "train_baselines.py",
            "evaluate_frozen_test.py",
            "build_drift_reference.py",
        )
        for fragment in forbidden:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, workflow)

    def test_committed_metadata_checker_passes(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/check_committed_metadata.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("schemas, validation markers, and frozen model links", result.stdout)

    def test_model_card_checker_passes(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/check_model_card.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("model-card evidence rows", result.stdout)


if __name__ == "__main__":
    unittest.main()
