from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_experiment_registry

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "data" / "metadata" / "experiment_registry.json"
LOG_PATH = REPO_ROOT / "docs" / "EXPERIMENT_LOG.md"


def _sha256(path: Path) -> str:
    text = path.read_bytes().decode("utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ExperimentRegistryTests(unittest.TestCase):
    def test_report_hash_is_independent_of_checkout_line_endings(self) -> None:
        expected = hashlib.sha256(b"first\nsecond\n").hexdigest()
        with patch.object(Path, "read_bytes", return_value=b"first\r\nsecond\r\n"):
            actual = build_experiment_registry._canonical_text_sha256(Path("report.json"))
        self.assertEqual(actual, expected)

    def test_registry_is_deterministic_and_hash_linked(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/build_experiment_registry.py", "--check"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(registry["validation"]["status"], "pass")
        self.assertEqual(len(registry["experiments"]), 9)
        for experiment in registry["experiments"]:
            with self.subTest(experiment=experiment["experiment_id"]):
                self.assertEqual(experiment["status"], "complete")
                self.assertRegex(experiment["milestone_commit"], re.compile(r"^[0-9a-f]{7}$"))
                report = REPO_ROOT / experiment["report"]["relative_path"]
                self.assertTrue(report.is_file())
                self.assertEqual(_sha256(report), experiment["report"]["sha256"])

    def test_log_indexes_every_completed_and_unrun_entry(self) -> None:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        log = LOG_PATH.read_text(encoding="utf-8")
        for experiment in registry["experiments"]:
            with self.subTest(experiment=experiment["experiment_id"]):
                self.assertIn(f"`{experiment['experiment_id']}`", log)
                self.assertIn(f"`{experiment['milestone_commit']}`", log)
                self.assertIn(Path(experiment["report"]["relative_path"]).name, log)
        for experiment in registry["not_run"]:
            with self.subTest(not_run=experiment["experiment_id"]):
                self.assertIn(f"`{experiment['experiment_id']}`", log)


if __name__ == "__main__":
    unittest.main()
