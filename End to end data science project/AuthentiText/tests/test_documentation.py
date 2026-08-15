from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class DocumentationTests(unittest.TestCase):
    def test_documentation_links_and_readme_evidence_pass(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/check_documentation.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("README evidence rows", result.stdout)


if __name__ == "__main__":
    unittest.main()
