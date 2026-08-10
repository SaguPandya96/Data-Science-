from __future__ import annotations

import unittest
from pathlib import Path

from authentitext.inference import CANONICAL_DISCLAIMER, LIMITATIONS

REPO_ROOT = Path(__file__).resolve().parents[1]


class ResponsibleAITests(unittest.TestCase):
    def test_canonical_disclaimer_is_exposed_by_inference_and_interface(self) -> None:
        proof_limitation = next(
            item for item in LIMITATIONS if item["code"] == "not_authorship_proof"
        )
        interface = (REPO_ROOT / "src" / "authentitext" / "web" / "index.html").read_text(
            encoding="utf-8"
        )

        self.assertEqual(proof_limitation["message"], CANONICAL_DISCLAIMER)
        self.assertIn(CANONICAL_DISCLAIMER, interface)
        self.assertIn("should never be used as sole grounds", interface)

    def test_responsible_ai_guide_covers_required_risks(self) -> None:
        guide = (REPO_ROOT / "docs" / "RESPONSIBLE_AI.md").read_text(encoding="utf-8")
        normalized = guide.replace("\n> ", " ")

        self.assertIn(CANONICAL_DISCLAIMER, normalized)
        required_sections = (
            "## False-positive and false-negative risk",
            "## Distribution shift",
            "## Short, long, and mixed text",
            "## Required human review",
            "## Prohibited and out-of-scope uses",
            "## Privacy",
            "## Misuse and incident response",
            "## Current governance gaps",
        )
        for section in required_sections:
            with self.subTest(section=section):
                self.assertIn(section, guide)


if __name__ == "__main__":
    unittest.main()
