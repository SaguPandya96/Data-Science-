from __future__ import annotations

import hashlib
import io
import json
import unittest
from pathlib import Path

from scripts.download_runtime_artifacts import (
    ArtifactDownloadError,
    _copy_and_hash,
    _validate_entry,
)


class RuntimeArtifactDownloadTests(unittest.TestCase):
    def _entry(self) -> dict[str, object]:
        content = b"verified artifact"
        return {
            "filename": "model.joblib",
            "url": "https://github.com/example/project/releases/download/v1/model.joblib",
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    def test_copy_records_exact_identity(self) -> None:
        content = b"verified artifact"
        target = io.BytesIO()
        size, digest = _copy_and_hash(io.BytesIO(content), target)
        self.assertEqual(target.getvalue(), content)
        self.assertEqual(size, len(content))
        self.assertEqual(digest, hashlib.sha256(content).hexdigest())

    def test_accepts_well_formed_github_entry(self) -> None:
        filename, url, size, digest = _validate_entry(self._entry())
        self.assertEqual(filename, "model.joblib")
        self.assertEqual(url, self._entry()["url"])
        self.assertEqual(size, self._entry()["bytes"])
        self.assertEqual(digest, self._entry()["sha256"])

    def test_rejects_non_github_download_url(self) -> None:
        entry = self._entry()
        entry["url"] = "https://example.com/model.joblib"
        with self.assertRaisesRegex(ArtifactDownloadError, "approved GitHub URL"):
            _validate_entry(entry)

    def test_committed_manifest_matches_frozen_reports(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (root / "data/metadata/runtime_artifact_manifest.json").read_text(encoding="utf-8")
        )
        training = json.loads(
            (root / "data/metadata/mage_baseline_training_report.json").read_text(encoding="utf-8")
        )
        calibration = json.loads(
            (root / "data/metadata/mage_calibration_report.json").read_text(encoding="utf-8")
        )
        model = next(
            item for item in training["artifacts"] if item["model_type"] == "word_tfidf_logistic"
        )
        entries = {item["filename"]: item for item in manifest["artifacts"]}
        for identity in (model, calibration["artifact"]):
            entry = entries[identity["relative_path"]]
            self.assertEqual(entry["bytes"], identity["bytes"])
            self.assertEqual(entry["sha256"], identity["sha256"])


if __name__ == "__main__":
    unittest.main()
