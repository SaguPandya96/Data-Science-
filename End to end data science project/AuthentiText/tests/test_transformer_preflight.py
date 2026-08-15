from __future__ import annotations

import json
import shutil
import unittest
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from authentitext.modeling.transformer_preflight import (
    MODEL_REVISION,
    TransformerPreflightError,
    build_preflight_report,
    sha256_file,
    verify_preflight_report,
)

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "data" / "interim" / "test-preflight"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def temporary_repo() -> Iterator[Path]:
    root = TEST_TEMP_ROOT / f"case-{uuid.uuid4().hex}"
    train_dir = root / "data" / "processed" / "mage_splits" / "id"
    metadata_dir = root / "data" / "metadata"
    train_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    train_path = train_dir / "train.jsonl.gz"
    train_path.write_bytes(b"fixture training bytes")
    report = {
        "dataset_id": "example/mage",
        "revision": "abc123",
        "outputs": [
            {
                "partition": "train",
                "relative_path": train_path.name,
                "rows": 8,
                "bytes": train_path.stat().st_size,
                "sha256": sha256_file(train_path),
            },
            {
                "partition": "test",
                "relative_path": "missing-sealed-test.jsonl.gz",
                "rows": 4,
                "bytes": 100,
                "sha256": "f" * 64,
            },
        ],
    }
    (metadata_dir / "mage_id_split_report.json").write_text(json.dumps(report), encoding="utf-8")
    try:
        yield root
    finally:
        shutil.rmtree(root)


class TransformerPreflightTests(unittest.TestCase):
    def test_ready_report_uses_only_verified_training_input(self) -> None:
        with temporary_repo() as root:
            report = build_preflight_report(
                root,
                python_version=(3, 11, 9),
                logical_cpus=8,
                total_memory_bytes=16 * 1024**3,
                free_disk_bytes=10 * 1024**3,
                dependency_versions={
                    "torch": "2.9.0",
                    "transformers": "5.0.0",
                    "tokenizers": "0.22.0",
                    "accelerate": "1.11.0",
                },
                model_revision_cached=True,
                generated_at="2026-08-15T00:00:00+00:00",
            )
            verify_preflight_report(report)

            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["candidate"]["revision"], MODEL_REVISION)
            self.assertEqual(report["training_input"]["partition"], "train")
            self.assertFalse(report["training_input"]["test_data_read"])
            self.assertFalse(report["validation"]["transformer_trained"])

    def test_missing_dependencies_and_weights_are_explicit_blockers(self) -> None:
        with temporary_repo() as root:
            report = build_preflight_report(
                root,
                python_version=(3, 11, 9),
                logical_cpus=8,
                total_memory_bytes=16 * 1024**3,
                free_disk_bytes=10 * 1024**3,
                dependency_versions={
                    name: None for name in ("torch", "transformers", "tokenizers", "accelerate")
                },
                model_revision_cached=False,
            )

            self.assertEqual(report["status"], "not_ready")
            self.assertEqual(
                report["blockers"],
                ["dependencies_installed", "model_revision_cached"],
            )

    def test_tampered_training_input_is_rejected(self) -> None:
        with temporary_repo() as root:
            train_path = root / "data" / "processed" / "mage_splits" / "id" / "train.jsonl.gz"
            train_path.write_bytes(b"tampered but same length!!")
            with self.assertRaises(TransformerPreflightError):
                build_preflight_report(root)

    def test_verifier_rejects_completion_claim(self) -> None:
        with temporary_repo() as root:
            report = build_preflight_report(root, model_revision_cached=False)
            report["validation"]["transformer_trained"] = True
            with self.assertRaisesRegex(TransformerPreflightError, "transformer_trained"):
                verify_preflight_report(report)


if __name__ == "__main__":
    unittest.main()
