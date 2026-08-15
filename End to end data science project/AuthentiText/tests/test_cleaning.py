from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import unittest
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from authentitext.data.cleaning import (
    CleaningError,
    clean_manifest,
    verify_cleaned_outputs,
)

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "data" / "interim" / "test-cleaning"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def temporary_directory() -> Iterator[Path]:
    path = TEST_TEMP_ROOT / f"case-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def make_manifest(payload: bytes) -> dict[str, object]:
    return {
        "dataset_id": "example/test",
        "revision": "abc123",
        "label_semantics": {"canonical_machine_positive_mapping": {"0": 1, "1": 0}},
        "files": [
            {
                "name": "train",
                "relative_path": "example/train.csv",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "expected_columns": ["text", "label", "src"],
            }
        ],
    }


def read_records(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return [json.loads(line) for line in handle]


class CleaningTests(unittest.TestCase):
    def test_excludes_writingprompts_maps_labels_and_preserves_text(self) -> None:
        payload = (
            b"text,label,src\n"
            b"Machine text,0,news_model\n"
            b"Human text,1,news_human\n"
            b"Excluded machine,0,wp_machine_model\n"
            b"Excluded human,1,wp_human\n"
        )
        manifest = make_manifest(payload)
        with temporary_directory() as root:
            input_path = root / "raw" / "example" / "train.csv"
            input_path.parent.mkdir(parents=True)
            input_path.write_bytes(payload)
            output_root = root / "processed"
            report = clean_manifest(manifest, root / "raw", output_root)
            records = read_records(output_root / "train.jsonl.gz")

            self.assertEqual([record["text"] for record in records], ["Machine text", "Human text"])
            self.assertEqual([record["target"] for record in records], [1, 0])
            self.assertEqual(report["aggregate"]["rows_seen"], 4)
            self.assertEqual(report["aggregate"]["rows_written"], 2)
            self.assertEqual(report["aggregate"]["rows_excluded"], 2)
            self.assertEqual(report["aggregate"]["excluded_raw_label_counts"], {"0": 1, "1": 1})
            self.assertEqual(len({record["record_id"] for record in records}), 2)
            self.assertEqual(
                report["partitions"][0]["input_relative_path"],
                "example/train.csv",
            )
            self.assertEqual(
                records[0]["content_id"],
                hashlib.sha256(b"Machine text").hexdigest(),
            )
            verify_cleaned_outputs(report, output_root)

    def test_compressed_output_is_deterministic(self) -> None:
        payload = b"text,label,src\nExample,0,news_model\n"
        manifest = make_manifest(payload)
        reports = []
        outputs = []
        with temporary_directory() as root:
            for run in ("first", "second"):
                input_path = root / run / "raw" / "example" / "train.csv"
                input_path.parent.mkdir(parents=True)
                input_path.write_bytes(payload)
                output_root = root / run / "processed"
                reports.append(clean_manifest(manifest, root / run / "raw", output_root))
                outputs.append((output_root / "train.jsonl.gz").read_bytes())

        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(reports[0], reports[1])

    def test_duplicate_record_identity_is_rejected(self) -> None:
        payload = b"text,label,src\nRepeated,0,news_model\nRepeated,0,news_model\n"
        manifest = make_manifest(payload)
        with temporary_directory() as root:
            input_path = root / "raw" / "example" / "train.csv"
            input_path.parent.mkdir(parents=True)
            input_path.write_bytes(payload)
            output_root = root / "processed"
            with self.assertRaises(CleaningError):
                clean_manifest(manifest, root / "raw", output_root)
            self.assertFalse((output_root / "train.jsonl.gz").exists())
            self.assertFalse((output_root / "train.jsonl.gz.tmp").exists())

    def test_unknown_label_is_rejected(self) -> None:
        payload = b"text,label,src\nExample,2,news_unknown\n"
        manifest = make_manifest(payload)
        with temporary_directory() as root:
            input_path = root / "raw" / "example" / "train.csv"
            input_path.parent.mkdir(parents=True)
            input_path.write_bytes(payload)
            with self.assertRaises(CleaningError):
                clean_manifest(manifest, root / "raw", root / "processed")


if __name__ == "__main__":
    unittest.main()
