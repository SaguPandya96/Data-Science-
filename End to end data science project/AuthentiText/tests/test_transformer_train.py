from __future__ import annotations

import json
import unittest
from pathlib import Path

from authentitext.data.cleaning import open_deterministic_gzip, sha256_file
from authentitext.data.transformer_train import (
    TransformerTrainError,
    load_train_decisions,
    materialize_transformer_train,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_TEMP_ROOT = REPO_ROOT / "data" / "interim"


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_deterministic_gzip(path) as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            )


def _identity(path: Path, rows: int) -> dict[str, object]:
    return {"rows": rows, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


class TransformerTrainTests(unittest.TestCase):
    def test_committed_decisions_are_complete_and_text_free(self) -> None:
        path = REPO_ROOT / "data" / "metadata" / "transformer_train_decisions.json"
        decisions = load_train_decisions(path)

        self.assertEqual(len(decisions["record_ids_to_drop"]), 69)
        self.assertEqual(decisions["expected_output"]["rows"], 287843)
        self.assertNotIn("text", decisions)

    def test_materialization_applies_only_declared_train_exclusions(self) -> None:
        kept_id = "1" * 64
        dropped_id = "2" * 64
        records = [
            {"record_id": kept_id, "partition": "train", "target": 0, "text": "kept"},
            {"record_id": dropped_id, "partition": "train", "target": 1, "text": "drop"},
        ]
        cleaned_path = TEST_TEMP_ROOT / "transformer-train-cleaned-fixture.jsonl.gz"
        expected_path = TEST_TEMP_ROOT / "transformer-train-expected-fixture.jsonl.gz"
        output_path = TEST_TEMP_ROOT / "transformer-train-output-fixture.jsonl.gz"
        for path in (cleaned_path, expected_path, output_path):
            self.addCleanup(path.unlink, missing_ok=True)
        _write_records(cleaned_path, records)
        _write_records(expected_path, [records[0]])
        decisions = {
            "schema_version": 1,
            "dataset_id": "fixture",
            "revision": "fixture-revision",
            "partition": "train",
            "record_ids_to_drop": [dropped_id],
            "expected_input": _identity(cleaned_path, 2),
            "expected_output": _identity(expected_path, 1),
        }

        report = materialize_transformer_train(cleaned_path, output_path, decisions)

        self.assertEqual(report["rows_written"], 1)
        self.assertEqual(report["rows_dropped"], 1)
        self.assertFalse(report["test_data_read"])
        self.assertEqual(output_path.read_bytes(), expected_path.read_bytes())

    def test_materialization_rejects_missing_exclusion(self) -> None:
        record_id = "3" * 64
        records = [{"record_id": record_id, "partition": "train", "target": 0, "text": "kept"}]
        cleaned_path = TEST_TEMP_ROOT / "transformer-train-missing-cleaned-fixture.jsonl.gz"
        expected_path = TEST_TEMP_ROOT / "transformer-train-missing-expected-fixture.jsonl.gz"
        output_path = TEST_TEMP_ROOT / "transformer-train-missing-output-fixture.jsonl.gz"
        for path in (cleaned_path, expected_path, output_path):
            self.addCleanup(path.unlink, missing_ok=True)
        _write_records(cleaned_path, records)
        _write_records(expected_path, records)
        decisions = {
            "schema_version": 1,
            "dataset_id": "fixture",
            "revision": "fixture-revision",
            "partition": "train",
            "record_ids_to_drop": ["4" * 64],
            "expected_input": _identity(cleaned_path, 1),
            "expected_output": _identity(expected_path, 1),
        }

        with self.assertRaisesRegex(TransformerTrainError, "exclusion"):
            materialize_transformer_train(cleaned_path, output_path, decisions)


if __name__ == "__main__":
    unittest.main()
