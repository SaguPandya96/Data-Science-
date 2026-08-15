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

from authentitext.data.cleaning import open_deterministic_gzip, sha256_file
from authentitext.data.splitting import (
    SplitError,
    create_id_split,
    verify_id_split,
)

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "data" / "interim" / "test-splitting"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def temporary_directory() -> Iterator[Path]:
    path = TEST_TEMP_ROOT / f"case-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def make_record(partition: str, name: str, text: str, target: int) -> dict[str, object]:
    record_id = hashlib.sha256(f"{partition}:{name}".encode()).hexdigest()
    return {
        "content_id": hashlib.sha256(text.encode()).hexdigest(),
        "dataset_id": "example/test",
        "partition": partition,
        "raw_label": str(1 - target),
        "record_id": record_id,
        "revision": "abc123",
        "source": f"source_{name}",
        "target": target,
        "text": text,
    }


def write_inputs(root: Path) -> tuple[dict[str, object], dict[str, str]]:
    records = {
        "train": [
            make_record("train", "shared", " Shared ", 1),
            make_record("train", "conflict_machine", "Conflict", 1),
            make_record("train", "conflict_human", " conflict ", 0),
            make_record("train", "near", "near alpha unique", 1),
            make_record("train", "unique", "train only", 0),
        ],
        "validation": [
            make_record("validation", "shared", "shared", 1),
            make_record("validation", "test_duplicate", "Val Test Duplicate", 0),
            make_record("validation", "near", "near beta unique", 1),
            make_record("validation", "unique", "validation only", 0),
        ],
        "test": [
            make_record("test", "validation_duplicate", " val test duplicate ", 0),
            make_record("test", "unique", "test only", 1),
        ],
    }
    partition_reports = []
    ids: dict[str, str] = {}
    for partition, partition_records in records.items():
        path = root / f"{partition}.jsonl.gz"
        with open_deterministic_gzip(path) as handle:
            for record in partition_records:
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                ids[f"{partition}:{record['source']}"] = str(record["record_id"])
        targets = {
            str(target): sum(record["target"] == target for record in partition_records)
            for target in (0, 1)
        }
        partition_reports.append(
            {
                "partition": partition,
                "output_relative_path": path.name,
                "rows_written": len(partition_records),
                "output_bytes": path.stat().st_size,
                "output_sha256": sha256_file(path),
                "output_target_counts": targets,
            }
        )
    cleaning_report = {
        "dataset_id": "example/test",
        "revision": "abc123",
        "partitions": partition_reports,
        "aggregate": {"rows_written": sum(map(len, records.values()))},
    }
    return cleaning_report, ids


def read_ids(path: Path) -> set[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {json.loads(line)["record_id"] for line in handle}


class SplittingTests(unittest.TestCase):
    def test_groups_overlap_drops_conflicts_and_applies_precedence(self) -> None:
        with temporary_directory() as root:
            input_root = root / "input"
            input_root.mkdir()
            cleaning_report, ids = write_inputs(input_root)
            near_report = {
                "dataset_id": "example/test",
                "revision": "abc123",
                "high_overlap_findings": {
                    "confirmed_pairs": 1,
                    "confirmed_edges": [
                        {
                            "first_record_id": ids["train:source_near"],
                            "second_record_id": ids["validation:source_near"],
                            "similarity": 0.9,
                        }
                    ],
                },
            }
            output_root = root / "output"
            report = create_id_split(cleaning_report, near_report, input_root, output_root)
            verify_id_split(report, output_root)

            output_ids = set().union(
                *(read_ids(output_root / f"{partition}.jsonl.gz") for partition in records_order())
            )
            self.assertIn(ids["train:source_shared"], output_ids)
            self.assertNotIn(ids["validation:source_shared"], output_ids)
            self.assertIn(ids["test:source_validation_duplicate"], output_ids)
            self.assertNotIn(ids["validation:source_test_duplicate"], output_ids)
            self.assertIn(ids["train:source_near"], output_ids)
            self.assertNotIn(ids["validation:source_near"], output_ids)
            self.assertNotIn(ids["train:source_conflict_machine"], output_ids)
            self.assertNotIn(ids["train:source_conflict_human"], output_ids)
            self.assertEqual(report["aggregate"]["input_rows"], 11)
            self.assertEqual(report["aggregate"]["output_rows"], 6)
            self.assertEqual(report["aggregate"]["dropped_rows"], 5)
            self.assertEqual(
                report["aggregate"]["dropped_reason_counts"],
                {"conflicting_target_component": 2, "grouped_overlap": 3},
            )
            self.assertEqual(
                report["aggregate"]["dropped_partition_reason_counts"]["train"],
                {"conflicting_target_component": 2, "grouped_overlap": 0},
            )
            self.assertEqual(report["components"]["confirmed_near_duplicate_edges"], 1)

    def test_outputs_are_deterministic_and_edge_counts_must_reconcile(self) -> None:
        with temporary_directory() as root:
            input_root = root / "input"
            input_root.mkdir()
            cleaning_report, _ = write_inputs(input_root)
            near_report = {
                "dataset_id": "example/test",
                "revision": "abc123",
                "high_overlap_findings": {"confirmed_pairs": 0, "confirmed_edges": []},
            }
            reports = []
            digests = []
            for run in ("first", "second"):
                report = create_id_split(cleaning_report, near_report, input_root, root / run)
                reports.append(report)
                digests.append([output["sha256"] for output in report["outputs"]])
            self.assertEqual(digests[0], digests[1])
            self.assertEqual(reports[0]["aggregate"], reports[1]["aggregate"])

            bad_report = {
                **near_report,
                "high_overlap_findings": {"confirmed_pairs": 1, "confirmed_edges": []},
            }
            with self.assertRaisesRegex(SplitError, "does not reconcile"):
                create_id_split(cleaning_report, bad_report, input_root, root / "bad")


def records_order() -> tuple[str, ...]:
    return ("train", "validation", "test")


if __name__ == "__main__":
    unittest.main()
