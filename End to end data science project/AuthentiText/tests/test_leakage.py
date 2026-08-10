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

from authentitext.data.cleaning import make_record_id
from authentitext.data.leakage import analyze_processed_data, render_report

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "data" / "interim" / "test-leakage"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def temporary_directory() -> Iterator[Path]:
    path = TEST_TEMP_ROOT / f"case-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def make_record(text: str, partition: str, target: int, source: str) -> dict[str, object]:
    content_id = hashlib.sha256(text.encode()).hexdigest()
    raw_label = "0" if target == 1 else "1"
    return {
        "content_id": content_id,
        "dataset_id": "example/test",
        "partition": partition,
        "raw_label": raw_label,
        "record_id": make_record_id(
            dataset_id="example/test",
            revision="abc123",
            partition=partition,
            source=source,
            raw_label=raw_label,
            content_id=content_id,
        ),
        "revision": "abc123",
        "source": source,
        "target": target,
        "text": text,
    }


def write_partition(path: Path, records: list[dict[str, object]]) -> dict[str, object]:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return {
        "partition": records[0]["partition"],
        "output_relative_path": path.name,
        "output_bytes": path.stat().st_size,
        "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows_written": len(records),
    }


class LeakageTests(unittest.TestCase):
    def test_distinguishes_exact_and_normalization_only_leakage(self) -> None:
        with temporary_directory() as root:
            train = [
                make_record("Exact duplicate", "train", 1, "shared_source"),
                make_record("Hello   WORLD", "train", 0, "shared_source"),
            ]
            validation = [
                make_record("Exact duplicate", "validation", 1, "shared_source"),
                make_record("hello world", "validation", 0, "shared_source"),
            ]
            partitions = [
                write_partition(root / "train.jsonl.gz", train),
                write_partition(root / "validation.jsonl.gz", validation),
            ]
            cleaning_report = {
                "dataset_id": "example/test",
                "revision": "abc123",
                "configuration": {"raw_to_canonical_target": {"0": 1, "1": 0}},
                "partitions": partitions,
                "aggregate": {"rows_written": 4},
            }
            report = analyze_processed_data(cleaning_report, root)

        self.assertEqual(report["validation"]["status"], "pass")
        self.assertEqual(report["exact"]["cross_partition_groups"], 1)
        self.assertEqual(report["exact"]["cross_partition_set_counts"], {"train|validation": 1})
        self.assertEqual(report["normalized_exact"]["cross_partition_groups"], 2)
        self.assertEqual(
            report["normalized_exact"]["normalization_only_cross_partition_groups"],
            1,
        )
        self.assertEqual(report["source_overlap"]["sources_spanning_all_partitions"], 1)

    def test_conflicting_duplicate_targets_are_reported_as_data_findings(self) -> None:
        with temporary_directory() as root:
            train = [make_record("Same", "train", 1, "source")]
            test = [make_record("Same", "test", 0, "source")]
            partitions = [
                write_partition(root / "train.jsonl.gz", train),
                write_partition(root / "test.jsonl.gz", test),
            ]
            cleaning_report = {
                "dataset_id": "example/test",
                "revision": "abc123",
                "configuration": {"raw_to_canonical_target": {"0": 1, "1": 0}},
                "partitions": partitions,
                "aggregate": {"rows_written": 2},
            }
            report = analyze_processed_data(cleaning_report, root)

        self.assertEqual(report["validation"]["status"], "pass")
        self.assertEqual(report["exact"]["conflicting_target_groups"], 1)
        self.assertEqual(report["normalized_exact"]["conflicting_target_groups"], 1)
        self.assertEqual(
            report["exact"]["conflicting_target_partition_set_counts"],
            {"train|test": 1},
        )
        self.assertIn(
            "exact duplicate groups have conflicting targets",
            report["validation"]["warnings"],
        )

    def test_rendering_is_deterministic_and_text_free(self) -> None:
        with temporary_directory() as root:
            records = [make_record("private sample text", "train", 1, "source")]
            partitions = [write_partition(root / "train.jsonl.gz", records)]
            cleaning_report = {
                "dataset_id": "example/test",
                "revision": "abc123",
                "configuration": {"raw_to_canonical_target": {"0": 1, "1": 0}},
                "partitions": partitions,
                "aggregate": {"rows_written": 1},
            }
            report = analyze_processed_data(cleaning_report, root)

        first = render_report(report)
        self.assertEqual(first, render_report(report))
        self.assertNotIn("private sample text", first)


if __name__ == "__main__":
    unittest.main()
