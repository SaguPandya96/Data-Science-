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
from authentitext.data.near_duplicates import audit_near_duplicates, render_report

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "data" / "interim" / "test-near"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def temporary_directory() -> Iterator[Path]:
    path = TEST_TEMP_ROOT / f"case-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def make_record(text: str, partition: str, source: str, target: int = 1) -> dict[str, object]:
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


class NearDuplicateTests(unittest.TestCase):
    def test_confirms_high_overlap_without_counting_normalized_equals(self) -> None:
        base_tokens = [f"word{index}" for index in range(50)]
        changed_tokens = list(base_tokens)
        changed_tokens[25] = "replacement"
        base = " ".join(base_tokens)
        changed = " ".join(changed_tokens)
        with temporary_directory() as root:
            train = [make_record(base, "train", "source_a")]
            validation = [
                make_record(changed, "validation", "source_a"),
                make_record(base.upper(), "validation", "source_b"),
            ]
            partitions = [
                write_partition(root / "train.jsonl.gz", train),
                write_partition(root / "validation.jsonl.gz", validation),
            ]
            cleaning_report = {
                "dataset_id": "example/test",
                "revision": "abc123",
                "partitions": partitions,
                "aggregate": {"rows_written": 3},
            }
            report = audit_near_duplicates(
                cleaning_report,
                root,
                per_source=2,
                similarity_threshold=0.8,
            )

        self.assertEqual(report["validation"]["status"], "pass")
        self.assertEqual(report["sample"]["rows"], 3)
        self.assertEqual(report["high_overlap_findings"]["confirmed_pairs"], 1)
        self.assertEqual(len(report["high_overlap_findings"]["confirmed_edges"]), 1)
        self.assertEqual(report["high_overlap_findings"]["groups"], 1)
        self.assertEqual(
            report["high_overlap_findings"]["pair_partition_counts"],
            {"train|validation": 1},
        )
        self.assertGreaterEqual(
            report["candidate_generation"]["normalized_equal_candidate_pairs_excluded"],
            1,
        )

    def test_report_is_deterministic_and_text_free(self) -> None:
        text = "private " + " ".join(f"token{index}" for index in range(20))
        with temporary_directory() as root:
            records = [make_record(text, "train", "source")]
            partitions = [write_partition(root / "train.jsonl.gz", records)]
            cleaning_report = {
                "dataset_id": "example/test",
                "revision": "abc123",
                "partitions": partitions,
                "aggregate": {"rows_written": 1},
            }
            report = audit_near_duplicates(cleaning_report, root, per_source=1)

        rendered = render_report(report)
        self.assertEqual(rendered, render_report(report))
        self.assertNotIn(text, rendered)


if __name__ == "__main__":
    unittest.main()
