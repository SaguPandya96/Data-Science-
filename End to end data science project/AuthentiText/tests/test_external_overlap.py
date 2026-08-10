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

from authentitext.data.external_overlap import audit_external_overlap, render_report
from authentitext.data.leakage import normalize_text

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "data" / "interim" / "test-external-overlap"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def temporary_directory() -> Iterator[Path]:
    path = TEST_TEMP_ROOT / f"case-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def make_record(text: str, record_id: str, partition: str, target: int) -> dict[str, object]:
    return {
        "content_id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "normalized_id": hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest(),
        "partition": partition,
        "record_id": record_id,
        "target": target,
        "text": text,
    }


def write_records(path: Path, records: list[dict[str, object]]) -> tuple[int, str]:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()


class ExternalOverlapTests(unittest.TestCase):
    def test_population_equality_near_overlap_and_exclusions(self) -> None:
        base_tokens = [f"token{index}" for index in range(40)]
        base = " ".join(base_tokens)
        changed_tokens = list(base_tokens)
        changed_tokens[20] = "replacement"
        changed = " ".join(changed_tokens)
        with temporary_directory() as root:
            mage_root = root / "mage"
            mage_root.mkdir()
            external_path = root / "external.jsonl.gz"
            mage_records = [
                make_record("Exact shared document", "mage-exact", "train", 0),
                make_record("normalized shared document", "mage-normalized", "validation", 1),
                make_record(base, "mage-near", "test", 1),
                make_record("!!!", "mage-tokenless", "test", 0),
            ]
            mage_outputs = []
            for partition in ("train", "validation", "test"):
                records = [item for item in mage_records if item["partition"] == partition]
                path = mage_root / f"{partition}.jsonl.gz"
                size, digest = write_records(path, records)
                mage_outputs.append(
                    {
                        "partition": partition,
                        "relative_path": path.name,
                        "rows_written": len(records),
                        "bytes": size,
                        "sha256": digest,
                    }
                )
            external_records = [
                make_record("Exact shared document", "external-exact", "test", 0),
                make_record("NORMALIZED   SHARED DOCUMENT", "external-normalized", "test", 1),
                make_record(changed, "external-near", "test", 1),
                make_record("Independent duplicate", "external-duplicate-a", "test", 0),
                make_record("INDEPENDENT DUPLICATE", "external-duplicate-b", "test", 0),
                make_record("Unrelated external record", "external-unique", "test", 0),
            ]
            external_size, external_digest = write_records(external_path, external_records)
            split_report = {
                "dataset_id": "development",
                "revision": "dev-revision",
                "outputs": mage_outputs,
                "aggregate": {"output_rows": 4},
            }
            external_manifest = {
                "dataset_id": "external",
                "revision": "external-revision",
                "output": {
                    "relative_path": external_path.name,
                    "rows": 6,
                    "bytes": external_size,
                    "sha256": external_digest,
                },
            }

            report = audit_external_overlap(
                split_report,
                mage_root,
                external_manifest,
                external_path,
                shingle_size=3,
                similarity_threshold=0.8,
            )

        self.assertEqual(report["cross_dataset_exact"]["pairs"], 1)
        self.assertEqual(report["cross_dataset_normalized"]["pairs"], 2)
        self.assertEqual(report["cross_dataset_normalized"]["normalization_only_pairs"], 1)
        self.assertEqual(report["cross_dataset_near"]["confirmed_pairs"], 1)
        self.assertEqual(
            report["cross_dataset_near"]["tokenless_development_records_not_blocked"],
            1,
        )
        self.assertEqual(report["external_internal_overlap"]["normalized"]["duplicate_groups"], 1)
        self.assertEqual(report["exclusions"]["external_records"], 4)
        self.assertEqual(report["exclusions"]["external_rows_after_exclusion"], 2)
        self.assertFalse(report["validation"]["model_scoring_performed"])
        rendered = render_report(report)
        self.assertNotIn("Exact shared document", rendered)
        self.assertEqual(rendered, render_report(report))


if __name__ == "__main__":
    unittest.main()
