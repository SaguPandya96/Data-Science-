from __future__ import annotations

import json
import shutil
import unittest
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from authentitext.analysis.eda import analyze_split, render_report
from authentitext.data.cleaning import open_deterministic_gzip, sha256_file

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "data" / "interim" / "test-eda"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def temporary_directory() -> Iterator[Path]:
    path = TEST_TEMP_ROOT / f"case-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def write_fixture(root: Path) -> dict[str, object]:
    by_partition = {
        "train": [
            ("domain_human", 0, "Human text with https://example.invalid"),
            ("domain_machine_continuation_model", 1, "# Machine  text\n```code```"),
        ],
        "validation": [
            ("domain_human", 0, " Plain human "),
            ("domain_machine_continuation_model", 1, "Café machine"),
        ],
        "test": [
            ("domain_human", 0, " short "),
            ("domain_machine_continuation_model", 1, "Machine result"),
        ],
    }
    outputs = []
    for partition, records in by_partition.items():
        path = root / f"{partition}.jsonl.gz"
        with open_deterministic_gzip(path) as handle:
            for index, (source, target, text) in enumerate(records):
                handle.write(
                    json.dumps(
                        {
                            "partition": partition,
                            "record_id": f"{partition}:{index}",
                            "source": source,
                            "target": target,
                            "text": text,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        outputs.append(
            {
                "partition": partition,
                "relative_path": path.name,
                "rows_written": len(records),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "dataset_id": "example/test",
        "revision": "abc123",
        "split_name": "id",
        "outputs": outputs,
        "aggregate": {"output_rows": 6},
    }


class EdaTests(unittest.TestCase):
    def test_profiles_lengths_composition_and_predefined_artifacts(self) -> None:
        with temporary_directory() as root:
            report = analyze_split(write_fixture(root), root)

        self.assertEqual(report["population"]["analysis_rows"], 4)
        self.assertEqual(report["population"]["excluded_partition_counts"], {"test": 2})
        self.assertEqual(report["population"]["partition_target_counts"]["train"], {"0": 1, "1": 1})
        self.assertEqual(report["length_groups"]["target:0"]["rows"], 2)
        self.assertEqual(
            report["artifact_indicators"]["contains_url_marker"]["overall"]["count"], 1
        )
        self.assertEqual(
            report["artifact_indicators"]["contains_markdown_fence"]["by_target"]["1"]["count"],
            1,
        )
        self.assertEqual(
            report["artifact_indicators"]["leading_or_trailing_whitespace"]["by_target"]["0"][
                "count"
            ],
            1,
        )
        self.assertEqual(report["population"]["generator_machine_counts"], {"model": 2})
        self.assertNotIn("partition:test", report["length_groups"])

    def test_rendering_is_deterministic_and_text_free(self) -> None:
        with temporary_directory() as root:
            split_report = write_fixture(root)
            first = render_report(analyze_split(split_report, root))
            second = render_report(analyze_split(split_report, root))
        self.assertEqual(first, second)
        self.assertNotIn("Human text with", first)
        self.assertNotIn('"train:0"', first)


if __name__ == "__main__":
    unittest.main()
