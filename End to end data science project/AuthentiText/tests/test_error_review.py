from __future__ import annotations

import gzip
import json
import shutil
import unittest
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from authentitext.data.cleaning import open_deterministic_gzip, sha256_file
from authentitext.modeling.error_review import build_error_review, render_report

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "data" / "interim" / "test-error-review"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def temporary_directory() -> Iterator[Path]:
    path = TEST_TEMP_ROOT / f"case-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with open_deterministic_gzip(path) as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


class ErrorReviewTests(unittest.TestCase):
    def test_selects_stratified_records_and_finalizes_text_free_annotations(self) -> None:
        prediction_records = [
            {
                "record_id": "human-a",
                "source": "alpha_human",
                "target": 0,
                "category": "likely_machine",
                "calibrated_score": 0.9,
                "raw_score": 0.8,
                "whitespace_tokens": 4,
            },
            {
                "record_id": "human-b",
                "source": "beta_human",
                "target": 0,
                "category": "likely_machine",
                "calibrated_score": 0.8,
                "raw_score": 0.7,
                "whitespace_tokens": 4,
            },
            {
                "record_id": "machine-x",
                "source": "alpha_machine_original_model-x",
                "target": 1,
                "category": "likely_human",
                "calibrated_score": 0.1,
                "raw_score": 0.2,
                "whitespace_tokens": 4,
            },
            {
                "record_id": "machine-y",
                "source": "beta_machine_prompt1_model-y",
                "target": 1,
                "category": "likely_human",
                "calibrated_score": 0.2,
                "raw_score": 0.3,
                "whitespace_tokens": 4,
            },
            {
                "record_id": "uncertain-human-a",
                "source": "alpha_human",
                "target": 0,
                "category": "uncertain",
                "calibrated_score": 0.7,
                "raw_score": 0.6,
                "whitespace_tokens": 4,
            },
            {
                "record_id": "uncertain-machine-b",
                "source": "beta_machine_writing_model-y",
                "target": 1,
                "category": "uncertain",
                "calibrated_score": 0.3,
                "raw_score": 0.4,
                "whitespace_tokens": 4,
            },
        ]
        with temporary_directory() as root:
            predictions = root / "predictions.jsonl.gz"
            write_jsonl(predictions, prediction_records)
            external_records = [
                {
                    "partition": "test",
                    "record_id": record["record_id"],
                    "text": f"Private review text for {record['record_id']}",
                }
                for record in prediction_records
            ]
            external = root / "external.jsonl.gz"
            write_jsonl(external, external_records)
            evaluation_report = {
                "dataset_id": "external/test",
                "revision": "abc123",
                "input": {
                    "relative_path": external.name,
                    "rows": len(external_records),
                    "bytes": external.stat().st_size,
                    "sha256": sha256_file(external),
                },
                "predictions": {
                    "relative_path": predictions.name,
                    "rows": len(prediction_records),
                    "bytes": predictions.stat().st_size,
                    "sha256": sha256_file(predictions),
                },
                "validation": {
                    "status": "pass",
                    "external_outcomes_used_for_retuning": False,
                },
            }
            packet = root / "packet.jsonl.gz"
            preview, template = build_error_review(
                evaluation_report=evaluation_report,
                evaluation_report_identity={"relative_path": "evaluation.json"},
                external_path=external,
                prediction_path=predictions,
                packet_path=packet,
                costly_rows_per_group=1,
                uncertain_rows_per_group=1,
            )
            annotations = [
                {
                    "record_id": item["record_id"],
                    "observed_cues": ["no_clear_surface_cue"],
                    "reviewer_note": "No single declared surface cue dominates the local excerpt.",
                }
                for item in template
            ]
            report, _ = build_error_review(
                evaluation_report=evaluation_report,
                evaluation_report_identity={"relative_path": "evaluation.json"},
                external_path=external,
                prediction_path=predictions,
                packet_path=packet,
                annotations=annotations,
                costly_rows_per_group=1,
                uncertain_rows_per_group=1,
            )

            self.assertEqual(preview["validation"]["status"], "awaiting_review")
            self.assertEqual(report["validation"]["status"], "pass")
            self.assertEqual(report["sample"]["rows"], 6)
            self.assertEqual(report["annotations"]["cue_counts"], {"no_clear_surface_cue": 6})
            rendered = render_report(report)
            self.assertNotIn("Private review text", rendered)
            with gzip.open(packet, "rt", encoding="utf-8") as handle:
                self.assertIn("Private review text", handle.read())


if __name__ == "__main__":
    unittest.main()
