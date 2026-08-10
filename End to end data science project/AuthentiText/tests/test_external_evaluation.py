from __future__ import annotations

import gzip
import json
import shutil
import unittest
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from authentitext.data.cleaning import sha256_file
from authentitext.modeling.external_evaluation import (
    evaluate_ghostbuster_external,
    verify_ghostbuster_external,
)
from tests.test_frozen_test import make_artifacts, write_partition

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "data" / "interim" / "test-external-eval"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def temporary_directory() -> Iterator[Path]:
    path = TEST_TEMP_ROOT / f"case-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


class ExternalEvaluationTests(unittest.TestCase):
    def test_scores_only_overlap_gated_ids_and_verifies_metrics(self) -> None:
        with temporary_directory() as root:
            base_path, base_identity, calibration_path, calibration_identity = make_artifacts(root)
            external_path = root / "external.jsonl.gz"
            write_partition(external_path, "test")
            excluded_id = "test:0"
            overlap_report = {
                "datasets": {
                    "external": {
                        "rows": 8,
                        "sha256": sha256_file(external_path),
                    }
                },
                "cross_dataset_exact": {"pairs": 0},
                "cross_dataset_normalized": {"pairs": 0},
                "cross_dataset_near": {"confirmed_pairs": 0},
                "exclusions": {
                    "external_records": 1,
                    "external_rows_after_exclusion": 7,
                    "records": [
                        {
                            "external_record_id": excluded_id,
                            "reasons": ["internal_normalized_duplicate"],
                        }
                    ],
                },
                "validation": {
                    "status": "pass",
                    "model_scoring_performed": False,
                    "outcome_data_used_to_define_exclusions": False,
                    "all_confirmed_overlap_resolved_by_external_exclusion": True,
                },
            }
            overlap_path = root / "overlap.json"
            overlap_path.write_text(json.dumps(overlap_report), encoding="utf-8")
            prediction_path = root / "predictions.jsonl.gz"
            report = evaluate_ghostbuster_external(
                test_path=external_path,
                test_identity={
                    "relative_path": external_path.name,
                    "rows": 8,
                    "bytes": external_path.stat().st_size,
                    "sha256": sha256_file(external_path),
                },
                overlap_report=overlap_report,
                overlap_identity={
                    "relative_path": overlap_path.name,
                    "bytes": overlap_path.stat().st_size,
                    "sha256": sha256_file(overlap_path),
                },
                base_model_path=base_path,
                base_model_identity=base_identity,
                calibration_path=calibration_path,
                calibration_identity=calibration_identity,
                prediction_path=prediction_path,
                dataset_id="external/test",
                revision="external-revision",
            )
            verify_ghostbuster_external(
                report,
                prediction_path=prediction_path,
                calibration_path=calibration_path,
                external_path=external_path,
                overlap_report=overlap_report,
                overlap_path=overlap_path,
            )

            self.assertEqual(report["predictions"]["rows"], 7)
            self.assertEqual(report["selection"]["excluded_rows"], 1)
            with gzip.open(prediction_path, "rt", encoding="utf-8") as handle:
                prediction_ids = {json.loads(line)["record_id"] for line in handle}
            self.assertNotIn(excluded_id, prediction_ids)


if __name__ == "__main__":
    unittest.main()
