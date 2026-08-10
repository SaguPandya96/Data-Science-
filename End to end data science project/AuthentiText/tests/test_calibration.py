from __future__ import annotations

import json
import shutil
import unittest
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np

from authentitext.data.cleaning import open_deterministic_gzip, sha256_file
from authentitext.modeling.calibration import (
    CalibrationError,
    calibrate_scores,
    fit_calibration_policy,
    policy_metrics,
    verify_calibration_policy,
)

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "data" / "interim" / "test-calibration"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def temporary_directory() -> Iterator[Path]:
    path = TEST_TEMP_ROOT / f"case-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def write_predictions(path: Path) -> dict[str, object]:
    with open_deterministic_gzip(path) as handle:
        for index in range(600):
            target = index % 2
            domain = "alpha" if index % 4 < 2 else "beta"
            source = f"{domain}_human" if target == 0 else f"{domain}_machine_continuation_model"
            offset = (index % 50) / 100
            score = 0.05 + offset if target == 0 else 0.46 + offset
            handle.write(
                json.dumps(
                    {
                        "record_id": f"record-{index:04d}",
                        "score": score,
                        "source": source,
                        "target": target,
                        "whitespace_tokens": 100,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    return {
        "relative_path": path.name,
        "rows": 600,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


class CalibrationTests(unittest.TestCase):
    def test_policy_marks_absent_target_rate_not_estimable(self) -> None:
        metrics = policy_metrics(
            np.asarray([1, 1], dtype=np.int8),
            np.asarray([0.2, 0.8], dtype=np.float64),
            0.25,
            0.75,
        )

        self.assertIsNone(metrics["human_false_machine_rate"])
        self.assertEqual(metrics["machine_false_human_rate"], 0.5)
        self.assertEqual(metrics["target_counts"], {"0": 0, "1": 2})

    def test_fits_three_way_calibration_and_abstention_policy(self) -> None:
        with temporary_directory() as root:
            prediction_path = root / "predictions.jsonl.gz"
            prediction_identity = write_predictions(prediction_path)
            artifact_path = root / "calibration_policy.joblib"
            report = fit_calibration_policy(
                prediction_path=prediction_path,
                prediction_identity=prediction_identity,
                base_model_identity={
                    "relative_path": "base.joblib",
                    "bytes": 10,
                    "sha256": "a" * 64,
                },
                artifact_path=artifact_path,
                dataset_id="example/test",
                revision="abc123",
            )
            verify_calibration_policy(report, prediction_path, artifact_path)

            self.assertEqual(sum(role["rows"] for role in report["roles"].values()), 600)
            self.assertIn(
                report["method_selection"]["selected_method"],
                {"raw", "sigmoid", "isotonic"},
            )
            self.assertLess(
                report["thresholds"]["human_threshold"],
                report["thresholds"]["machine_threshold"],
            )
            self.assertLessEqual(
                report["policy_selection_metrics"]["human_false_machine_rate"], 0.05
            )
            self.assertLessEqual(
                report["policy_selection_metrics"]["machine_false_human_rate"], 0.05
            )
            self.assertEqual(
                set(report["calibration_audit"]["policy_by_domain"]), {"alpha", "beta"}
            )
            payload = joblib.load(artifact_path)
            scores = calibrate_scores(
                payload["calibration_method"],
                payload["calibrator"],
                np.asarray([0.1, 0.9]),
            )
            self.assertTrue(((scores >= 0) & (scores <= 1)).all())

    def test_verification_rejects_artifact_tampering(self) -> None:
        with temporary_directory() as root:
            prediction_path = root / "predictions.jsonl.gz"
            prediction_identity = write_predictions(prediction_path)
            artifact_path = root / "calibration_policy.joblib"
            report = fit_calibration_policy(
                prediction_path=prediction_path,
                prediction_identity=prediction_identity,
                base_model_identity={
                    "relative_path": "base.joblib",
                    "bytes": 10,
                    "sha256": "a" * 64,
                },
                artifact_path=artifact_path,
                dataset_id="example/test",
                revision="abc123",
            )
            artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(CalibrationError, "size mismatch"):
                verify_calibration_policy(report, prediction_path, artifact_path)


if __name__ == "__main__":
    unittest.main()
