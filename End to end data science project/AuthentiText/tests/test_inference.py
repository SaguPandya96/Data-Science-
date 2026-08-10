from __future__ import annotations

import json
import shutil
import unittest
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import joblib

from authentitext.data.cleaning import open_deterministic_gzip, sha256_file
from authentitext.inference import (
    MAX_CHARACTERS,
    AuthentiTextPredictor,
    PredictionError,
)
from authentitext.modeling.baselines import train_baselines

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "data" / "interim" / "test-inference"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def temporary_directory() -> Iterator[Path]:
    path = TEST_TEMP_ROOT / f"case-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def make_predictor(root: Path) -> tuple[AuthentiTextPredictor, Path]:
    train_path = root / "train.jsonl.gz"
    examples = [
        ("A personal story written by a person", 0),
        ("My own experience and opinion", 0),
        ("A natural review from yesterday", 0),
        ("I remember this event clearly", 0),
        ("Generated response with formal structure", 1),
        ("Synthetic answer with repeated explanation", 1),
        ("Automated result organized into sections", 1),
        ("Model output presenting a concise summary", 1),
    ]
    with open_deterministic_gzip(train_path) as handle:
        for index, (text, target) in enumerate(examples):
            handle.write(
                json.dumps(
                    {
                        "partition": "train",
                        "record_id": str(index),
                        "source": "fixture",
                        "target": target,
                        "text": text,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    artifact_root = root / "artifacts"
    training_report = train_baselines(
        train_path=train_path,
        artifact_root=artifact_root,
        dataset_id="example/test",
        revision="abc123",
        input_sha256=sha256_file(train_path),
        tfidf_config={"min_df": 1, "max_df": 1.0, "max_features": 100},
        logistic_config={"solver": "liblinear", "max_iter": 100},
    )
    base_identity = next(
        item for item in training_report["artifacts"] if item["model_type"] == "word_tfidf_logistic"
    )
    calibration_path = artifact_root / "calibration_policy.joblib"
    joblib.dump(
        {
            "artifact_version": 1,
            "model_type": "calibration_policy",
            "base_model_type": "word_tfidf_logistic",
            "base_model_sha256": base_identity["sha256"],
            "calibration_method": "raw",
            "calibrator": None,
            "human_threshold": 0.25,
            "machine_threshold": 0.75,
        },
        calibration_path,
        compress=3,
    )
    calibration_report = {
        "artifact": {
            "relative_path": calibration_path.name,
            "bytes": calibration_path.stat().st_size,
            "sha256": sha256_file(calibration_path),
        }
    }
    training_report_path = root / "training.json"
    calibration_report_path = root / "calibration.json"
    training_report_path.write_text(json.dumps(training_report), encoding="utf-8")
    calibration_report_path.write_text(json.dumps(calibration_report), encoding="utf-8")
    predictor = AuthentiTextPredictor.from_reports(
        training_report_path=training_report_path,
        calibration_report_path=calibration_report_path,
        artifact_root=artifact_root,
    )
    return predictor, artifact_root / base_identity["relative_path"]


class InferenceTests(unittest.TestCase):
    def test_returns_versioned_text_free_result_with_low_evidence_warnings(self) -> None:
        with temporary_directory() as root:
            predictor, _ = make_predictor(root)
            submitted = "# Café  example\nhttps://example.invalid"
            result = predictor.predict(submitted)

        self.assertIn(result["category"], {"likely_human", "uncertain", "likely_machine"})
        self.assertGreaterEqual(result["calibrated_machine_likelihood"], 0)
        self.assertLessEqual(result["calibrated_machine_likelihood"], 1)
        self.assertEqual(result["evidence_quality"], "low")
        warning_codes = {warning["code"] for warning in result["warnings"]}
        self.assertEqual(
            warning_codes,
            {
                "short_text_low_evidence",
                "length_outside_development_range",
                "format_out_of_profile",
            },
        )
        self.assertNotIn(submitted, json.dumps(result))
        self.assertEqual(result["model"]["dataset_id"], "example/test")

    def test_rejects_invalid_text(self) -> None:
        with temporary_directory() as root:
            predictor, _ = make_predictor(root)
            cases = [
                (None, "text_type"),
                ("   ", "text_blank"),
                ("bad\0text", "text_null"),
                ("x" * (MAX_CHARACTERS + 1), "text_too_long"),
            ]
            for value, code in cases:
                with self.subTest(code=code), self.assertRaises(PredictionError) as caught:
                    predictor.predict(value)
                self.assertEqual(caught.exception.code, code)

    def test_loading_rejects_tampered_artifact(self) -> None:
        with temporary_directory() as root:
            _, base_path = make_predictor(root)
            base_path.write_bytes(base_path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(PredictionError, "size mismatch"):
                AuthentiTextPredictor.from_reports(
                    training_report_path=root / "training.json",
                    calibration_report_path=root / "calibration.json",
                    artifact_root=root / "artifacts",
                )


if __name__ == "__main__":
    unittest.main()
