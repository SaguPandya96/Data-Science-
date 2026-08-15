from __future__ import annotations

import gzip
import json
import shutil
import unittest
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import joblib

from authentitext.data.cleaning import open_deterministic_gzip, sha256_file
from authentitext.modeling.baselines import train_baselines
from authentitext.modeling.truncation_robustness import (
    TruncationRobustnessError,
    evaluate_truncation_robustness,
    truncate_prefix,
    verify_truncation_robustness,
)

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "data" / "interim" / "test-truncation"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def temporary_directory() -> Iterator[Path]:
    path = TEST_TEMP_ROOT / f"case-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def write_partition(path: Path, partition: str) -> None:
    examples = [
        ("alpha_human", 0, "A personal story from one person with several details today"),
        ("alpha_human", 0, "My own experience contains a specific opinion and memory here"),
        ("beta_human", 0, "A natural review explains what happened during yesterday's visit"),
        ("beta_human", 0, "I remember this event clearly and can describe every moment"),
        (
            "alpha_machine_continuation_model",
            1,
            "Generated response uses formal structure and provides several ordered details",
        ),
        (
            "alpha_machine_continuation_model",
            1,
            "Synthetic answer offers an explanation with a concise concluding summary",
        ),
        (
            "beta_machine_topical_model",
            1,
            "Automated result is organized into sections with consistently neutral phrasing",
        ),
        (
            "beta_machine_topical_model",
            1,
            "Model output presents a concise summary and multiple supporting points",
        ),
    ]
    with open_deterministic_gzip(path) as handle:
        for index, (source, target, text) in enumerate(examples):
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


def make_artifacts(root: Path) -> tuple[Path, dict[str, object], Path, dict[str, object]]:
    train_path = root / "train.jsonl.gz"
    write_partition(train_path, "train")
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
    base_path = artifact_root / base_identity["relative_path"]
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
    calibration_identity = {
        "relative_path": calibration_path.name,
        "bytes": calibration_path.stat().st_size,
        "sha256": sha256_file(calibration_path),
    }
    return base_path, base_identity, calibration_path, calibration_identity


class TruncationRobustnessTests(unittest.TestCase):
    def test_truncate_prefix_preserves_the_original_prefix(self) -> None:
        text = "  one\ttwo \nthree   four  "
        self.assertEqual(truncate_prefix(text, 3), "  one\ttwo \nthree")
        self.assertEqual(truncate_prefix(text, 10), text)

    def test_scores_and_recomputes_paired_text_free_predictions(self) -> None:
        with temporary_directory() as root:
            base_path, base_identity, calibration_path, calibration_identity = make_artifacts(root)
            test_path = root / "test.jsonl.gz"
            write_partition(test_path, "test")
            prediction_path = root / "truncation.jsonl.gz"
            report = evaluate_truncation_robustness(
                test_path=test_path,
                test_identity={
                    "relative_path": test_path.name,
                    "rows": 8,
                    "bytes": test_path.stat().st_size,
                    "sha256": sha256_file(test_path),
                },
                base_model_path=base_path,
                base_model_identity=base_identity,
                calibration_path=calibration_path,
                calibration_identity=calibration_identity,
                prediction_path=prediction_path,
                dataset_id="example/test",
                revision="abc123",
                budgets=(3, 5),
            )
            verify_truncation_robustness(report, prediction_path, calibration_path)

            self.assertEqual(report["predictions"]["rows"], 16)
            self.assertEqual([item["selection"]["rows"] for item in report["conditions"]], [8, 8])
            transitions = report["conditions"][0]["paired_effects"]["category_changes"][
                "transitions"
            ]
            self.assertEqual(sum(sum(row.values()) for row in transitions.values()), 8)
            with gzip.open(prediction_path, "rt", encoding="utf-8") as handle:
                rendered = handle.read()
            self.assertNotIn("personal story", rendered)
            self.assertNotIn('"text"', rendered)

    def test_verification_rejects_prediction_tampering(self) -> None:
        with temporary_directory() as root:
            base_path, base_identity, calibration_path, calibration_identity = make_artifacts(root)
            test_path = root / "test.jsonl.gz"
            write_partition(test_path, "test")
            prediction_path = root / "truncation.jsonl.gz"
            report = evaluate_truncation_robustness(
                test_path=test_path,
                test_identity={
                    "relative_path": test_path.name,
                    "rows": 8,
                    "bytes": test_path.stat().st_size,
                    "sha256": sha256_file(test_path),
                },
                base_model_path=base_path,
                base_model_identity=base_identity,
                calibration_path=calibration_path,
                calibration_identity=calibration_identity,
                prediction_path=prediction_path,
                dataset_id="example/test",
                revision="abc123",
                budgets=(3,),
            )
            prediction_path.write_bytes(prediction_path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(TruncationRobustnessError, "size mismatch"):
                verify_truncation_robustness(report, prediction_path, calibration_path)


if __name__ == "__main__":
    unittest.main()
