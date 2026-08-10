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
from authentitext.modeling.frozen_test import (
    FrozenTestError,
    evaluate_frozen_test,
    verify_frozen_test,
)

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "data" / "interim" / "test-frozen"
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
        ("alpha_human", 0, "A personal story from a person"),
        ("alpha_human", 0, "My own experience and opinion"),
        ("beta_human", 0, "A natural review from yesterday"),
        ("beta_human", 0, "I remember this event clearly"),
        ("alpha_machine_continuation_model", 1, "Generated response with formal structure"),
        ("alpha_machine_continuation_model", 1, "Synthetic answer with explanation"),
        ("beta_machine_topical_model", 1, "Automated result organized into sections"),
        ("beta_machine_topical_model", 1, "Model output presenting a concise summary"),
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


class FrozenTestTests(unittest.TestCase):
    def test_scores_frozen_artifacts_and_recomputes_text_free_predictions(self) -> None:
        with temporary_directory() as root:
            base_path, base_identity, calibration_path, calibration_identity = make_artifacts(root)
            test_path = root / "test.jsonl.gz"
            write_partition(test_path, "test")
            prediction_path = root / "predictions.jsonl.gz"
            report = evaluate_frozen_test(
                test_path=test_path,
                test_identity={"relative_path": test_path.name, "rows": 8},
                base_model_path=base_path,
                base_model_identity=base_identity,
                calibration_path=calibration_path,
                calibration_identity=calibration_identity,
                prediction_path=prediction_path,
                dataset_id="example/test",
                revision="abc123",
            )
            verify_frozen_test(report, prediction_path, calibration_path)

            self.assertEqual(report["metrics"]["policy"]["rows"], 8)
            self.assertEqual(set(report["metrics"]["policy_by_domain"]), {"alpha", "beta"})
            self.assertEqual(
                set(report["metrics"]["machine_outcomes_by_strategy"]),
                {"continuation", "topical"},
            )
            with gzip.open(prediction_path, "rt", encoding="utf-8") as handle:
                rendered = handle.read()
            self.assertNotIn("A personal story", rendered)

    def test_verification_rejects_prediction_tampering(self) -> None:
        with temporary_directory() as root:
            base_path, base_identity, calibration_path, calibration_identity = make_artifacts(root)
            test_path = root / "test.jsonl.gz"
            write_partition(test_path, "test")
            prediction_path = root / "predictions.jsonl.gz"
            report = evaluate_frozen_test(
                test_path=test_path,
                test_identity={"relative_path": test_path.name},
                base_model_path=base_path,
                base_model_identity=base_identity,
                calibration_path=calibration_path,
                calibration_identity=calibration_identity,
                prediction_path=prediction_path,
                dataset_id="example/test",
                revision="abc123",
            )
            prediction_path.write_bytes(prediction_path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(FrozenTestError, "size mismatch"):
                verify_frozen_test(report, prediction_path, calibration_path)


if __name__ == "__main__":
    unittest.main()
