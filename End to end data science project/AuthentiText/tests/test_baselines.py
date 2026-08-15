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
from authentitext.modeling.baselines import (
    BaselineError,
    length_features,
    train_baselines,
    verify_baselines,
)

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "data" / "interim" / "test-baselines"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def temporary_directory() -> Iterator[Path]:
    path = TEST_TEMP_ROOT / f"case-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def write_training_data(path: Path) -> None:
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
    with open_deterministic_gzip(path) as handle:
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


class BaselineTests(unittest.TestCase):
    def test_trains_and_verifies_three_baselines(self) -> None:
        with temporary_directory() as root:
            train_path = root / "train.jsonl.gz"
            write_training_data(train_path)
            artifact_root = root / "artifacts"
            report = train_baselines(
                train_path=train_path,
                artifact_root=artifact_root,
                dataset_id="example/test",
                revision="abc123",
                input_sha256=sha256_file(train_path),
                tfidf_config={"min_df": 1, "max_df": 1.0, "max_features": 100},
                logistic_config={"solver": "liblinear", "max_iter": 100, "tol": 1e-4},
            )
            verify_baselines(report, artifact_root)

            self.assertEqual(report["input"]["target_counts"], {"0": 4, "1": 4})
            self.assertEqual(
                {artifact["model_type"] for artifact in report["artifacts"]},
                {"majority", "length_logistic", "word_tfidf_logistic"},
            )
            tfidf = next(
                artifact
                for artifact in report["artifacts"]
                if artifact["model_type"] == "word_tfidf_logistic"
            )
            self.assertGreater(tfidf["vocabulary_size"], 0)
            payload = joblib.load(artifact_root / "length_logistic.joblib")
            probabilities = payload["model"].predict_proba(length_features(["sample text"]))
            self.assertEqual(probabilities.shape, (1, 2))
            self.assertTrue(np.isfinite(probabilities).all())

    def test_verification_rejects_artifact_tampering(self) -> None:
        with temporary_directory() as root:
            train_path = root / "train.jsonl.gz"
            write_training_data(train_path)
            artifact_root = root / "artifacts"
            report = train_baselines(
                train_path=train_path,
                artifact_root=artifact_root,
                dataset_id="example/test",
                revision="abc123",
                input_sha256=sha256_file(train_path),
                tfidf_config={"min_df": 1, "max_df": 1.0, "max_features": 100},
                logistic_config={"solver": "liblinear", "max_iter": 100},
            )
            path = artifact_root / "majority.joblib"
            path.write_bytes(path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(BaselineError, "size mismatch"):
                verify_baselines(report, artifact_root)

    def test_length_features_are_log_scaled_and_finite(self) -> None:
        features = length_features(["", "one two"])
        self.assertEqual(features.shape, (2, 2))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features[0].tolist(), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
