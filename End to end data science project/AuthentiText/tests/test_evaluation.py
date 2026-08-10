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
from authentitext.modeling.baselines import train_baselines
from authentitext.modeling.evaluation import (
    EvaluationError,
    evaluate_baselines,
    verify_evaluation,
)

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "data" / "interim" / "test-evaluation"
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


def train_fixture(root: Path) -> tuple[dict[str, object], Path]:
    train_path = root / "train.jsonl.gz"
    write_partition(train_path, "train")
    artifact_root = root / "models"
    report = train_baselines(
        train_path=train_path,
        artifact_root=artifact_root,
        dataset_id="example/test",
        revision="abc123",
        input_sha256=sha256_file(train_path),
        tfidf_config={"min_df": 1, "max_df": 1.0, "max_features": 100},
        logistic_config={"solver": "liblinear", "max_iter": 100},
    )
    return report, artifact_root


class EvaluationTests(unittest.TestCase):
    def test_evaluates_and_recomputes_metrics_from_text_free_predictions(self) -> None:
        with temporary_directory() as root:
            training_report, artifact_root = train_fixture(root)
            validation_path = root / "validation.jsonl.gz"
            write_partition(validation_path, "validation")
            prediction_root = root / "predictions"
            report = evaluate_baselines(
                evaluation_path=validation_path,
                evaluation_partition="validation",
                evaluation_identity={
                    "relative_path": validation_path.name,
                    "rows": 8,
                    "bytes": validation_path.stat().st_size,
                    "sha256": sha256_file(validation_path),
                },
                training_report=training_report,
                artifact_root=artifact_root,
                prediction_root=prediction_root,
            )
            verify_evaluation(report, prediction_root)

            majority = next(
                model for model in report["models"] if model["model_type"] == "majority"
            )
            self.assertEqual(
                majority["metrics"]["overall"]["confusion"],
                {"true_negative": 0, "false_positive": 4, "false_negative": 0, "true_positive": 4},
            )
            self.assertEqual(majority["metrics"]["overall"]["roc_auc"], 0.5)
            self.assertEqual(len(majority["metrics"]["calibration"]["bins"]), 15)
            self.assertEqual(set(majority["metrics"]["by_domain"]), {"alpha", "beta"})
            prediction_path = prediction_root / majority["predictions"]["relative_path"]
            with gzip.open(prediction_path, "rt", encoding="utf-8") as handle:
                rendered = handle.read()
            self.assertNotIn("A personal story", rendered)

    def test_verification_rejects_prediction_tampering(self) -> None:
        with temporary_directory() as root:
            training_report, artifact_root = train_fixture(root)
            validation_path = root / "validation.jsonl.gz"
            write_partition(validation_path, "validation")
            prediction_root = root / "predictions"
            report = evaluate_baselines(
                evaluation_path=validation_path,
                evaluation_partition="validation",
                evaluation_identity={"relative_path": validation_path.name},
                training_report=training_report,
                artifact_root=artifact_root,
                prediction_root=prediction_root,
            )
            prediction = report["models"][0]["predictions"]
            path = prediction_root / prediction["relative_path"]
            path.write_bytes(path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(EvaluationError, "size mismatch"):
                verify_evaluation(report, prediction_root)


if __name__ == "__main__":
    unittest.main()
