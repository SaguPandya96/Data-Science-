from __future__ import annotations

import csv
import gzip
import json
import shutil
import unittest
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import joblib

from authentitext.data.cleaning import open_deterministic_gzip
from authentitext.modeling.baselines import sha256_file, train_baselines
from authentitext.modeling.ood_evaluation import evaluate_ood, verify_ood

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "data" / "interim" / "test-ood"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def temporary_directory() -> Iterator[Path]:
    path = TEST_TEMP_ROOT / f"case-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def make_artifacts(root: Path) -> tuple[Path, dict[str, object], Path, dict[str, object]]:
    train_path = root / "train.jsonl.gz"
    examples = [
        ("alpha_human", 0, "A personal recollection from a neighborhood meeting"),
        ("beta_human", 0, "My honest review of the event from yesterday"),
        ("alpha_human", 0, "I remember how the discussion began that afternoon"),
        ("beta_human", 0, "A naturally written account with an individual opinion"),
        ("alpha_machine_topical_model", 1, "Generated structured summary with formal details"),
        ("beta_machine_topical_model", 1, "Automated response organized into clear sections"),
        ("alpha_machine_topical_model", 1, "Synthetic explanation presenting concise results"),
        ("beta_machine_topical_model", 1, "Model output with systematic language and conclusions"),
    ]
    with open_deterministic_gzip(train_path) as handle:
        for index, (source, target, text) in enumerate(examples):
            handle.write(
                json.dumps(
                    {
                        "partition": "train",
                        "record_id": f"train:{index}",
                        "source": source,
                        "target": target,
                        "text": text,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    artifact_root = root / "artifacts"
    training = train_baselines(
        train_path=train_path,
        artifact_root=artifact_root,
        dataset_id="example/test",
        revision="abc123",
        input_sha256=sha256_file(train_path),
        tfidf_config={"min_df": 1, "max_df": 1.0, "max_features": 100},
        logistic_config={"solver": "liblinear", "max_iter": 100},
    )
    base_identity = next(
        item for item in training["artifacts"] if item["model_type"] == "word_tfidf_logistic"
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


def write_ood_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "src"])
        writer.writeheader()
        writer.writerows(rows)


class OodEvaluationTests(unittest.TestCase):
    def test_scores_and_verifies_text_free_ood_predictions(self) -> None:
        with temporary_directory() as root:
            base_path, base_identity, calibration_path, calibration_identity = make_artifacts(root)
            raw_root = root / "raw"
            human_one = "A human review describing a memorable film and its characters"
            human_two = "A personal news summary written after reading the original report"
            files = {
                "gpt_ood": [
                    {"text": human_one, "label": "1", "src": "imdb_human"},
                    {"text": human_two, "label": "1", "src": "cnn_human"},
                    {
                        "text": "Generated film review with organized observations",
                        "label": "0",
                        "src": "imdb_gpt4",
                    },
                    {
                        "text": "Generated news summary with structured details",
                        "label": "0",
                        "src": "cnn_gpt4",
                    },
                ],
                "paraphrase_ood": [
                    {"text": human_one, "label": "1", "src": "imdb_human"},
                    {"text": human_two, "label": "1", "src": "cnn_human"},
                    {
                        "text": "Paraphrased generated review retaining the main ideas",
                        "label": "0",
                        "src": "imdb_gpt4_para",
                    },
                    {
                        "text": "Paraphrased generated report with the same facts",
                        "label": "0",
                        "src": "cnn_gpt4_para",
                    },
                    {
                        "text": "Machine paraphrase of an originally human film review",
                        "label": "0",
                        "src": "imdb_human_para",
                    },
                    {
                        "text": "Machine paraphrase of an originally human news summary",
                        "label": "0",
                        "src": "cnn_human_para",
                    },
                ],
            }
            manifest_files = []
            for name, rows in files.items():
                relative_path = f"mage/{name}.csv"
                path = raw_root / relative_path
                write_ood_csv(path, rows)
                manifest_files.append(
                    {
                        "name": name,
                        "relative_path": relative_path,
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "expected_columns": ["text", "label", "src"],
                    }
                )
            prediction_path = root / "predictions.jsonl.gz"
            report = evaluate_ood(
                raw_root=raw_root,
                manifest={
                    "dataset_id": "example/test",
                    "revision": "abc123",
                    "files": manifest_files,
                },
                base_model_path=base_path,
                base_model_identity=base_identity,
                calibration_path=calibration_path,
                calibration_identity=calibration_identity,
                prediction_path=prediction_path,
            )
            verify_ood(report, prediction_path, calibration_path)

            self.assertEqual(report["predictions"]["rows"], 10)
            self.assertEqual(report["content_accounting"]["duplicate_extra_rows"], 2)
            self.assertEqual(report["metrics"]["content_deduplicated_combined"]["rows"], 8)
            self.assertEqual(
                set(report["metrics"]["source_family_outcomes_content_deduplicated"]),
                {"human", "gpt4", "gpt4_para", "human_para"},
            )
            with gzip.open(prediction_path, "rt", encoding="utf-8") as handle:
                rendered = handle.read()
            self.assertNotIn(human_one, rendered)
            self.assertNotIn(human_two, rendered)


if __name__ == "__main__":
    unittest.main()
