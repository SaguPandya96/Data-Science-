from __future__ import annotations

import json
import shutil
import unittest
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from authentitext.data.cleaning import open_deterministic_gzip
from authentitext.modeling.generator_holdouts import (
    GeneratorHoldoutError,
    materialize_generator_role,
    run_generator_holdouts,
    verify_generator_holdouts,
)

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "data" / "interim" / "test-generator"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def temporary_directory() -> Iterator[Path]:
    path = TEST_TEMP_ROOT / f"case-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def write_partition(path: Path, partition: str, rows_per_source: int) -> None:
    with open_deterministic_gzip(path) as handle:
        for domain in ("alpha", "beta"):
            sources = (
                (f"{domain}_human", 0),
                (f"{domain}_machine_continuation_model_a", 1),
                (f"{domain}_machine_topical_model_b", 1),
            )
            for source, target in sources:
                for index in range(rows_per_source):
                    effective_target = 1 - target if index % 8 == 0 else target
                    style = (
                        "personal memory natural opinion"
                        if effective_target == 0
                        else "generated formal synthetic summary"
                    )
                    handle.write(
                        json.dumps(
                            {
                                "partition": partition,
                                "record_id": f"{partition}:{source}:{index:04d}",
                                "source": source,
                                "target": target,
                                "text": f"{style} example number {index}",
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )


def expected_role(rows_per_source: int) -> dict[str, object]:
    return {
        "rows": rows_per_source * 4,
        "target_counts": {"0": rows_per_source * 2, "1": rows_per_source * 2},
        "distinct_sources": 4,
    }


class GeneratorHoldoutTests(unittest.TestCase):
    def test_materializes_metadata_only_selector_and_reconciles_counts(self) -> None:
        with temporary_directory() as root:
            input_path = root / "validation.jsonl.gz"
            output_path = root / "selected.jsonl.gz"
            write_partition(input_path, "validation", 5)
            identity = materialize_generator_role(
                input_path=input_path,
                output_path=output_path,
                partition="validation",
                role="validation",
                held_out_generator="model_a",
                expected=expected_role(5),
            )
            self.assertEqual(identity["rows"], 20)
            self.assertEqual(identity["target_counts"], {"0": 10, "1": 10})

            with self.assertRaisesRegex(GeneratorHoldoutError, "expected 21"):
                materialize_generator_role(
                    input_path=input_path,
                    output_path=output_path,
                    partition="validation",
                    role="validation",
                    held_out_generator="model_a",
                    expected={**expected_role(5), "rows": 21},
                )

    def test_runs_verifies_and_resumes_fold_without_retaining_selected_text(self) -> None:
        with temporary_directory() as root:
            input_root = root / "input"
            input_root.mkdir()
            write_partition(input_root / "train.jsonl.gz", "train", 30)
            write_partition(input_root / "validation.jsonl.gz", "validation", 180)
            write_partition(input_root / "test.jsonl.gz", "test", 20)
            regime_report = {
                "dataset_id": "example/test",
                "revision": "abc123",
                "generator_holdouts": [
                    {
                        "regime_id": "generator:model_a",
                        "held_out_generator": "model_a",
                        "roles": {
                            "train": expected_role(30),
                            "validation": expected_role(180),
                            "test": expected_role(20),
                        },
                    },
                    {
                        "regime_id": "generator:model_b",
                        "held_out_generator": "model_b",
                        "roles": {
                            "train": expected_role(30),
                            "validation": expected_role(180),
                            "test": expected_role(20),
                        },
                    },
                ],
            }
            artifact_root = root / "artifacts"
            work_root = root / "work"
            checkpoints = []
            report = run_generator_holdouts(
                regime_report=regime_report,
                input_root=input_root,
                artifact_root=artifact_root,
                work_root=work_root,
                checkpoint=checkpoints.append,
                tfidf_config={"min_df": 1, "max_df": 1.0, "max_features": 100},
                logistic_config={"solver": "liblinear", "max_iter": 100},
            )
            verify_generator_holdouts(report, artifact_root)

            self.assertEqual(report["validation"]["folds_completed"], 2)
            self.assertEqual(checkpoints[0]["validation"]["status"], "in_progress")
            self.assertEqual(checkpoints[-1]["validation"]["status"], "pass")
            self.assertFalse(
                report["configuration"]["test_data_used_for_training_calibration_or_thresholds"]
            )
            self.assertFalse((work_root / "model_a").exists())
            self.assertFalse((work_root / "model_b").exists())
            prediction_path = artifact_root / "model_a" / "test_predictions.jsonl.gz"
            self.assertTrue(prediction_path.is_file())
            self.assertNotIn("personal memory", prediction_path.read_bytes().decode("latin-1"))

            resumed = run_generator_holdouts(
                regime_report=regime_report,
                input_root=input_root,
                artifact_root=artifact_root,
                work_root=work_root,
                resume_report=report,
            )
            self.assertEqual(resumed, report)

            stale_report = {**report, "revision": "different"}
            with self.assertRaisesRegex(GeneratorHoldoutError, "different datasets"):
                run_generator_holdouts(
                    regime_report=regime_report,
                    input_root=input_root,
                    artifact_root=artifact_root,
                    work_root=work_root,
                    resume_report=stale_report,
                )


if __name__ == "__main__":
    unittest.main()
