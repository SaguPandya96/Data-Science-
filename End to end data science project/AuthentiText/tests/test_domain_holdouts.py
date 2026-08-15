from __future__ import annotations

import json
import shutil
import unittest
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from authentitext.data.cleaning import open_deterministic_gzip
from authentitext.modeling.domain_holdouts import (
    DomainHoldoutError,
    materialize_domain_role,
    run_domain_holdouts,
    verify_domain_holdouts,
)

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "data" / "interim" / "test-domain"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def temporary_directory() -> Iterator[Path]:
    path = TEST_TEMP_ROOT / f"case-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def write_partition(path: Path, partition: str, rows_per_cell: int) -> None:
    with open_deterministic_gzip(path) as handle:
        for domain in ("alpha", "beta"):
            for target in (0, 1):
                source = (
                    f"{domain}_human" if target == 0 else f"{domain}_machine_continuation_model"
                )
                for index in range(rows_per_cell):
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
                                "record_id": f"{partition}:{domain}:{target}:{index:04d}",
                                "source": source,
                                "target": target,
                                "text": f"{style} example number {index}",
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )


def expected_role(rows_per_cell: int, distinct_sources: int = 2) -> dict[str, object]:
    return {
        "rows": rows_per_cell * 2,
        "target_counts": {"0": rows_per_cell, "1": rows_per_cell},
        "distinct_sources": distinct_sources,
    }


class DomainHoldoutTests(unittest.TestCase):
    def test_materializes_metadata_only_selector_and_reconciles_counts(self) -> None:
        with temporary_directory() as root:
            input_path = root / "validation.jsonl.gz"
            output_path = root / "selected.jsonl.gz"
            write_partition(input_path, "validation", 5)
            identity = materialize_domain_role(
                input_path=input_path,
                output_path=output_path,
                partition="validation",
                role="validation",
                held_out_domain="alpha",
                expected=expected_role(5),
            )
            self.assertEqual(identity["rows"], 10)
            self.assertEqual(identity["target_counts"], {"0": 5, "1": 5})

            with self.assertRaisesRegex(DomainHoldoutError, "expected 11"):
                materialize_domain_role(
                    input_path=input_path,
                    output_path=output_path,
                    partition="validation",
                    role="validation",
                    held_out_domain="alpha",
                    expected={**expected_role(5), "rows": 11},
                )

    def test_runs_and_verifies_complete_fold_without_retaining_selected_text(self) -> None:
        with temporary_directory() as root:
            input_root = root / "input"
            input_root.mkdir()
            write_partition(input_root / "train.jsonl.gz", "train", 30)
            write_partition(input_root / "validation.jsonl.gz", "validation", 180)
            write_partition(input_root / "test.jsonl.gz", "test", 20)
            regime_report = {
                "dataset_id": "example/test",
                "revision": "abc123",
                "domain_holdouts": [
                    {
                        "regime_id": "domain:alpha",
                        "held_out_domain": "alpha",
                        "roles": {
                            "train": expected_role(30),
                            "validation": expected_role(180),
                            "test": expected_role(20),
                        },
                    }
                ],
            }
            artifact_root = root / "artifacts"
            work_root = root / "work"
            report = run_domain_holdouts(
                regime_report=regime_report,
                input_root=input_root,
                artifact_root=artifact_root,
                work_root=work_root,
                tfidf_config={"min_df": 1, "max_df": 1.0, "max_features": 100},
                logistic_config={"solver": "liblinear", "max_iter": 100},
            )
            verify_domain_holdouts(report, artifact_root)

            self.assertEqual(report["validation"]["folds_completed"], 1)
            self.assertFalse(
                report["configuration"]["test_data_used_for_training_calibration_or_thresholds"]
            )
            self.assertFalse((work_root / "alpha").exists())
            prediction_path = artifact_root / "alpha" / "test_predictions.jsonl.gz"
            self.assertTrue(prediction_path.is_file())
            self.assertNotIn("personal memory", prediction_path.read_bytes().decode("latin-1"))


if __name__ == "__main__":
    unittest.main()
