from __future__ import annotations

import json
import shutil
import unittest
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from authentitext.data.cleaning import open_deterministic_gzip, sha256_file
from authentitext.data.regimes import (
    RegimeError,
    assign_domain_role,
    assign_generator_role,
    define_source_regimes,
    parse_source_id,
)

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "data" / "interim" / "test-regimes"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def temporary_directory() -> Iterator[Path]:
    path = TEST_TEMP_ROOT / f"case-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def write_fixture(root: Path) -> dict[str, object]:
    source_targets = {
        "alpha_domain_human": 0,
        "alpha_domain_machine_continuation_model_a": 1,
        "alpha_domain_machine_topical_model_b": 1,
        "beta_human": 0,
        "beta_machine_continuation_model_a": 1,
        "beta_machine_specified_model_b": 1,
    }
    outputs = []
    total = 0
    for partition in ("train", "validation", "test"):
        path = root / f"{partition}.jsonl.gz"
        with open_deterministic_gzip(path) as handle:
            for source, target in source_targets.items():
                record = {
                    "partition": partition,
                    "record_id": f"{partition}:{source}",
                    "source": source,
                    "target": target,
                    "text": f"text for {partition} {source}",
                }
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                total += 1
        outputs.append(
            {
                "partition": partition,
                "relative_path": path.name,
                "rows_written": len(source_targets),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "dataset_id": "example/test",
        "revision": "abc123",
        "split_name": "id",
        "outputs": outputs,
        "aggregate": {"output_rows": total},
    }


class RegimeTests(unittest.TestCase):
    def test_source_parser_preserves_domain_and_exact_generator(self) -> None:
        human = parse_source_id("sci_gen_human")
        machine = parse_source_id("sci_gen_machine_continuation_gpt-3.5-trubo")
        self.assertEqual((human.domain, human.kind), ("sci_gen", "human"))
        self.assertEqual(machine.domain, "sci_gen")
        self.assertEqual(machine.strategy, "continuation")
        self.assertEqual(machine.generator, "gpt-3.5-trubo")
        self.assertEqual(assign_domain_role("test", machine, "sci_gen"), "test")
        self.assertEqual(assign_generator_role("test", machine, "gpt-3.5-trubo"), "test")
        with self.assertRaises(RegimeError):
            parse_source_id("unsupported")

    def test_defines_reconciled_domain_and_generator_holdouts(self) -> None:
        with temporary_directory() as root:
            split_report = write_fixture(root)
            report = define_source_regimes(split_report, root)

        self.assertEqual(report["population"]["rows"], 18)
        self.assertEqual(report["population"]["domains"], ["alpha_domain", "beta"])
        self.assertEqual(report["population"]["generators"], ["model_a", "model_b"])
        self.assertEqual(len(report["domain_holdouts"]), 2)
        alpha = report["domain_holdouts"][0]
        self.assertEqual(alpha["roles"]["train"]["rows"], 3)
        self.assertEqual(alpha["roles"]["validation"]["rows"], 3)
        self.assertEqual(alpha["roles"]["test"]["rows"], 3)
        self.assertEqual(alpha["roles"]["excluded"]["rows"], 9)
        model_a = report["generator_holdouts"][0]
        self.assertEqual(model_a["roles"]["train"]["rows"], 4)
        self.assertEqual(model_a["roles"]["validation"]["rows"], 4)
        self.assertEqual(model_a["roles"]["test"]["rows"], 4)
        self.assertEqual(model_a["roles"]["test"]["target_counts"], {"0": 2, "1": 2})
        self.assertEqual(model_a["roles"]["excluded"]["rows"], 6)
        self.assertTrue(report["validation"]["every_regime_reconciles_population"])

    def test_rejects_source_target_conflicts(self) -> None:
        with temporary_directory() as root:
            split_report = write_fixture(root)
            path = root / "train.jsonl.gz"
            bad_record = {
                "partition": "train",
                "record_id": "bad",
                "source": "alpha_domain_human",
                "target": 1,
                "text": "bad",
            }
            with open_deterministic_gzip(path) as handle:
                handle.write(json.dumps(bad_record) + "\n")
            split_report["outputs"][0]["rows_written"] = 1
            split_report["aggregate"]["output_rows"] = 13
            with self.assertRaisesRegex(RegimeError, "conflicts with source kind"):
                define_source_regimes(split_report, root)


if __name__ == "__main__":
    unittest.main()
