from __future__ import annotations

import shutil
import unittest
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from authentitext.data.ghostbuster import (
    CONDITIONS,
    DOMAINS,
    GhostbusterError,
    prepare_ghostbuster_main,
    render_report,
    verify_ghostbuster_main,
)

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "data" / "interim" / "test-ghostbuster"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def temporary_directory() -> Iterator[Path]:
    path = TEST_TEMP_ROOT / f"case-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def write_fixture(repository_root: Path, documents_per_cell: int = 3) -> None:
    (repository_root / "LICENSE").write_text("fixture license", encoding="utf-8")
    for repository_domain in DOMAINS:
        for condition, metadata in CONDITIONS.items():
            condition_root = repository_root / repository_domain / condition
            if repository_domain == "reuter":
                condition_root /= "fixture_author"
            condition_root.mkdir(parents=True)
            for index in range(1, documents_per_cell + 1):
                style = "human personal draft" if metadata["target"] == 0 else "machine formal text"
                (condition_root / f"{index}.txt").write_text(
                    f"{style} for {repository_domain} {condition} {index}",
                    encoding="utf-8",
                )
            logprobs = condition_root / "logprobs"
            logprobs.mkdir()
            (logprobs / "1-ada.txt").write_text("ignored", encoding="utf-8")
    auxiliary = repository_root / "other" / "fixture"
    auxiliary.mkdir(parents=True)
    (auxiliary / "1.txt").write_text("ignored auxiliary", encoding="utf-8")


class GhostbusterPreparationTests(unittest.TestCase):
    def test_prepares_and_verifies_only_declared_main_corpus(self) -> None:
        with temporary_directory() as root:
            repository_root = root / "repository"
            repository_root.mkdir()
            write_fixture(repository_root)
            output_path = root / "main.jsonl.gz"
            report = prepare_ghostbuster_main(
                repository_root=repository_root,
                output_path=output_path,
                revision="abc123",
                documents_per_cell=3,
            )
            verify_ghostbuster_main(report, repository_root, output_path)

            self.assertEqual(report["output"]["rows"], 63)
            self.assertEqual(report["profile"]["target_counts"], {"0": 9, "1": 54})
            self.assertEqual(report["validation"]["test_scoring_performed"], False)
            self.assertNotIn("human personal draft", render_report(report))

    def test_rejects_incomplete_declared_cell(self) -> None:
        with temporary_directory() as root:
            repository_root = root / "repository"
            repository_root.mkdir()
            write_fixture(repository_root)
            (repository_root / "essay" / "human" / "3.txt").unlink()
            with self.assertRaisesRegex(GhostbusterError, "expected 3"):
                prepare_ghostbuster_main(
                    repository_root=repository_root,
                    output_path=root / "main.jsonl.gz",
                    revision="abc123",
                    documents_per_cell=3,
                )

    def test_excludes_and_records_blank_source_document(self) -> None:
        with temporary_directory() as root:
            repository_root = root / "repository"
            repository_root.mkdir()
            write_fixture(repository_root)
            blank_path = repository_root / "essay" / "human" / "2.txt"
            blank_path.write_text("\n\t", encoding="utf-8")
            output_path = root / "main.jsonl.gz"

            report = prepare_ghostbuster_main(
                repository_root=repository_root,
                output_path=output_path,
                revision="abc123",
                documents_per_cell=3,
            )
            verify_ghostbuster_main(report, repository_root, output_path)

            self.assertEqual(report["selection"]["paper_main_corpus_rows"], 63)
            self.assertEqual(report["output"]["rows"], 62)
            self.assertEqual(report["profile"]["target_counts"], {"0": 8, "1": 54})
            self.assertEqual(report["validation"]["blank_documents_excluded"], 1)
            self.assertEqual(
                report["validation"]["blank_relative_paths"],
                ["essay/human/2.txt"],
            )


if __name__ == "__main__":
    unittest.main()
