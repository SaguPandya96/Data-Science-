from __future__ import annotations

import hashlib
import shutil
import unittest
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from scripts import profile_data

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "data" / "interim" / "test-profiles"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def temporary_directory() -> Iterator[Path]:
    path = TEST_TEMP_ROOT / f"case-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def make_entry(name: str, relative_path: str, payload: bytes) -> dict[str, object]:
    return {
        "name": name,
        "relative_path": relative_path,
        "url": f"https://example.test/{relative_path}",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "expected_columns": ["text", "label", "src"],
    }


def make_manifest(entries: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset": "Example",
        "dataset_id": "example/test",
        "revision": "abc123",
        "label_semantics": {
            "raw_to_semantic": {
                "0": "machine_generated",
                "1": "human_written",
            },
            "canonical_machine_positive_mapping": {"0": 1, "1": 0},
            "source_revision": "def456",
            "source_url": "https://example.test/labels",
        },
        "files": entries,
    }


class ProfileDataTests(unittest.TestCase):
    def test_profiles_counts_lengths_and_cross_file_duplicates(self) -> None:
        first = b"text,label,src\nsame,1,a_human\nmachine text,0,a_model\n"
        second = b"text,label,src\nsame,1,a_human\n"
        entries = [
            make_entry("first", "example/first.csv", first),
            make_entry("second", "example/second.csv", second),
        ]

        with temporary_directory() as root:
            for entry, payload in zip(entries, (first, second), strict=True):
                path = root / str(entry["relative_path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            profile = profile_data.build_profile(make_manifest(entries), root)

        self.assertEqual(profile["validation"]["status"], "pass")
        self.assertEqual(profile["aggregate"]["rows"], 3)
        self.assertIsNone(profile["files"][0]["record_identifier_column"])
        self.assertIn(
            "No stable record identifier",
            profile["validation"]["warnings"][0],
        )
        self.assertEqual(
            profile["aggregate"]["semantic_label_counts"],
            {"human_written": 2, "machine_generated": 1},
        )
        self.assertEqual(profile["aggregate"]["canonical_target_counts"], {"0": 2, "1": 1})
        duplication = profile["aggregate"]["exact_text_duplication"]
        self.assertEqual(duplication["unique_texts"], 2)
        self.assertEqual(duplication["duplicate_groups"], 1)
        self.assertEqual(duplication["cross_file_groups"], 1)
        self.assertEqual(duplication["conflicting_label_groups"], 0)
        self.assertEqual(duplication["group_raw_label_counts"], {"1": 1})
        self.assertEqual(duplication["cross_file_set_counts"], {"first|second": 1})

    def test_blank_required_value_fails_validation(self) -> None:
        payload = b"text,label,src\n   ,1,a_human\n"
        entry = make_entry("blank", "example/blank.csv", payload)
        with temporary_directory() as root:
            path = root / str(entry["relative_path"])
            path.parent.mkdir(parents=True)
            path.write_bytes(payload)
            profile = profile_data.build_profile(make_manifest([entry]), root)

        self.assertEqual(profile["validation"]["status"], "fail")
        self.assertIn("blank text values", profile["validation"]["violations"][0])

    def test_unknown_label_fails_validation(self) -> None:
        payload = b"text,label,src\nexample,2,a_unknown\n"
        entry = make_entry("labels", "example/labels.csv", payload)
        with temporary_directory() as root:
            path = root / str(entry["relative_path"])
            path.parent.mkdir(parents=True)
            path.write_bytes(payload)
            profile = profile_data.build_profile(make_manifest([entry]), root)

        self.assertEqual(profile["validation"]["status"], "fail")
        self.assertEqual(profile["files"][0]["invalid_label_counts"], {"2": 1})

    def test_rendering_is_deterministic_and_text_free(self) -> None:
        payload = b"text,label,src\nprivate example,0,a_model\n"
        entry = make_entry("sample", "example/sample.csv", payload)
        with temporary_directory() as root:
            path = root / str(entry["relative_path"])
            path.parent.mkdir(parents=True)
            path.write_bytes(payload)
            profile = profile_data.build_profile(make_manifest([entry]), root)

        first = profile_data.render_profile(profile)
        second = profile_data.render_profile(profile)
        self.assertEqual(first, second)
        self.assertNotIn("private example", first)


if __name__ == "__main__":
    unittest.main()
