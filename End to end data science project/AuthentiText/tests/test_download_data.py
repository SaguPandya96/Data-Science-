from __future__ import annotations

import hashlib
import io
import json
import shutil
import unittest
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from scripts import download_data

TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / "data" / "interim" / "test-downloads"
TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def temporary_directory() -> Iterator[str]:
    path = TEST_TEMP_ROOT / f"case-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield str(path)
    finally:
        shutil.rmtree(path)


class FakeResponse(io.BytesIO):
    def __init__(
        self,
        payload: bytes,
        *,
        status: int = 200,
        content_range: str | None = None,
    ) -> None:
        super().__init__(payload)
        self.status = status
        self.headers = {}
        if content_range is not None:
            self.headers["Content-Range"] = content_range

    def getcode(self) -> int:
        return self.status


def make_entry(payload: bytes) -> dict[str, object]:
    return {
        "name": "sample",
        "relative_path": "example/sample.csv",
        "url": "https://example.test/sample.csv",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "expected_columns": ["text", "label"],
    }


class DownloadDataTests(unittest.TestCase):
    payload = b"text,label\nhello,1\n"

    def test_downloads_and_verifies_new_file(self) -> None:
        entry = make_entry(self.payload)
        with temporary_directory() as directory:
            output_root = Path(directory)
            with patch.object(
                download_data,
                "urlopen",
                return_value=FakeResponse(self.payload),
            ) as opener:
                result = download_data.acquire_file(entry, output_root)

            self.assertEqual(result, "downloaded")
            self.assertEqual(
                (output_root / "example" / "sample.csv").read_bytes(),
                self.payload,
            )
            self.assertEqual(opener.call_count, 1)

    def test_valid_existing_file_uses_no_network(self) -> None:
        entry = make_entry(self.payload)
        with temporary_directory() as directory:
            output_root = Path(directory)
            target = output_root / "example" / "sample.csv"
            target.parent.mkdir()
            target.write_bytes(self.payload)

            with patch.object(
                download_data,
                "urlopen",
                side_effect=AssertionError("network should not be used"),
            ):
                result = download_data.acquire_file(entry, output_root)

            self.assertEqual(result, "verified")

    def test_resumes_partial_download_with_range_request(self) -> None:
        entry = make_entry(self.payload)
        split_at = 8
        with temporary_directory() as directory:
            output_root = Path(directory)
            partial = output_root / "example" / "sample.csv.part"
            partial.parent.mkdir()
            partial.write_bytes(self.payload[:split_at])

            response = FakeResponse(
                self.payload[split_at:],
                status=206,
                content_range=f"bytes {split_at}-{len(self.payload) - 1}/{len(self.payload)}",
            )
            with patch.object(
                download_data,
                "urlopen",
                return_value=response,
            ) as opener:
                result = download_data.acquire_file(entry, output_root)

            request = opener.call_args.args[0]
            self.assertEqual(request.get_header("Range"), f"bytes={split_at}-")
            self.assertEqual(result, "downloaded")
            self.assertEqual(
                (output_root / "example" / "sample.csv").read_bytes(),
                self.payload,
            )

    def test_restarts_when_server_ignores_range(self) -> None:
        entry = make_entry(self.payload)
        with temporary_directory() as directory:
            output_root = Path(directory)
            partial = output_root / "example" / "sample.csv.part"
            partial.parent.mkdir()
            partial.write_bytes(b"stale")

            with patch.object(
                download_data,
                "urlopen",
                return_value=FakeResponse(self.payload, status=200),
            ):
                result = download_data.acquire_file(entry, output_root)

            self.assertEqual(result, "downloaded")
            self.assertEqual(
                (output_root / "example" / "sample.csv").read_bytes(),
                self.payload,
            )

    def test_corrupt_existing_file_is_not_overwritten(self) -> None:
        entry = make_entry(self.payload)
        with temporary_directory() as directory:
            output_root = Path(directory)
            target = output_root / "example" / "sample.csv"
            target.parent.mkdir()
            target.write_bytes(b"corrupt")

            with patch.object(
                download_data,
                "urlopen",
                side_effect=AssertionError("network should not be used"),
            ):
                with self.assertRaises(download_data.IntegrityError):
                    download_data.acquire_file(entry, output_root)

            self.assertEqual(target.read_bytes(), b"corrupt")

    def test_checksum_failure_removes_completed_partial(self) -> None:
        entry = make_entry(self.payload)
        bad_payload = b"text,label\nwrong,0\n"
        self.assertEqual(len(bad_payload), len(self.payload))
        with temporary_directory() as directory:
            output_root = Path(directory)
            with patch.object(
                download_data,
                "urlopen",
                return_value=FakeResponse(bad_payload),
            ):
                with self.assertRaises(download_data.IntegrityError):
                    download_data.acquire_file(entry, output_root)

            self.assertFalse((output_root / "example" / "sample.csv.part").exists())
            self.assertFalse((output_root / "example" / "sample.csv").exists())

    def test_corrupt_completed_partial_is_removed_without_network(self) -> None:
        entry = make_entry(self.payload)
        bad_payload = b"text,label\nwrong,0\n"
        with temporary_directory() as directory:
            output_root = Path(directory)
            partial = output_root / "example" / "sample.csv.part"
            partial.parent.mkdir()
            partial.write_bytes(bad_payload)

            with patch.object(
                download_data,
                "urlopen",
                side_effect=AssertionError("network should not be used"),
            ):
                with self.assertRaises(download_data.IntegrityError):
                    download_data.acquire_file(entry, output_root)

            self.assertFalse(partial.exists())

    def test_manifest_rejects_path_traversal(self) -> None:
        entry = make_entry(self.payload)
        entry["relative_path"] = "../outside.csv"
        manifest = {
            "schema_version": 1,
            "dataset_id": "example/test",
            "revision": "abc123",
            "files": [entry],
        }
        with temporary_directory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(download_data.ManifestError):
                download_data.load_manifest(path)


if __name__ == "__main__":
    unittest.main()
