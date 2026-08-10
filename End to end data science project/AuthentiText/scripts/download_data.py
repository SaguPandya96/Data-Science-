"""Download pinned dataset files and verify their recorded identities."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.request import Request, urlopen

DEFAULT_CHUNK_SIZE = 1024 * 1024
USER_AGENT = "AuthentiText/0.1 dataset-acquisition"


class ManifestError(ValueError):
    """Raised when a dataset manifest is incomplete or unsafe."""


class IntegrityError(RuntimeError):
    """Raised when a local or downloaded file does not match its manifest."""


def _safe_relative_path(value: str) -> Path:
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise ManifestError(f"Unsafe relative path: {value!r}")
    return Path(*candidate.parts)


def validate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate the fields needed for deterministic, safe acquisition."""
    for field in ("schema_version", "dataset_id", "revision", "files"):
        if field not in manifest:
            raise ManifestError(f"Manifest is missing {field!r}")

    if manifest["schema_version"] != 1:
        raise ManifestError("Only manifest schema version 1 is supported")
    if not isinstance(manifest["dataset_id"], str) or not manifest["dataset_id"]:
        raise ManifestError("Manifest dataset_id must be a non-empty string")
    if not isinstance(manifest["revision"], str) or not manifest["revision"]:
        raise ManifestError("Manifest revision must be a non-empty string")
    if not isinstance(manifest["files"], list) or not manifest["files"]:
        raise ManifestError("Manifest files must be a non-empty list")

    seen_names: set[str] = set()
    seen_paths: set[Path] = set()
    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            raise ManifestError("Each file entry must be a JSON object")
        for field in (
            "name",
            "relative_path",
            "url",
            "size_bytes",
            "sha256",
            "expected_columns",
        ):
            if field not in entry:
                raise ManifestError(f"File entry is missing {field!r}")

        name = entry["name"]
        if not isinstance(name, str) or not name:
            raise ManifestError("File names must be non-empty strings")
        if name in seen_names:
            raise ManifestError(f"Duplicate file name: {name}")
        seen_names.add(name)

        if not isinstance(entry["relative_path"], str):
            raise ManifestError(f"File {name} has an invalid relative path")
        relative_path = _safe_relative_path(entry["relative_path"])
        if relative_path in seen_paths:
            raise ManifestError(f"Duplicate file path: {relative_path}")
        seen_paths.add(relative_path)

        if not isinstance(entry["url"], str) or not entry["url"].startswith("https://"):
            raise ManifestError(f"File {name} must use an HTTPS URL")
        if not isinstance(entry["size_bytes"], int) or entry["size_bytes"] <= 0:
            raise ManifestError(f"File {name} has an invalid size")
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ManifestError(f"File {name} has an invalid SHA-256 digest")
        columns = entry["expected_columns"]
        if (
            not isinstance(columns, list)
            or not columns
            or not all(isinstance(column, str) and column for column in columns)
        ):
            raise ManifestError(f"File {name} has invalid expected columns")

    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and validate a JSON acquisition manifest."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"Could not read manifest {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ManifestError("Manifest root must be a JSON object")
    return validate_manifest(manifest)


def sha256_file(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Return the lowercase SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, entry: dict[str, Any]) -> None:
    """Verify file size, checksum, and CSV header."""
    if not path.is_file():
        raise IntegrityError(f"Missing file: {path}")

    actual_size = path.stat().st_size
    if actual_size != entry["size_bytes"]:
        raise IntegrityError(f"{path} has {actual_size} bytes; expected {entry['size_bytes']}")

    actual_sha256 = sha256_file(path)
    if actual_sha256 != entry["sha256"]:
        raise IntegrityError(f"{path} has SHA-256 {actual_sha256}; expected {entry['sha256']}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        columns = next(csv.reader(handle), None)
    if columns != entry["expected_columns"]:
        raise IntegrityError(
            f"{path} has columns {columns!r}; expected {entry['expected_columns']!r}"
        )


def _download_to_partial(
    entry: dict[str, Any],
    partial_path: Path,
    *,
    timeout: float,
    chunk_size: int,
) -> None:
    offset = partial_path.stat().st_size if partial_path.exists() else 0
    if offset > entry["size_bytes"]:
        raise IntegrityError(f"Partial file {partial_path} is larger than the expected download")

    if offset == entry["size_bytes"]:
        return

    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(entry["url"], headers=headers)

    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - pinned HTTPS
        status = getattr(response, "status", response.getcode())
        if offset and status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                raise IntegrityError(f"Server resumed from an unexpected range: {content_range!r}")
            mode = "ab"
        elif status == 200:
            mode = "wb"
        else:
            raise IntegrityError(f"Unexpected HTTP status {status} for {entry['name']}")

        with partial_path.open(mode) as output:
            while chunk := response.read(chunk_size):
                output.write(chunk)


def acquire_file(
    entry: dict[str, Any],
    output_root: Path,
    *,
    verify_only: bool = False,
    timeout: float = 60,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """Acquire one manifest entry and return ``downloaded`` or ``verified``."""
    target = output_root / _safe_relative_path(entry["relative_path"])
    if target.exists():
        verify_file(target, entry)
        return "verified"
    if verify_only:
        raise IntegrityError(f"Missing file in verify-only mode: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    partial_path = target.with_name(f"{target.name}.part")
    try:
        _download_to_partial(
            entry,
            partial_path,
            timeout=timeout,
            chunk_size=chunk_size,
        )
        verify_file(partial_path, entry)
    except IntegrityError:
        if partial_path.is_file() and partial_path.stat().st_size >= entry["size_bytes"]:
            partial_path.unlink()
        raise
    partial_path.replace(target)
    return "downloaded"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_ood_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "raw",
    )
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        help="Manifest file name to acquire; repeat to select multiple (default: all)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check local files without making network requests",
    )
    parser.add_argument("--timeout", type=float, default=60)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        entries = {entry["name"]: entry for entry in manifest["files"]}
        selected = args.files or list(entries)
        unknown = sorted(set(selected) - set(entries))
        if unknown:
            raise ManifestError(f"Unknown file selection: {', '.join(unknown)}")

        for name in selected:
            result = acquire_file(
                entries[name],
                args.output_dir,
                verify_only=args.verify_only,
                timeout=args.timeout,
            )
            print(f"{name}: {result}")
    except (ManifestError, IntegrityError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
