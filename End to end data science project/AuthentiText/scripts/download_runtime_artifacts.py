"""Download and verify the frozen runtime artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlparse

CHUNK_BYTES = 1024 * 1024
ALLOWED_HOST = "github.com"


class ArtifactDownloadError(RuntimeError):
    """Raised when a runtime artifact cannot be trusted."""


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ArtifactDownloadError("Unsupported runtime artifact manifest")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ArtifactDownloadError("Runtime artifact manifest is empty")
    return artifacts


def _validate_entry(entry: dict[str, Any]) -> tuple[str, str, int, str]:
    filename = entry.get("filename")
    url = entry.get("url")
    expected_bytes = entry.get("bytes")
    expected_sha256 = entry.get("sha256")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ArtifactDownloadError("Artifact filename is invalid")
    parsed = urlparse(url) if isinstance(url, str) else None
    if parsed is None or parsed.scheme != "https" or parsed.hostname != ALLOWED_HOST:
        raise ArtifactDownloadError(f"Artifact URL is not an approved GitHub URL: {filename}")
    if not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise ArtifactDownloadError(f"Artifact size is invalid: {filename}")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ArtifactDownloadError(f"Artifact SHA-256 is invalid: {filename}")
    return filename, url, expected_bytes, expected_sha256


def _copy_and_hash(source: BinaryIO, target: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    while chunk := source.read(CHUNK_BYTES):
        target.write(chunk)
        digest.update(chunk)
        total += len(chunk)
    return total, digest.hexdigest()


def download_runtime_artifacts(
    manifest_path: Path,
    output_dir: Path,
    *,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
    timeout: float = 120.0,
) -> list[Path]:
    """Download every manifest entry atomically after identity verification."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for entry in _load_manifest(manifest_path):
        filename, url, expected_bytes, expected_sha256 = _validate_entry(entry)
        destination = output_dir / filename
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{filename}.", dir=output_dir)
        try:
            with os.fdopen(descriptor, "wb") as target:
                request = urllib.request.Request(url, headers={"User-Agent": "AuthentiText/1"})
                with opener(request, timeout=timeout) as source:
                    actual_bytes, actual_sha256 = _copy_and_hash(source, target)
            if actual_bytes != expected_bytes:
                raise ArtifactDownloadError(f"Artifact size mismatch: {filename}")
            if actual_sha256 != expected_sha256:
                raise ArtifactDownloadError(f"Artifact SHA-256 mismatch: {filename}")
            Path(temporary_name).replace(destination)
            written.append(destination)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = download_runtime_artifacts(args.manifest, args.output_dir, timeout=args.timeout)
    for path in paths:
        print(f"verified runtime artifact: {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
