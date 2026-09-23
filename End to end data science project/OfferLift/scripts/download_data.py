"""Download the Hillstrom MineThatData email experiment from a pinned public mirror."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.request import Request, urlopen

# The original minethatdata.com file, as mirrored by the scikit-uplift project.
MIRROR_URL = "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz"
EXPECTED_SHA256 = "bab6578f60db5d792f1c2372c502f029152a5249cf5ea84390f3b7f885d7234f"
DEFAULT_OUTPUT = Path("data/raw/hillstrom.csv.gz")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(output: Path, force: bool = False) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        observed = sha256(output)
        if observed == EXPECTED_SHA256:
            print(f"Data already present and checksum verified: {output}")
            return output
        raise RuntimeError(
            f"Refusing to overwrite {output}: checksum {observed} does not match expected "
            f"{EXPECTED_SHA256}. Re-run with --force only after investigating."
        )

    temporary = output.with_suffix(output.suffix + ".download")
    request = Request(MIRROR_URL, headers={"User-Agent": "OfferLift/0.1"})
    try:
        with urlopen(request, timeout=90) as response, temporary.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        observed = sha256(temporary)
        if observed != EXPECTED_SHA256:
            raise RuntimeError(
                f"Downloaded checksum {observed} does not match expected {EXPECTED_SHA256}"
            )
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"Downloaded {output} ({output.stat().st_size:,} bytes); SHA-256 {EXPECTED_SHA256}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Overwrite a mismatched file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        download(args.output, force=args.force)
    except Exception as error:  # noqa: BLE001 - surface any failure as a clean exit
        print(f"Download failed: {error}", file=sys.stderr)
        sys.exit(1)
