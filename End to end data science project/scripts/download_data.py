"""Download the immutable SCMS shipment file from a pinned public mirror."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.request import Request, urlopen

MIRROR_URL = (
    "https://raw.githubusercontent.com/SanjogRam619/FedEx-Logistics-EDA/"
    "27dc0c7d20267ec2627b39f2290994bcf7186f30/SCMS_Delivery_History_Dataset.csv"
)
EXPECTED_SHA256 = "918b992dd3e8d4b64d2a727b2c4ea607603d0c58f19484e73f7b78528c6a8673"
DEFAULT_OUTPUT = Path("data/raw/SCMS_Delivery_History_Dataset.csv")


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
    request = Request(MIRROR_URL, headers={"User-Agent": "SupplyLens/0.1"})
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
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        download(args.output, args.force)
    except Exception as exc:  # pragma: no cover - CLI boundary
        print(f"Data download failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

