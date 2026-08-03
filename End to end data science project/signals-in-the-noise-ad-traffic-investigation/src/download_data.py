from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DATA_URL = (
    "https://huggingface.co/datasets/criteo/criteo-attribution-dataset/"
    "resolve/main/criteo_attribution_dataset.tsv.gz"
)
SOURCE_PAGE = "https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/"
LICENSE = "CC BY-NC-SA 4.0"

EXPECTED_COLUMNS = [
    "timestamp",
    "uid",
    "campaign",
    "conversion",
    "conversion_timestamp",
    "conversion_id",
    "attribution",
    "click",
    "click_pos",
    "click_nb",
    "cost",
    "cpo",
    "time_since_last_click",
    "cat1",
    "cat2",
    "cat3",
    "cat4",
    "cat5",
    "cat6",
    "cat7",
    "cat8",
    "cat9",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _existing_sample_is_usable(target: Path, requested_rows: int) -> bool:
    metadata_path = target.with_suffix(".metadata.json")
    if not target.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("requested_rows") == requested_rows and metadata.get("sha256") == _sha256(target)


def download_sample(target: Path, requested_rows: int = 500_000, force: bool = False) -> dict:
    """Stream a chronological sample without saving the 653 MB archive.

    A value of 0 for requested_rows reads the complete source file. The source is
    timestamp-sorted, so a finite sample is deliberately an early time slice, not
    a random sample. That limitation is recorded in the metadata and report.
    """

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not force and _existing_sample_is_usable(target, requested_rows):
        return json.loads(target.with_suffix(".metadata.json").read_text(encoding="utf-8"))

    temp_target = target.with_suffix(target.suffix + ".partial")
    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "signals-in-the-noise/1.0"})
    row_count = 0
    last_timestamp = -1
    click_count = 0
    conversion_count = 0

    print(f"Streaming Criteo data to {target} ...")
    with urllib.request.urlopen(request, timeout=60) as response:
        with gzip.GzipFile(fileobj=response) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_stream:
                reader = csv.DictReader(text_stream, delimiter="\t")
                if reader.fieldnames != EXPECTED_COLUMNS:
                    raise ValueError(f"Unexpected source schema: {reader.fieldnames}")
                with temp_target.open("w", encoding="utf-8", newline="") as output:
                    writer = csv.DictWriter(output, fieldnames=EXPECTED_COLUMNS, delimiter="\t")
                    writer.writeheader()
                    for row in reader:
                        timestamp = int(row["timestamp"])
                        if timestamp < last_timestamp:
                            raise ValueError("Source timestamps are not sorted as documented.")
                        if row["click"] not in {"0", "1"} or row["conversion"] not in {"0", "1"}:
                            raise ValueError("Click and conversion fields must be binary.")
                        writer.writerow(row)
                        row_count += 1
                        last_timestamp = timestamp
                        click_count += int(row["click"])
                        conversion_count += int(row["conversion"])
                        if row_count % 100_000 == 0:
                            print(f"  {row_count:,} rows")
                        if requested_rows and row_count >= requested_rows:
                            break

    if row_count == 0:
        raise ValueError("The source returned no rows.")
    temp_target.replace(target)

    metadata = {
        "source": SOURCE_PAGE,
        "download_url": DATA_URL,
        "license": LICENSE,
        "requested_rows": requested_rows,
        "sample_rows": row_count,
        "sampling_method": "first n rows from the timestamp-sorted source",
        "max_timestamp_seconds": last_timestamp,
        "clicks": click_count,
        "conversions": conversion_count,
        "sha256": _sha256(target),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    target.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata
