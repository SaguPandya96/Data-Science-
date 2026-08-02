"""Download the Amazon Reviews 2023 (McAuley Lab) All_Beauty category.

Source: https://amazon-reviews-2023.github.io/
Files are gzip-compressed JSONL, no auth required.
"""

from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
CATEGORY = "All_Beauty"

REVIEWS_URL = f"{BASE_URL}/review_categories/{CATEGORY}.jsonl.gz"
META_URL = f"{BASE_URL}/meta_categories/meta_{CATEGORY}.jsonl.gz"

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"Already downloaded: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        tmp_path = dest.with_suffix(dest.suffix + ".part")
        with open(tmp_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as progress:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                progress.update(len(chunk))
        tmp_path.rename(dest)
    print(f"Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    download_file(REVIEWS_URL, RAW_DIR / f"{CATEGORY}.jsonl.gz")
    download_file(META_URL, RAW_DIR / f"meta_{CATEGORY}.jsonl.gz")


if __name__ == "__main__":
    main()
