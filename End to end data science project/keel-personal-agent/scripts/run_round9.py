"""Round 9 of the memory benchmark: smaller encoders.

Downloads four encoders a third of mpnet's size or less (about 490 MB in all, pinned by
SHA-256) on first use, chooses each one's weight on question sets already seen, then
tests the smallest qualifying one once on the eighth held-out set. Writes
reports/metrics/round9.json and round9.md, and machine-dependent timings to
reports/metrics/round9_timing.json.

    python scripts/run_round9.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from keel.evaluation.round9 import run_round9, to_markdown

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval.toml"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "metrics"))
    args = parser.parse_args()

    results, timings = run_round9(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "round9.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "round9.md").write_text(to_markdown(results))
    (out / "round9_timing.json").write_text(json.dumps(timings, indent=2) + "\n")
    print(to_markdown(results, timings))


if __name__ == "__main__":
    main()
