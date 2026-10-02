"""Round 12 of the memory benchmark: larger encoders.

Downloads three encoders of about three times e5-base's size (about 4 GB in all, pinned by
SHA-256) on first use, chooses each one's weight on question sets already seen, then tests
the best once on the ninth held-out set if it beat e5-base there. Writes
reports/metrics/round12.json and round12.md, and machine-dependent timings to
reports/metrics/round12_timing.json.

    python scripts/run_round12.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from keel.evaluation.round11 import run_round11, to_markdown

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval.toml"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "metrics"))
    args = parser.parse_args()

    results, timings = run_round11(args.config, section="round12")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "round12.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "round12.md").write_text(to_markdown(results, section="round12"))
    (out / "round12_timing.json").write_text(json.dumps(timings, indent=2) + "\n")
    print(to_markdown(results, timings, section="round12"))


if __name__ == "__main__":
    main()
