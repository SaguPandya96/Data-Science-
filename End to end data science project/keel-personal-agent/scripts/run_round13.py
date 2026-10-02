"""Round 13 of the memory benchmark: other encoders of e5-large's size.

Downloads three encoders the size of e5-large-v2 (about 4 GB in all, pinned by SHA-256) on
first use, chooses each one's weight on question sets already seen, then tests the best
once on the tenth held-out set if it beat e5-large-v2 there. Writes
reports/metrics/round13.json and round13.md, and machine-dependent timings to
reports/metrics/round13_timing.json.

    python scripts/run_round13.py
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

    results, timings = run_round11(args.config, section="round13")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "round13.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "round13.md").write_text(to_markdown(results, section="round13"))
    (out / "round13_timing.json").write_text(json.dumps(timings, indent=2) + "\n")
    print(to_markdown(results, timings, section="round13"))


if __name__ == "__main__":
    main()
