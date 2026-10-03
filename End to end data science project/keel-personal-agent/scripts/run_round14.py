"""Round 14 of the memory benchmark: encoders of about 2.2 GB.

Downloads three encoders of about 2.2 GB each (about 7 GB in all, pinned by SHA-256) on
first use, chooses each one's weight on question sets already seen, then tests the best
once on the tenth held-out set if it beat e5-large-v2 there. Writes
reports/metrics/round14.json and round14.md, and machine-dependent timings to
reports/metrics/round14_timing.json.

    python scripts/run_round14.py
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

    results, timings = run_round11(args.config, section="round14")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "round14.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "round14.md").write_text(to_markdown(results, section="round14"))
    (out / "round14_timing.json").write_text(json.dumps(timings, indent=2) + "\n")
    print(to_markdown(results, timings, section="round14"))


if __name__ == "__main__":
    main()
