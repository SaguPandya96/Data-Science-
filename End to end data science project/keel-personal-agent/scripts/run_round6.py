"""Round 6 of the memory benchmark: larger transformer encoders.

Downloads bge-base-en-v1.5 and all-mpnet-base-v2 (about 600 MB, pinned by SHA-256) on first
use, chooses the encoder and weight on question sets already seen, then scores the choice
once on the fifth held-out set. Writes reports/metrics/round6.json and round6.md, and
machine-dependent timings to reports/metrics/round6_timing.json.

    python scripts/run_round6.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from keel.evaluation.round6 import run_round6, to_markdown

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval.toml"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "metrics"))
    args = parser.parse_args()

    results, timings = run_round6(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "round6.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "round6.md").write_text(to_markdown(results))
    (out / "round6_timing.json").write_text(json.dumps(timings, indent=2) + "\n")
    print(to_markdown(results, timings))


if __name__ == "__main__":
    main()
