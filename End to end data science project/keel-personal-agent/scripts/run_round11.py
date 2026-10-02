"""Round 11 of the memory benchmark: other encoders of mpnet's size.

Downloads six encoders of about mpnet's size (about 2.7 GB in all, pinned by SHA-256) on
first use, chooses each one's weight on question sets already seen, then tests the best
once on the eighth held-out set if it beat mpnet there. Writes reports/metrics/round11.json
and round11.md, and machine-dependent timings to reports/metrics/round11_timing.json.

    python scripts/run_round11.py
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

    results, timings = run_round11(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "round11.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "round11.md").write_text(to_markdown(results))
    (out / "round11_timing.json").write_text(json.dumps(timings, indent=2) + "\n")
    print(to_markdown(results, timings))


if __name__ == "__main__":
    main()
