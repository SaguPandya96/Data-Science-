"""Round 7 of the memory benchmark: cross-encoder reranking.

Downloads two ms-marco MiniLM cross-encoders (about 220 MB, pinned by SHA-256) on first
use, chooses the reranker, short-list size and weight on question sets already seen, then
scores the choice once on the sixth held-out set. Writes reports/metrics/round7.json and
round7.md, and machine-dependent timings to reports/metrics/round7_timing.json.

    python scripts/run_round7.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from keel.evaluation.round7 import run_round7, to_markdown

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval.toml"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "metrics"))
    args = parser.parse_args()

    results, timings = run_round7(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "round7.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "round7.md").write_text(to_markdown(results))
    (out / "round7_timing.json").write_text(json.dumps(timings, indent=2) + "\n")
    print(to_markdown(results, timings))


if __name__ == "__main__":
    main()
