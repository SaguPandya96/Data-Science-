"""Round 18 of the memory benchmark: other cross-encoder rerankers.

Downloads three int8 cross-encoders (about 95 MB in all, pinned by SHA-256) on first use,
chooses each one's blend weight on question sets already seen, then tests the best once on
the twelfth held-out set if it differs from the shipped reranker. Writes
reports/metrics/round18.json and round18.md, and machine-dependent timings to
reports/metrics/round18_timing.json.

    python scripts/run_round18.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from keel.evaluation.round18 import run_round18, to_markdown

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval.toml"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "metrics"))
    args = parser.parse_args()

    results, timings = run_round18(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "round18.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "round18.md").write_text(to_markdown(results))
    (out / "round18_timing.json").write_text(json.dumps(timings, indent=2) + "\n")
    print(to_markdown(results, timings))


if __name__ == "__main__":
    main()
