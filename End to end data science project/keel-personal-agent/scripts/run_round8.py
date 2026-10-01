"""Round 8 of the memory benchmark: quantized models.

Downloads int8 versions of mpnet and the reranker (about 135 MB, pinned by SHA-256) on first
use, scores them on question sets already seen, then tests the chosen variant once on the
seventh held-out set. Writes reports/metrics/round8.json and round8.md, and
machine-dependent timings to reports/metrics/round8_timing.json.

    python scripts/run_round8.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from keel.evaluation.round8 import run_round8, to_markdown

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval.toml"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "metrics"))
    args = parser.parse_args()

    results, timings = run_round8(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "round8.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "round8.md").write_text(to_markdown(results))
    (out / "round8_timing.json").write_text(json.dumps(timings, indent=2) + "\n")
    print(to_markdown(results, timings))


if __name__ == "__main__":
    main()
