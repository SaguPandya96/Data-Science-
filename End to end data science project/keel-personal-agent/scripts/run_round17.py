"""Round 17 of the memory benchmark: the reranker's setting for the current encoder.

Uses the shipped int8 e5-large-v2 and int8 reranker (downloaded once, pinned by SHA-256),
chooses the short-list size and blend weight on question sets already seen, then tests the
choice once on the eleventh held-out set if it differs from the current setting. Writes
reports/metrics/round17.json and round17.md, and machine-dependent timings to
reports/metrics/round17_timing.json.

    python scripts/run_round17.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from keel.evaluation.round17 import run_round17, to_markdown

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval.toml"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "metrics"))
    args = parser.parse_args()

    results, timings = run_round17(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "round17.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "round17.md").write_text(to_markdown(results))
    (out / "round17_timing.json").write_text(json.dumps(timings, indent=2) + "\n")
    print(to_markdown(results, timings))


if __name__ == "__main__":
    main()
