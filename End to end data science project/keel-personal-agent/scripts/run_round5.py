"""Round 5 of the memory benchmark: a wider weight grid for the MiniLM encoder.

Chooses the weight on question sets that have already been seen, then scores it once on
the fourth held-out set. Writes reports/metrics/round5.json and round5.md.

    python scripts/run_round5.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from keel.evaluation.round5 import run_round5, to_markdown

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval.toml"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "metrics"))
    args = parser.parse_args()

    results = run_round5(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "round5.json").write_text(json.dumps(results, indent=2) + "\n")
    markdown = to_markdown(results)
    (out / "round5.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
