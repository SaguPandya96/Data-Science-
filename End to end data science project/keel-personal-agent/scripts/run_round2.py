"""Round 2 of the memory benchmark: embedding search.

Chooses the embedding weight on the dev wordings, then scores every arm on the second
held-out set. Writes reports/metrics/round2.json and round2.md.

    python scripts/run_round2.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from keel.evaluation.round2 import run_round2, to_markdown

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval.toml"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "metrics"))
    args = parser.parse_args()

    results = run_round2(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "round2.json").write_text(json.dumps(results, indent=2) + "\n")
    markdown = to_markdown(results)
    (out / "round2.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
