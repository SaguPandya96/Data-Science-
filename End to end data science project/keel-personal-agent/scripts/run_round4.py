"""Round 4 of the memory benchmark: transformer sentence encoders.

Downloads the two encoders (pinned by SHA-256) on first use, chooses the encoder and weight
on the dev wordings, then scores every arm on the third held-out set. Writes
reports/metrics/round4.json and round4.md, and machine-dependent timings to
reports/metrics/round4_timing.json.

    python scripts/run_round4.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from keel.evaluation.round4 import run_round4, to_markdown

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval.toml"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "metrics"))
    args = parser.parse_args()

    results, timings = run_round4(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "round4.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "round4.md").write_text(to_markdown(results))
    (out / "round4_timing.json").write_text(json.dumps(timings, indent=2) + "\n")
    print(to_markdown(results, timings))


if __name__ == "__main__":
    main()
