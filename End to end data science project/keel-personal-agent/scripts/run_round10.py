"""Round 10 of the memory benchmark: MiniLM below weight 4.

Reruns round 9's procedure with weights from 0.5 to 4 for the two MiniLM encoders, under
the shipped int8 reranker. Writes reports/metrics/round10.json and round10.md, and
machine-dependent timings to reports/metrics/round10_timing.json.

    python scripts/run_round10.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from keel.evaluation.round9 import run_round9, to_markdown

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval.toml"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "metrics"))
    args = parser.parse_args()

    results, timings = run_round9(args.config, section="round10")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "round10.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "round10.md").write_text(to_markdown(results, section="round10"))
    (out / "round10_timing.json").write_text(json.dumps(timings, indent=2) + "\n")
    print(to_markdown(results, timings, section="round10"))


if __name__ == "__main__":
    main()
