"""Run the memory benchmark and write reports/metrics/benchmark_<split>.json and .md.

python scripts/run_benchmark.py            # both splits
python scripts/run_benchmark.py --split dev
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from keel.evaluation.benchmark import run
from keel.evaluation.report import to_markdown

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test", "both"], default="both")
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval.toml"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "metrics"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = ["dev", "test"] if args.split == "both" else [args.split]
    for split in splits:
        results = run(args.config, split)
        (out / f"benchmark_{split}.json").write_text(json.dumps(results, indent=2) + "\n")
        markdown = to_markdown(results)
        (out / f"benchmark_{split}.md").write_text(markdown)
        print(markdown)


if __name__ == "__main__":
    main()
