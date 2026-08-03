"""CLI entrypoint: run the full sweep and write results to disk.

    python scripts/run_experiment.py --config config.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evasion_gap.pipeline import run_experiment  # noqa: E402
from evasion_gap.plots import plot_robustness  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    config = yaml.safe_load(args.config.read_text())
    result = run_experiment(config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.sweep.to_csv(args.output_dir / "sweep.csv", index=False)
    (args.output_dir / "operating_points.json").write_text(
        json.dumps(
            {
                "operating_points": [op.as_dict() for op in result.operating_points],
                "config": config,
            },
            indent=2,
        )
    )
    plot_robustness(
        result.sweep,
        result.operating_points,
        outfile=args.output_dir / "robustness.png",
    )

    for op in result.operating_points:
        sub = result.sweep[result.sweep["operating_point"] == op.name]
        print(f"\n### {op.name}  (threshold={op.threshold:.4f}, FPR={op.benign_fpr:.1%})\n")
        print(
            sub.drop(columns=["operating_point"]).to_markdown(index=False, floatfmt=".3f")
        )

    print(f"\nWrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
