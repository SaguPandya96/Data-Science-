"""Run the evaluation and write results to disk.

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
from evasion_gap.plots import plot_defense_effect, plot_recall_by_threshold  # noqa: E402


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

    shipping = config["eval"]["reporting_operating_point"]
    plot_recall_by_threshold(
        result.sweep, result.operating_points, outfile=args.output_dir / "recall_by_threshold.png"
    )
    plot_defense_effect(
        result.sweep, shipping, outfile=args.output_dir / "defense_effect.png"
    )

    for op in result.operating_points:
        print(f"\n### {op.name}  threshold={op.threshold:.4f}  benign FPR={op.benign_fpr:.1%}")
        for split in ["toxic", "benign"]:
            sub = result.sweep[
                (result.sweep["operating_point"] == op.name) & (result.sweep["split"] == split)
            ]
            table = sub.pivot(index="attack", columns="defense", values="rate")
            print(f"\n{split}:")
            print(table.to_markdown(floatfmt=".3f"))

    print(f"\nWrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
