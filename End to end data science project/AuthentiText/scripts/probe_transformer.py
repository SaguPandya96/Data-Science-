"""Run and record the prespecified train-only transformer throughput probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from authentitext.modeling.transformer_probe import TransformerProbeError, run_probe


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=repo_root / "build" / "transformer-data" / "train.jsonl.gz",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=repo_root / "build" / "transformer-checkpoint",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "build" / "transformer-probe" / "report.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_probe(args.train, args.checkpoint_dir)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(f"{args.report.suffix}.tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(args.report)
        estimate = report["estimate"]
        print(
            f"transformer probe: {report['measurement']['rows_per_second']:.2f} rows/s; "
            f"full run viable={estimate['full_run_viable']}"
        )
    except (OSError, TransformerProbeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
