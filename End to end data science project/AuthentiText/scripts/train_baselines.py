"""Train or verify AuthentiText baseline model artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from authentitext.data.splitting import SplitError, verify_id_split
from authentitext.modeling.baselines import (
    BaselineError,
    render_report,
    train_baselines,
    verify_baselines,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-report",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_id_split_report.json",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root / "data" / "processed" / "mage_splits" / "id",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=repo_root / "artifacts" / "baselines" / "id",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_baseline_training_report.json",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing model artifacts against the training report",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        split_report = json.loads(args.split_report.read_text(encoding="utf-8"))
        verify_id_split(split_report, args.input_dir)
        if args.verify_only:
            report = json.loads(args.report.read_text(encoding="utf-8"))
            verify_baselines(report, args.artifact_dir)
            print(f"baseline artifacts verified: {len(report['artifacts'])} models")
            return 0

        train_input = next(
            output for output in split_report["outputs"] if output["partition"] == "train"
        )
        report = train_baselines(
            train_path=args.input_dir / train_input["relative_path"],
            artifact_root=args.artifact_dir,
            dataset_id=split_report["dataset_id"],
            revision=split_report["revision"],
            input_sha256=train_input["sha256"],
        )
        verify_baselines(report, args.artifact_dir)
        report["validation"]["artifacts_reload_and_score"] = True
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(f"{args.report.suffix}.tmp")
        temporary.write_text(render_report(report), encoding="utf-8", newline="\n")
        temporary.replace(args.report)
        print(f"trained and verified {len(report['artifacts'])} baseline models")
    except (
        BaselineError,
        SplitError,
        OSError,
        StopIteration,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
