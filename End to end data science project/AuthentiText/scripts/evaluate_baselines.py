"""Evaluate or verify baselines on sanitized validation data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from authentitext.data.splitting import SplitError, verify_id_split
from authentitext.modeling.baselines import BaselineError, verify_baselines
from authentitext.modeling.evaluation import (
    EvaluationError,
    evaluate_baselines,
    render_report,
    verify_evaluation,
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
        "--training-report",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_baseline_training_report.json",
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
        "--prediction-dir",
        type=Path,
        default=repo_root / "artifacts" / "predictions" / "id" / "validation",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_baseline_validation_report.json",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify predictions and recompute metrics from the existing report",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        split_report = json.loads(args.split_report.read_text(encoding="utf-8"))
        training_report = json.loads(args.training_report.read_text(encoding="utf-8"))
        verify_id_split(split_report, args.input_dir)
        verify_baselines(training_report, args.artifact_dir)
        if args.verify_only:
            report = json.loads(args.report.read_text(encoding="utf-8"))
            if report["partition"] != "validation":
                raise EvaluationError("Expected a validation evaluation report")
            verify_evaluation(report, args.prediction_dir)
            print(f"baseline validation evaluation verified: {len(report['models'])} models")
            return 0

        evaluation_input = next(
            output for output in split_report["outputs"] if output["partition"] == "validation"
        )
        report = evaluate_baselines(
            evaluation_path=args.input_dir / evaluation_input["relative_path"],
            evaluation_partition="validation",
            evaluation_identity={
                "relative_path": evaluation_input["relative_path"],
                "rows": evaluation_input["rows_written"],
                "bytes": evaluation_input["bytes"],
                "sha256": evaluation_input["sha256"],
            },
            training_report=training_report,
            artifact_root=args.artifact_dir,
            prediction_root=args.prediction_dir,
        )
        verify_evaluation(report, args.prediction_dir)
        report["validation"]["prediction_files_verified"] = True
        report["validation"]["metrics_recomputed_from_predictions"] = True
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(f"{args.report.suffix}.tmp")
        temporary.write_text(render_report(report), encoding="utf-8", newline="\n")
        temporary.replace(args.report)
        print(f"evaluated {len(report['models'])} baselines on validation")
    except (
        BaselineError,
        EvaluationError,
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
