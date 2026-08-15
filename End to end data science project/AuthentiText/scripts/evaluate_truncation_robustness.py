"""Run or verify the frozen MAGE prefix-truncation robustness experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from authentitext.data.splitting import SplitError, verify_id_split
from authentitext.modeling.baselines import BaselineError, verify_baselines
from authentitext.modeling.calibration import CalibrationError, verify_calibration_policy
from authentitext.modeling.truncation_robustness import (
    TruncationRobustnessError,
    evaluate_truncation_robustness,
    render_report,
    verify_truncation_robustness,
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
        "--validation-report",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_baseline_validation_report.json",
    )
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_calibration_report.json",
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
        "--validation-prediction-dir",
        type=Path,
        default=repo_root / "artifacts" / "predictions" / "id" / "validation",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=(
            repo_root
            / "artifacts"
            / "predictions"
            / "robustness"
            / "mage_truncation_pairs.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_truncation_robustness_report.json",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify paired predictions and recompute every condition metric",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        split_report = json.loads(args.split_report.read_text(encoding="utf-8"))
        training_report = json.loads(args.training_report.read_text(encoding="utf-8"))
        validation_report = json.loads(args.validation_report.read_text(encoding="utf-8"))
        calibration_report = json.loads(args.calibration_report.read_text(encoding="utf-8"))
        verify_id_split(split_report, args.input_dir)
        verify_baselines(training_report, args.artifact_dir)
        validation_model = next(
            model
            for model in validation_report["models"]
            if model["model_type"] == "word_tfidf_logistic"
        )
        calibration_path = args.artifact_dir / calibration_report["artifact"]["relative_path"]
        verify_calibration_policy(
            calibration_report,
            args.validation_prediction_dir / validation_model["predictions"]["relative_path"],
            calibration_path,
        )
        if args.verify_only:
            report = json.loads(args.report.read_text(encoding="utf-8"))
            verify_truncation_robustness(report, args.predictions, calibration_path)
            print(
                f"MAGE truncation robustness verified: {report['predictions']['rows']} paired rows"
            )
            return 0

        test_input = next(
            output for output in split_report["outputs"] if output["partition"] == "test"
        )
        base_model_identity = validation_model["model_artifact"]
        report = evaluate_truncation_robustness(
            test_path=args.input_dir / test_input["relative_path"],
            test_identity={
                "relative_path": test_input["relative_path"],
                "rows": test_input["rows_written"],
                "bytes": test_input["bytes"],
                "sha256": test_input["sha256"],
            },
            base_model_path=args.artifact_dir / base_model_identity["relative_path"],
            base_model_identity=base_model_identity,
            calibration_path=calibration_path,
            calibration_identity=calibration_report["artifact"],
            prediction_path=args.predictions,
            dataset_id=split_report["dataset_id"],
            revision=split_report["revision"],
        )
        verify_truncation_robustness(report, args.predictions, calibration_path)
        report["validation"]["prediction_file_verified"] = True
        report["validation"]["metrics_recomputed_from_predictions"] = True
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(f"{args.report.suffix}.tmp")
        temporary.write_text(render_report(report), encoding="utf-8", newline="\n")
        temporary.replace(args.report)
        print(f"MAGE truncation robustness evaluated: {report['predictions']['rows']} paired rows")
    except (
        BaselineError,
        CalibrationError,
        SplitError,
        TruncationRobustnessError,
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
