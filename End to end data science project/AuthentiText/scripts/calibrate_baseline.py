"""Fit or verify validation-only calibration and abstention policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from authentitext.modeling.calibration import (
    CalibrationError,
    fit_calibration_policy,
    render_report,
    verify_calibration_policy,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_baseline_validation_report.json",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=repo_root / "artifacts" / "predictions" / "id" / "validation",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=repo_root / "artifacts" / "baselines" / "id",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_calibration_report.json",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the calibration artifact and recompute audit metrics",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        evaluation_report = json.loads(args.evaluation_report.read_text(encoding="utf-8"))
        model = next(
            item
            for item in evaluation_report["models"]
            if item["model_type"] == "word_tfidf_logistic"
        )
        prediction_path = args.prediction_dir / model["predictions"]["relative_path"]
        artifact_path = args.artifact_dir / "calibration_policy.joblib"
        if args.verify_only:
            report = json.loads(args.report.read_text(encoding="utf-8"))
            verify_calibration_policy(report, prediction_path, artifact_path)
            print(f"calibration policy verified: {report['method_selection']['selected_method']}")
            return 0

        report = fit_calibration_policy(
            prediction_path=prediction_path,
            prediction_identity=model["predictions"],
            base_model_identity=model["model_artifact"],
            artifact_path=artifact_path,
            dataset_id=evaluation_report["dataset_id"],
            revision=evaluation_report["revision"],
        )
        verify_calibration_policy(report, prediction_path, artifact_path)
        report["validation"]["artifact_reload_and_score"] = True
        report["validation"]["source_prediction_verified"] = True
        report["validation"]["audit_metrics_recomputed"] = True
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(f"{args.report.suffix}.tmp")
        temporary.write_text(render_report(report), encoding="utf-8", newline="\n")
        temporary.replace(args.report)
        print(
            "calibration policy fitted: "
            f"{report['method_selection']['selected_method']}, "
            f"thresholds {report['thresholds']['human_threshold']} / "
            f"{report['thresholds']['machine_threshold']}"
        )
    except (
        CalibrationError,
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
