"""Score Ghostbuster once or verify its frozen external evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from authentitext.data.cleaning import sha256_file
from authentitext.data.ghostbuster import GhostbusterError, verify_ghostbuster_main
from authentitext.modeling.baselines import BaselineError, verify_baselines
from authentitext.modeling.calibration import CalibrationError, verify_calibration_policy
from authentitext.modeling.external_evaluation import (
    ExternalEvaluationError,
    evaluate_ghostbuster_external,
    verify_ghostbuster_external,
)
from authentitext.modeling.frozen_test import render_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external-manifest",
        type=Path,
        default=repo_root / "data" / "metadata" / "ghostbuster_main_manifest.json",
    )
    parser.add_argument(
        "--external-repository-dir",
        type=Path,
        default=repo_root / "data" / "raw" / "ghostbuster" / "repository",
    )
    parser.add_argument(
        "--external-data",
        type=Path,
        default=repo_root / "data" / "processed" / "ghostbuster" / "main.jsonl.gz",
    )
    parser.add_argument(
        "--overlap-report",
        type=Path,
        default=repo_root / "data" / "metadata" / "ghostbuster_overlap_report.json",
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
        "--prediction",
        type=Path,
        default=(
            repo_root
            / "artifacts"
            / "predictions"
            / "external"
            / "ghostbuster"
            / "word_tfidf_policy.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "data" / "metadata" / "ghostbuster_evaluation_report.json",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = json.loads(args.external_manifest.read_text(encoding="utf-8"))
        overlap_report = json.loads(args.overlap_report.read_text(encoding="utf-8"))
        training_report = json.loads(args.training_report.read_text(encoding="utf-8"))
        validation_report = json.loads(args.validation_report.read_text(encoding="utf-8"))
        calibration_report = json.loads(args.calibration_report.read_text(encoding="utf-8"))
        verify_ghostbuster_main(manifest, args.external_repository_dir, args.external_data)
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
            verify_ghostbuster_external(
                report,
                prediction_path=args.prediction,
                calibration_path=calibration_path,
                external_path=args.external_data,
                overlap_report=overlap_report,
                overlap_path=args.overlap_report,
            )
            print(f"Ghostbuster external evaluation verified: {report['predictions']['rows']} rows")
            return 0
        if args.report.exists():
            raise ExternalEvaluationError(
                "External evaluation report already exists; use --verify-only"
            )

        base_model_identity = validation_model["model_artifact"]
        report = evaluate_ghostbuster_external(
            test_path=args.external_data,
            test_identity={
                "relative_path": "ghostbuster/main.jsonl.gz",
                "rows": manifest["output"]["rows"],
                "bytes": manifest["output"]["bytes"],
                "sha256": manifest["output"]["sha256"],
            },
            overlap_report=overlap_report,
            overlap_identity={
                "relative_path": args.overlap_report.name,
                "bytes": args.overlap_report.stat().st_size,
                "sha256": sha256_file(args.overlap_report),
            },
            base_model_path=args.artifact_dir / base_model_identity["relative_path"],
            base_model_identity=base_model_identity,
            calibration_path=calibration_path,
            calibration_identity=calibration_report["artifact"],
            prediction_path=args.prediction,
            dataset_id=manifest["dataset_id"],
            revision=manifest["revision"],
        )
        verify_ghostbuster_external(
            report,
            prediction_path=args.prediction,
            calibration_path=calibration_path,
            external_path=args.external_data,
            overlap_report=overlap_report,
            overlap_path=args.overlap_report,
        )
        report["validation"]["prediction_file_verified"] = True
        report["validation"]["metrics_recomputed_from_predictions"] = True
        report["validation"]["prediction_ids_match_overlap_gated_external_ids"] = True
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(f"{args.report.suffix}.tmp")
        try:
            temporary.write_text(render_report(report), encoding="utf-8", newline="\n")
            temporary.replace(args.report)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        print(f"Ghostbuster externally evaluated once: {report['predictions']['rows']} rows")
    except (
        BaselineError,
        CalibrationError,
        ExternalEvaluationError,
        GhostbusterError,
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
