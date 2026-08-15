"""Evaluate or verify the frozen baseline on MAGE development OOD files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from authentitext.modeling.ood_evaluation import (
    OodEvaluationError,
    evaluate_ood,
    load_ood_records,
    render_report,
    verify_ood,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_ood_manifest.json",
    )
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_calibration_report.json",
    )
    parser.add_argument("--raw-root", type=Path, default=repo_root / "data" / "raw")
    parser.add_argument(
        "--artifact-root", type=Path, default=repo_root / "artifacts" / "baselines" / "id"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=repo_root / "artifacts" / "predictions" / "ood" / "word_tfidf_policy.jsonl.gz",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_ood_evaluation_report.json",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        calibration_report = json.loads(args.calibration_report.read_text(encoding="utf-8"))
        calibration_path = args.artifact_root / calibration_report["artifact"]["relative_path"]
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if args.verify_only:
            report = json.loads(args.report.read_text(encoding="utf-8"))
            records, input_identities = load_ood_records(args.raw_root, manifest)
            if (
                input_identities != report.get("inputs")
                or len(records) != report["predictions"]["rows"]
            ):
                raise OodEvaluationError("Saved OOD report does not match raw inputs")
            verify_ood(report, args.predictions, calibration_path)
            print(f"OOD evaluation verified: {report['predictions']['rows']} rows")
            return 0
        report = evaluate_ood(
            raw_root=args.raw_root,
            manifest=manifest,
            base_model_path=args.artifact_root / calibration_report["base_model"]["relative_path"],
            base_model_identity=calibration_report["base_model"],
            calibration_path=calibration_path,
            calibration_identity=calibration_report["artifact"],
            prediction_path=args.predictions,
        )
        verify_ood(report, args.predictions, calibration_path)
        report["validation"]["prediction_file_verified"] = True
        report["validation"]["metrics_recomputed_from_predictions"] = True
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(f"{args.report.suffix}.tmp")
        temporary.write_text(render_report(report), encoding="utf-8", newline="\n")
        temporary.replace(args.report)
        combined = report["metrics"]["content_deduplicated_combined"]
        print(
            "OOD evaluation complete:",
            f"{report['predictions']['rows']} rows,",
            f"{combined['rows']} unique texts,",
            f"AUC {combined['raw_ranking']['roc_auc']},",
            f"uncertain {combined['policy']['uncertain_rate']}",
        )
    except (
        KeyError,
        OodEvaluationError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
