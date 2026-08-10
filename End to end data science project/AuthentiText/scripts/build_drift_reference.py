"""Build or verify the validation-only aggregate drift reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from authentitext.drift import DriftError, load_drift_reference
from authentitext.drift_reference import build_drift_reference, render_drift_reference


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation",
        type=Path,
        default=repo_root / "data" / "processed" / "mage_splits" / "id" / "validation.jsonl.gz",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=(
            repo_root
            / "artifacts"
            / "predictions"
            / "id"
            / "validation"
            / "word_tfidf_logistic.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=repo_root / "artifacts" / "baselines" / "id" / "calibration_policy.joblib",
    )
    parser.add_argument(
        "--split-report",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_id_split_report.json",
    )
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_calibration_report.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_drift_reference.json",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        generated = build_drift_reference(
            validation_path=args.validation,
            prediction_path=args.predictions,
            calibration_path=args.calibration,
            split_report_path=args.split_report,
            calibration_report_path=args.calibration_report,
        )
        if args.verify_only:
            saved = load_drift_reference(args.report)
            if saved != generated:
                raise DriftError("Saved drift reference does not match deterministic rebuild")
            print(f"drift reference verified: {saved['identity']['validation']['rows']} rows")
            return 0
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(f"{args.report.suffix}.tmp")
        temporary.write_text(render_drift_reference(generated), encoding="utf-8", newline="\n")
        temporary.replace(args.report)
        backtest = generated["backtest"]
        print(
            "drift reference built:",
            f"{generated['identity']['validation']['rows']} rows,",
            f"false-alert audit {backtest['false_alert_audit']['flagged']}/"
            f"{backtest['false_alert_audit']['windows']},",
            f"domain shifts {backtest['domain_shift']['detected']}/"
            f"{backtest['domain_shift']['domains']}",
        )
    except (DriftError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
