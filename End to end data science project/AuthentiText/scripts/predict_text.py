"""Run one local AuthentiText prediction from a UTF-8 text file or stdin."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from authentitext.inference import AuthentiTextPredictor, PredictionError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="UTF-8 input file; omit to read text from standard input",
    )
    parser.add_argument(
        "--training-report",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_baseline_training_report.json",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        text = args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read()
        predictor = AuthentiTextPredictor.from_reports(
            training_report_path=args.training_report,
            calibration_report_path=args.calibration_report,
            artifact_root=args.artifact_dir,
        )
        print(json.dumps(predictor.predict(text), ensure_ascii=False, indent=2, sort_keys=True))
    except (OSError, UnicodeDecodeError, PredictionError) as error:
        code = error.code if isinstance(error, PredictionError) else "input_read_error"
        print(json.dumps({"error": {"code": code, "message": str(error)}}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
