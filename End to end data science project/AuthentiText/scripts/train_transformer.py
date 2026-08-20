"""Train, evaluate, calibrate, and package the frozen BERT-Tiny candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from authentitext.modeling.calibration import CalibrationError, fit_calibration_policy
from authentitext.modeling.transformer_training import (
    TransformerTrainingError,
    train_and_evaluate,
)


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--calibration-artifact", type=Path, required=True)
    parser.add_argument(
        "--training-report",
        type=Path,
        default=repo_root / "build" / "transformer-full" / "training-report.json",
    )
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=repo_root / "build" / "transformer-full" / "calibration-report.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        training_report = train_and_evaluate(
            train_path=args.train,
            validation_path=args.validation,
            checkpoint_dir=args.checkpoint_dir,
            model_output_dir=args.model_dir,
            prediction_path=args.prediction,
        )
        _write_report(args.training_report, training_report)

        calibration_report = fit_calibration_policy(
            prediction_path=args.prediction,
            prediction_identity=training_report["predictions"],
            base_model_identity=training_report["model_artifact"],
            artifact_path=args.calibration_artifact,
            dataset_id=training_report["dataset_id"],
            revision=training_report["revision"],
        )
        calibration_report["validation"]["source_prediction_verified"] = True
        calibration_report["validation"]["test_data_read"] = False
        _write_report(args.calibration_report, calibration_report)
        overall = training_report["metrics"]["overall"]
        print(
            "transformer validation: "
            f"ROC AUC {overall['roc_auc']}, AP {overall['average_precision']}; "
            f"calibration {calibration_report['method_selection']['selected_method']}"
        )
    except (CalibrationError, OSError, TransformerTrainingError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
