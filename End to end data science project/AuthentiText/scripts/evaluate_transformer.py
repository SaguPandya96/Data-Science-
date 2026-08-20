"""Evaluate the frozen transformer once on test and MAGE development OOD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from authentitext.modeling.transformer_evaluation import (
    TransformerEvaluationError,
    evaluate_frozen_transformer,
    render_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--test-materialization", type=Path, required=True)
    parser.add_argument("--ood-root", type=Path, required=True)
    parser.add_argument(
        "--ood-manifest", type=Path, default=root / "data/metadata/mage_ood_manifest.json"
    )
    parser.add_argument(
        "--freeze", type=Path, default=root / "data/metadata/transformer_candidate_freeze.json"
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--ood-predictions", type=Path, required=True)
    parser.add_argument("--test-report", type=Path, required=True)
    parser.add_argument("--ood-report", type=Path, required=True)
    return parser.parse_args(argv)


def _write(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(render_report(report), encoding="utf-8", newline="\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
        materialization = json.loads(args.test_materialization.read_text(encoding="utf-8"))
        manifest = json.loads(args.ood_manifest.read_text(encoding="utf-8"))
        test_report, ood_report = evaluate_frozen_transformer(
            test_path=args.test,
            test_materialization=materialization,
            raw_ood_root=args.ood_root,
            ood_manifest=manifest,
            artifact_root=args.artifact_root,
            freeze=freeze,
            test_prediction_path=args.test_predictions,
            ood_prediction_path=args.ood_predictions,
        )
        _write(args.test_report, test_report)
        _write(args.ood_report, ood_report)
        print(
            "frozen transformer evaluation complete:",
            f"test AUC {test_report['metrics']['raw_score']['overall']['roc_auc']},",
            "OOD AUC",
            ood_report["metrics"]["content_deduplicated_combined"]["raw_ranking"]["roc_auc"],
        )
    except (
        KeyError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TransformerEvaluationError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
