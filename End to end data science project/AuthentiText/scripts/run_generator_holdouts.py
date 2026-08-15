"""Run, resume, or verify all MAGE leave-one-exact-generator-out experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from authentitext.data.splitting import SplitError, verify_id_split
from authentitext.modeling.generator_holdouts import (
    BaselineError,
    CalibrationError,
    EvaluationError,
    FrozenTestError,
    GeneratorHoldoutError,
    render_report,
    run_generator_holdouts,
    verify_generator_holdouts,
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
        "--regime-report",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_source_regimes.json",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root / "data" / "processed" / "mage_splits" / "id",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=repo_root / "artifacts" / "generalization" / "generator",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=repo_root / "data" / "interim" / "generator_holdouts",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_generator_holdout_report.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify and reuse completed folds from an existing checkpoint report",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify all saved artifacts and recompute prediction metrics",
    )
    return parser.parse_args(argv)


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(render_report(report), encoding="utf-8", newline="\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        split_report = json.loads(args.split_report.read_text(encoding="utf-8"))
        regime_report = json.loads(args.regime_report.read_text(encoding="utf-8"))
        verify_id_split(split_report, args.input_dir)
        if regime_report.get("dataset_id") != split_report.get("dataset_id") or regime_report.get(
            "revision"
        ) != split_report.get("revision"):
            raise GeneratorHoldoutError("Regime and split reports identify different datasets")
        if args.verify_only:
            report = json.loads(args.report.read_text(encoding="utf-8"))
            verify_generator_holdouts(report, args.artifact_dir)
            print(f"verified {len(report['folds'])} generator-holdout folds")
            return 0

        resume_report = None
        if args.resume and args.report.is_file():
            resume_report = json.loads(args.report.read_text(encoding="utf-8"))
        report = run_generator_holdouts(
            regime_report=regime_report,
            input_root=args.input_dir,
            artifact_root=args.artifact_dir,
            work_root=args.work_dir,
            progress=lambda message: print(message, flush=True),
            checkpoint=lambda partial: _write_report(args.report, partial),
            resume_report=resume_report,
        )
        verify_generator_holdouts(report, args.artifact_dir)
        _write_report(args.report, report)
        print(f"completed and verified {len(report['folds'])} generator-holdout folds")
    except (
        BaselineError,
        CalibrationError,
        EvaluationError,
        FrozenTestError,
        GeneratorHoldoutError,
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
