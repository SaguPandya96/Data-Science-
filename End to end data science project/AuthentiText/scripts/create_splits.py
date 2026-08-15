"""Create or verify leakage-sanitized MAGE in-distribution splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from authentitext.data.cleaning import CleaningError, verify_cleaned_outputs
from authentitext.data.splitting import (
    SplitError,
    create_id_split,
    render_report,
    verify_id_split,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cleaning-report",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_cleaning_report.json",
    )
    parser.add_argument(
        "--near-duplicate-report",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_near_duplicate_report.json",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root / "data" / "processed" / "mage_development",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "processed" / "mage_splits" / "id",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_id_split_report.json",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing outputs against the split report",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cleaning_report = json.loads(args.cleaning_report.read_text(encoding="utf-8"))
        verify_cleaned_outputs(cleaning_report, args.input_dir)
        if args.verify_only:
            report = json.loads(args.report.read_text(encoding="utf-8"))
            if report["dataset_id"] != cleaning_report["dataset_id"]:
                raise SplitError("Split report dataset does not match cleaning report")
            if report["revision"] != cleaning_report["revision"]:
                raise SplitError("Split report revision does not match cleaning report")
            verify_id_split(report, args.output_dir)
            print(f"ID split outputs verified: {report['aggregate']['output_rows']} rows")
            return 0

        near_report = json.loads(args.near_duplicate_report.read_text(encoding="utf-8"))
        report = create_id_split(cleaning_report, near_report, args.input_dir, args.output_dir)
        verify_id_split(report, args.output_dir)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(f"{args.report.suffix}.tmp")
        temporary.write_text(render_report(report), encoding="utf-8", newline="\n")
        temporary.replace(args.report)
        print(
            f"created ID split: {report['aggregate']['input_rows']} input, "
            f"{report['aggregate']['output_rows']} output, "
            f"{report['aggregate']['dropped_rows']} dropped"
        )
    except (
        CleaningError,
        SplitError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
