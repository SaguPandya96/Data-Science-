"""Analyze exact and normalized leakage in processed MAGE partitions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from authentitext.data.cleaning import CleaningError, verify_cleaned_outputs
from authentitext.data.leakage import (
    LeakageError,
    analyze_processed_data,
    render_report,
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
        "--input-dir",
        type=Path,
        default=repo_root / "data" / "processed" / "mage_development",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_leakage_report.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the existing report differs from a fresh analysis",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cleaning_report = json.loads(args.cleaning_report.read_text(encoding="utf-8"))
        verify_cleaned_outputs(cleaning_report, args.input_dir)
        report = analyze_processed_data(cleaning_report, args.input_dir)
        rendered = render_report(report)
        if args.check:
            if not args.output.is_file():
                raise LeakageError(f"Leakage report does not exist: {args.output}")
            if args.output.read_text(encoding="utf-8") != rendered:
                raise LeakageError(f"Leakage report is stale: {args.output}")
            print(f"leakage report verified: {args.output}")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
            temporary.write_text(rendered, encoding="utf-8", newline="\n")
            temporary.replace(args.output)
            print(f"leakage report written: {args.output}")
        if report["validation"]["status"] != "pass":
            print("error: leakage report contains validation violations", file=sys.stderr)
            return 1
    except (
        CleaningError,
        LeakageError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
