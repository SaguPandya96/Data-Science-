"""Run the bounded near-duplicate audit for processed MAGE data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from authentitext.data.cleaning import CleaningError, verify_cleaned_outputs
from authentitext.data.near_duplicates import (
    NearDuplicateError,
    audit_near_duplicates,
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
        default=repo_root / "data" / "metadata" / "mage_near_duplicate_report.json",
    )
    parser.add_argument("--per-source", type=int, default=20)
    parser.add_argument("--similarity-threshold", type=float, default=0.8)
    parser.add_argument("--max-bucket-size", type=int, default=100)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the existing report differs from a fresh audit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cleaning_report = json.loads(args.cleaning_report.read_text(encoding="utf-8"))
        verify_cleaned_outputs(cleaning_report, args.input_dir)
        report = audit_near_duplicates(
            cleaning_report,
            args.input_dir,
            per_source=args.per_source,
            similarity_threshold=args.similarity_threshold,
            max_bucket_size=args.max_bucket_size,
        )
        rendered = render_report(report)
        if args.check:
            if not args.output.is_file():
                raise NearDuplicateError(f"Near-duplicate report does not exist: {args.output}")
            if args.output.read_text(encoding="utf-8") != rendered:
                raise NearDuplicateError(f"Near-duplicate report is stale: {args.output}")
            print(f"near-duplicate report verified: {args.output}")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
            temporary.write_text(rendered, encoding="utf-8", newline="\n")
            temporary.replace(args.output)
            print(f"near-duplicate report written: {args.output}")
    except (
        CleaningError,
        NearDuplicateError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
