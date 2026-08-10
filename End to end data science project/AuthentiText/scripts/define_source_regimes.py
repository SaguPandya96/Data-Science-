"""Define or verify source-disjoint MAGE evaluation regimes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from authentitext.data.regimes import RegimeError, define_source_regimes, render_report
from authentitext.data.splitting import SplitError, verify_id_split


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-report",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_id_split_report.json",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root / "data" / "processed" / "mage_splits" / "id",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_source_regimes.json",
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
        split_report = json.loads(args.split_report.read_text(encoding="utf-8"))
        verify_id_split(split_report, args.input_dir)
        rendered = render_report(define_source_regimes(split_report, args.input_dir))
        if args.check:
            if not args.output.is_file():
                raise RegimeError(f"Source-regime report does not exist: {args.output}")
            if args.output.read_text(encoding="utf-8") != rendered:
                raise RegimeError(f"Source-regime report is stale: {args.output}")
            print(f"source-regime report verified: {args.output}")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
            temporary.write_text(rendered, encoding="utf-8", newline="\n")
            temporary.replace(args.output)
            print(f"source-regime report written: {args.output}")
    except (
        RegimeError,
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
