"""Record the resource and dependency gates for the pinned transformer experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from authentitext.modeling.transformer_preflight import (
    TransformerPreflightError,
    build_preflight_report,
    render_report,
    verify_preflight_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data" / "metadata" / "transformer_preflight_report.json",
    )
    parser.add_argument("--stdout", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_preflight_report(Path(__file__).resolve().parents[1])
        verify_preflight_report(report)
        rendered = render_report(report)
        if args.stdout:
            print(rendered, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
            temporary.write_text(rendered, encoding="utf-8", newline="\n")
            temporary.replace(args.output)
            print(f"transformer preflight: {report['status']} ({len(report['blockers'])} blockers)")
    except (OSError, TransformerPreflightError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
