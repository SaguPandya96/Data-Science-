"""Prepare, finalize, or verify the qualitative Ghostbuster error review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from authentitext.data.cleaning import sha256_file
from authentitext.modeling.error_review import (
    ErrorReviewError,
    build_error_review,
    render_report,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        default=repo_root / "data" / "metadata" / "ghostbuster_evaluation_report.json",
    )
    parser.add_argument(
        "--external-data",
        type=Path,
        default=repo_root / "data" / "processed" / "ghostbuster" / "main.jsonl.gz",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=(
            repo_root
            / "artifacts"
            / "predictions"
            / "external"
            / "ghostbuster"
            / "word_tfidf_policy.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--packet",
        type=Path,
        default=(repo_root / "artifacts" / "error_review" / "ghostbuster_review_packet.jsonl.gz"),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=(repo_root / "data" / "interim" / "ghostbuster_error_review_annotations.json"),
    )
    parser.add_argument(
        "--selection-preview",
        type=Path,
        default=repo_root / "data" / "interim" / "ghostbuster_error_review_selection.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "data" / "metadata" / "ghostbuster_error_review_report.json",
    )
    return parser.parse_args(argv)


def _write_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8", newline="\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        evaluation_report = json.loads(args.evaluation_report.read_text(encoding="utf-8"))
        annotations = None
        if args.finalize or args.verify_only:
            annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
        report, template = build_error_review(
            evaluation_report=evaluation_report,
            evaluation_report_identity={
                "relative_path": args.evaluation_report.name,
                "bytes": args.evaluation_report.stat().st_size,
                "sha256": sha256_file(args.evaluation_report),
            },
            external_path=args.external_data,
            prediction_path=args.predictions,
            packet_path=args.packet,
            annotations=annotations,
        )
        rendered = render_report(report)
        if args.prepare:
            if args.annotations.exists():
                raise ErrorReviewError(
                    f"Annotation template already exists; refusing to overwrite: {args.annotations}"
                )
            _write_atomic(args.annotations, json.dumps(template, indent=2) + "\n")
            _write_atomic(args.selection_preview, rendered)
            print(f"prepared {len(template)} review records and an annotation template")
            return 0
        if args.verify_only:
            if args.report.read_text(encoding="utf-8") != rendered:
                raise ErrorReviewError("Committed error-review report is stale")
            print(f"verified {report['sample']['rows']} qualitative review annotations")
            return 0
        _write_atomic(args.report, rendered)
        print(f"finalized {report['sample']['rows']} qualitative review annotations")
    except (
        ErrorReviewError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
