"""Run or verify the sealed Ghostbuster-to-MAGE overlap gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from authentitext.data.external_overlap import (
    ExternalOverlapError,
    audit_external_overlap,
    render_report,
)
from authentitext.data.ghostbuster import GhostbusterError, verify_ghostbuster_main
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
        "--mage-dir",
        type=Path,
        default=repo_root / "data" / "processed" / "mage_splits" / "id",
    )
    parser.add_argument(
        "--external-manifest",
        type=Path,
        default=repo_root / "data" / "metadata" / "ghostbuster_main_manifest.json",
    )
    parser.add_argument(
        "--external-repository-dir",
        type=Path,
        default=repo_root / "data" / "raw" / "ghostbuster" / "repository",
    )
    parser.add_argument(
        "--external-data",
        type=Path,
        default=repo_root / "data" / "processed" / "ghostbuster" / "main.jsonl.gz",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "data" / "metadata" / "ghostbuster_overlap_report.json",
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.8)
    parser.add_argument("--max-bucket-size", type=int, default=100)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        split_report = json.loads(args.split_report.read_text(encoding="utf-8"))
        external_manifest = json.loads(args.external_manifest.read_text(encoding="utf-8"))
        verify_id_split(split_report, args.mage_dir)
        verify_ghostbuster_main(
            external_manifest,
            args.external_repository_dir,
            args.external_data,
        )
        report = audit_external_overlap(
            split_report,
            args.mage_dir,
            external_manifest,
            args.external_data,
            similarity_threshold=args.similarity_threshold,
            max_bucket_size=args.max_bucket_size,
        )
        rendered = render_report(report)
        if args.check:
            if not args.report.is_file():
                raise ExternalOverlapError(f"Overlap report does not exist: {args.report}")
            if args.report.read_text(encoding="utf-8") != rendered:
                raise ExternalOverlapError(f"Overlap report is stale: {args.report}")
            print(
                "external overlap report verified: "
                f"{report['exclusions']['external_records']} exclusions"
            )
            return 0
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(f"{args.report.suffix}.tmp")
        try:
            temporary.write_text(rendered, encoding="utf-8", newline="\n")
            temporary.replace(args.report)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        print(
            "external overlap report written: "
            f"{report['exclusions']['external_records']} exclusions"
        )
    except (
        ExternalOverlapError,
        GhostbusterError,
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
