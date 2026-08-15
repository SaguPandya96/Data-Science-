"""Clean pinned MAGE development data into canonical compressed records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from authentitext.data.cleaning import (
    CleaningError,
    clean_manifest,
    render_report,
    verify_cleaned_outputs,
)

if __package__:
    from .download_data import IntegrityError, ManifestError, load_manifest, verify_file
else:
    from download_data import IntegrityError, ManifestError, load_manifest, verify_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_development_manifest.json",
    )
    parser.add_argument("--input-dir", type=Path, default=repo_root / "data" / "raw")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "processed" / "mage_development",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_cleaning_report.json",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing processed outputs against the cleaning report",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.verify_only:
            report = json.loads(args.report.read_text(encoding="utf-8"))
            if report["dataset_id"] != manifest["dataset_id"]:
                raise CleaningError("Cleaning report dataset does not match manifest")
            if report["revision"] != manifest["revision"]:
                raise CleaningError("Cleaning report revision does not match manifest")
            verify_cleaned_outputs(report, args.output_dir)
            print(f"cleaned outputs verified: {report['aggregate']['rows_written']} rows")
            return 0

        for entry in manifest["files"]:
            verify_file(args.input_dir / Path(entry["relative_path"]), entry)
        report = clean_manifest(manifest, args.input_dir, args.output_dir)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(f"{args.report.suffix}.tmp")
        temporary.write_text(render_report(report), encoding="utf-8", newline="\n")
        temporary.replace(args.report)
        print(
            f"cleaned {report['aggregate']['rows_seen']} rows: "
            f"wrote {report['aggregate']['rows_written']}, "
            f"excluded {report['aggregate']['rows_excluded']}"
        )
    except (
        CleaningError,
        IntegrityError,
        ManifestError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
