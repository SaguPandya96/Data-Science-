"""Rebuild the sealed transformer train input from the pinned raw train file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from authentitext.data.cleaning import CleaningError, clean_partition
from authentitext.data.transformer_train import (
    TransformerTrainError,
    load_train_decisions,
    materialize_transformer_train,
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
    parser.add_argument(
        "--decisions",
        type=Path,
        default=repo_root / "data" / "metadata" / "transformer_train_decisions.json",
    )
    parser.add_argument("--input-dir", type=Path, default=repo_root / "data" / "raw")
    parser.add_argument(
        "--cleaned-output",
        type=Path,
        default=repo_root / "build" / "transformer-data" / "cleaned-train.jsonl.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "build" / "transformer-data" / "train.jsonl.gz",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "build" / "transformer-data" / "materialization-report.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        decisions = load_train_decisions(args.decisions)
        if decisions["dataset_id"] != manifest["dataset_id"]:
            raise TransformerTrainError("Train decisions do not match the dataset")
        if decisions["revision"] != manifest["revision"]:
            raise TransformerTrainError("Train decisions do not match the revision")

        entries = {entry["name"]: entry for entry in manifest["files"]}
        train_entry = entries.get("train")
        if train_entry is None:
            raise TransformerTrainError("Manifest does not contain a train file")
        raw_train_path = args.input_dir / Path(train_entry["relative_path"])
        verify_file(raw_train_path, train_entry)

        raw_to_target = manifest["label_semantics"]["canonical_machine_positive_mapping"]
        clean_report = clean_partition(
            input_path=raw_train_path,
            output_path=args.cleaned_output,
            dataset_id=manifest["dataset_id"],
            revision=manifest["revision"],
            partition="train",
            raw_to_target=raw_to_target,
        )
        expected_input = decisions["expected_input"]
        if clean_report["rows_written"] != expected_input["rows"]:
            raise TransformerTrainError(
                "Cleaned train row count does not match its audited identity"
            )

        report = materialize_transformer_train(args.cleaned_output, args.output, decisions)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary_report = args.report.with_suffix(f"{args.report.suffix}.tmp")
        temporary_report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary_report.replace(args.report)
        print(
            f"materialized transformer train: {report['rows_written']} rows, "
            f"content SHA-256 {report['output_content_sha256']}"
        )
    except (
        CleaningError,
        IntegrityError,
        ManifestError,
        TransformerTrainError,
        KeyError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
