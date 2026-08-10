"""Create a deterministic, text-free aggregate profile of acquired CSV data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__:
    from .download_data import IntegrityError, ManifestError, load_manifest, verify_file
else:
    from download_data import IntegrityError, ManifestError, load_manifest, verify_file


PROFILE_SCHEMA_VERSION = 1
PROFILER_VERSION = 2
REQUIRED_COLUMNS = ("text", "label", "src")


class ProfileError(RuntimeError):
    """Raised when profiling configuration is missing or inconsistent."""


def _length_summary(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {}
    ordered = sorted(values)

    def nearest_percentile(percentile: float) -> int:
        index = int((len(ordered) - 1) * percentile + 0.5)
        return ordered[index]

    middle = len(ordered) // 2
    if len(ordered) % 2:
        median: int | float = ordered[middle]
    else:
        median = (ordered[middle - 1] + ordered[middle]) / 2

    return {
        "min": ordered[0],
        "p05": nearest_percentile(0.05),
        "p25": nearest_percentile(0.25),
        "median": median,
        "p75": nearest_percentile(0.75),
        "p95": nearest_percentile(0.95),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 3),
    }


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _label_configuration(manifest: dict[str, Any]) -> tuple[dict[str, str], dict[str, int]]:
    semantics = manifest.get("label_semantics")
    if not isinstance(semantics, dict):
        raise ProfileError("Manifest is missing label_semantics")
    raw_to_semantic = semantics.get("raw_to_semantic")
    canonical_mapping = semantics.get("canonical_machine_positive_mapping")
    if not isinstance(raw_to_semantic, dict) or not raw_to_semantic:
        raise ProfileError("Manifest has invalid raw_to_semantic labels")
    if not isinstance(canonical_mapping, dict) or not canonical_mapping:
        raise ProfileError("Manifest has invalid canonical label mapping")
    if set(raw_to_semantic) != set(canonical_mapping):
        raise ProfileError("Raw and canonical label mappings must have the same keys")
    if set(canonical_mapping.values()) != {0, 1}:
        raise ProfileError("Canonical labels must contain both 0 and 1")
    return raw_to_semantic, canonical_mapping


def build_profile(manifest: dict[str, Any], input_root: Path) -> dict[str, Any]:
    """Profile every manifest file without retaining or emitting source text."""
    raw_to_semantic, canonical_mapping = _label_configuration(manifest)
    file_profiles: list[dict[str, Any]] = []
    violations: list[str] = []

    aggregate_labels: Counter[str] = Counter()
    aggregate_semantics: Counter[str] = Counter()
    aggregate_canonical: Counter[str] = Counter()
    aggregate_sources: Counter[str] = Counter()
    aggregate_characters: list[int] = []
    aggregate_tokens: list[int] = []
    text_occurrences: defaultdict[str, list[tuple[str, str, str]]] = defaultdict(list)

    for entry in manifest["files"]:
        path = input_root / Path(*Path(entry["relative_path"]).parts)
        verify_file(path, entry)

        label_counts: Counter[str] = Counter()
        semantic_counts: Counter[str] = Counter()
        canonical_counts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        null_values: Counter[str] = Counter()
        blank_values: Counter[str] = Counter()
        invalid_labels: Counter[str] = Counter()
        character_lengths: list[int] = []
        whitespace_tokens: list[int] = []
        malformed_rows = 0
        row_count = 0

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
                raise ProfileError(
                    f"{path} has columns {reader.fieldnames!r}; expected {REQUIRED_COLUMNS!r}"
                )

            for row in reader:
                row_count += 1
                if None in row:
                    malformed_rows += 1

                for column in REQUIRED_COLUMNS:
                    value = row.get(column)
                    if value is None:
                        null_values[column] += 1
                    elif not value.strip():
                        blank_values[column] += 1

                text = row.get("text") or ""
                raw_label = row.get("label") or ""
                source = row.get("src") or ""
                label_counts[raw_label] += 1
                source_counts[source] += 1
                character_lengths.append(len(text))
                whitespace_tokens.append(len(text.split()))

                if raw_label in raw_to_semantic:
                    semantic = raw_to_semantic[raw_label]
                    semantic_counts[semantic] += 1
                    canonical_counts[str(canonical_mapping[raw_label])] += 1
                else:
                    invalid_labels[raw_label] += 1

                text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                text_occurrences[text_digest].append((entry["name"], raw_label, source))

        for field in REQUIRED_COLUMNS:
            if null_values[field]:
                violations.append(f"{entry['name']}: {null_values[field]} null {field} values")
            if blank_values[field]:
                violations.append(f"{entry['name']}: {blank_values[field]} blank {field} values")
        if malformed_rows:
            violations.append(f"{entry['name']}: {malformed_rows} malformed rows")
        if invalid_labels:
            violations.append(f"{entry['name']}: invalid labels {_sorted_counts(invalid_labels)}")

        file_profiles.append(
            {
                "name": entry["name"],
                "relative_path": entry["relative_path"],
                "rows": row_count,
                "columns": list(REQUIRED_COLUMNS),
                "record_identifier_column": None,
                "raw_label_counts": _sorted_counts(label_counts),
                "semantic_label_counts": _sorted_counts(semantic_counts),
                "canonical_target_counts": _sorted_counts(canonical_counts),
                "distinct_sources": len(source_counts),
                "source_counts": _sorted_counts(source_counts),
                "null_values": {field: null_values[field] for field in REQUIRED_COLUMNS},
                "blank_values": {field: blank_values[field] for field in REQUIRED_COLUMNS},
                "malformed_rows": malformed_rows,
                "invalid_label_counts": _sorted_counts(invalid_labels),
                "lengths": {
                    "characters": _length_summary(character_lengths),
                    "whitespace_tokens": _length_summary(whitespace_tokens),
                },
            }
        )

        aggregate_labels.update(label_counts)
        aggregate_semantics.update(semantic_counts)
        aggregate_canonical.update(canonical_counts)
        aggregate_sources.update(source_counts)
        aggregate_characters.extend(character_lengths)
        aggregate_tokens.extend(whitespace_tokens)

    duplicate_groups = [values for values in text_occurrences.values() if len(values) > 1]
    rows_in_duplicate_groups = sum(len(values) for values in duplicate_groups)
    cross_file_groups = sum(
        len({file_name for file_name, _, _ in values}) > 1 for values in duplicate_groups
    )
    conflicting_label_groups = sum(
        len({label for _, label, _ in values}) > 1 for values in duplicate_groups
    )
    duplicate_group_labels: Counter[str] = Counter()
    cross_file_set_counts: Counter[str] = Counter()
    for values in duplicate_groups:
        labels = {label for _, label, _ in values}
        duplicate_group_labels[next(iter(labels)) if len(labels) == 1 else "<conflict>"] += 1
        files = sorted({file_name for file_name, _, _ in values})
        if len(files) > 1:
            cross_file_set_counts["|".join(files)] += 1

    warnings = [
        "No stable record identifier column is present; downstream IDs must be "
        "derived deterministically"
    ]
    if duplicate_groups:
        warnings.append(
            f"{len(duplicate_groups)} exact-text groups contain repeated rows; "
            f"{cross_file_groups} groups span files"
        )
    if conflicting_label_groups:
        violations.append(f"{conflicting_label_groups} exact-text groups have conflicting labels")

    semantics = manifest["label_semantics"]
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profiler_version": PROFILER_VERSION,
        "dataset": manifest.get("dataset", manifest["dataset_id"]),
        "dataset_id": manifest["dataset_id"],
        "revision": manifest["revision"],
        "label_semantics": {
            "raw_to_semantic": raw_to_semantic,
            "canonical_machine_positive_mapping": canonical_mapping,
            "source_revision": semantics.get("source_revision"),
            "source_url": semantics.get("source_url"),
        },
        "files": file_profiles,
        "aggregate": {
            "rows": sum(profile["rows"] for profile in file_profiles),
            "raw_label_counts": _sorted_counts(aggregate_labels),
            "semantic_label_counts": _sorted_counts(aggregate_semantics),
            "canonical_target_counts": _sorted_counts(aggregate_canonical),
            "distinct_source_values": len(aggregate_sources),
            "source_counts": _sorted_counts(aggregate_sources),
            "lengths": {
                "characters": _length_summary(aggregate_characters),
                "whitespace_tokens": _length_summary(aggregate_tokens),
            },
            "exact_text_duplication": {
                "unique_texts": len(text_occurrences),
                "duplicate_groups": len(duplicate_groups),
                "rows_in_duplicate_groups": rows_in_duplicate_groups,
                "duplicate_extra_rows": rows_in_duplicate_groups - len(duplicate_groups),
                "cross_file_groups": cross_file_groups,
                "conflicting_label_groups": conflicting_label_groups,
                "group_raw_label_counts": _sorted_counts(duplicate_group_labels),
                "cross_file_set_counts": _sorted_counts(cross_file_set_counts),
            },
        },
        "validation": {
            "status": "pass" if not violations else "fail",
            "violations": violations,
            "warnings": warnings,
        },
    }


def render_profile(profile: dict[str, Any]) -> str:
    return json.dumps(profile, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_ood_manifest.json",
    )
    parser.add_argument("--input-dir", type=Path, default=repo_root / "data" / "raw")
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data" / "metadata" / "mage_ood_profile.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the existing report differs from a fresh profile",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        profile = build_profile(load_manifest(args.manifest), args.input_dir)
        rendered = render_profile(profile)
        if args.check:
            if not args.output.is_file():
                raise ProfileError(f"Profile does not exist: {args.output}")
            if args.output.read_text(encoding="utf-8") != rendered:
                raise ProfileError(f"Profile is stale: {args.output}")
            print(f"profile verified: {args.output}")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
            temporary.write_text(rendered, encoding="utf-8", newline="\n")
            temporary.replace(args.output)
            print(f"profile written: {args.output}")
        if profile["validation"]["status"] != "pass":
            print("error: profile contains validation violations", file=sys.stderr)
            return 1
    except (
        IntegrityError,
        ManifestError,
        ProfileError,
        OSError,
        UnicodeDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
