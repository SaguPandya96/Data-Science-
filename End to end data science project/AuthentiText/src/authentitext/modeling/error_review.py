"""Deterministic, text-free qualitative review of external policy errors."""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from authentitext.data.cleaning import open_deterministic_gzip, sha256_file
from authentitext.data.regimes import parse_source_id

REVIEW_VERSION = 1
REPORT_SCHEMA_VERSION = 1
ROLE_ORDER = ("human_false_machine", "machine_false_human", "uncertain_boundary")
ALLOWED_CUES = {
    "academic_essay_conventions",
    "citation_or_reference_markers",
    "enumerative_or_heading_structure",
    "first_person_or_personal_voice",
    "formal_or_formulaic_register",
    "informal_or_conversational_register",
    "institutional_or_newswire_style",
    "lexical_variation_or_idiosyncrasy",
    "mixed_surface_signals",
    "narrative_or_dialogue_style",
    "no_clear_surface_cue",
    "short_or_fragmentary",
    "source_like_specific_detail",
}
FIRST_PERSON_PATTERN = re.compile(r"\b(?:i|me|my|mine|myself|we|us|our|ours)\b", re.I)
CITATION_PATTERN = re.compile(r"(?:\[[0-9]+\]|\([A-Z][A-Za-z-]+,?\s+\d{4}\))")
BULLET_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


class ErrorReviewError(RuntimeError):
    """Raised when an external error review cannot be reproduced safely."""


def _load_predictions(path: Path, identity: dict[str, Any]) -> list[dict[str, Any]]:
    if path.stat().st_size != identity["bytes"] or sha256_file(path) != identity["sha256"]:
        raise ErrorReviewError("External prediction identity does not match evaluation report")
    records = []
    record_ids = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ErrorReviewError(f"{path}:{line_number} is invalid JSON") from error
            required = {
                "calibrated_score",
                "category",
                "raw_score",
                "record_id",
                "source",
                "target",
                "whitespace_tokens",
            }
            if not isinstance(record, dict) or not required <= record.keys():
                raise ErrorReviewError(f"{path}:{line_number} is missing prediction fields")
            if (
                not isinstance(record["record_id"], str)
                or record["record_id"] in record_ids
                or record["target"] not in (0, 1)
                or record["category"] not in {"likely_human", "uncertain", "likely_machine"}
                or not isinstance(record["calibrated_score"], int | float)
            ):
                raise ErrorReviewError(f"{path}:{line_number} has invalid prediction fields")
            record_ids.add(record["record_id"])
            identity_metadata = parse_source_id(record["source"])
            record["domain"] = identity_metadata.domain
            record["generator"] = identity_metadata.generator
            record["strategy"] = identity_metadata.strategy
            records.append(record)
    if len(records) != identity["rows"]:
        raise ErrorReviewError("External prediction row count does not match report")
    return records


def _grouped_selection(
    records: list[dict[str, Any]],
    *,
    group_key,
    rows_per_group: int,
    score_descending: bool,
) -> list[dict[str, Any]]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(group_key(record))].append(record)
    selected = []
    for group in sorted(groups):
        ordered = sorted(
            groups[group],
            key=lambda record: (
                -record["calibrated_score"] if score_descending else record["calibrated_score"],
                record["record_id"],
            ),
        )
        if len(ordered) < rows_per_group:
            raise ErrorReviewError(
                f"Review group {group!r} has {len(ordered)} rows; expected {rows_per_group}"
            )
        selected.extend(ordered[:rows_per_group])
    return selected


def _select_records(
    predictions: list[dict[str, Any]],
    *,
    costly_rows_per_group: int,
    uncertain_rows_per_group: int,
) -> list[dict[str, Any]]:
    human_false_machine = [
        record
        for record in predictions
        if record["target"] == 0 and record["category"] == "likely_machine"
    ]
    machine_false_human = [
        record
        for record in predictions
        if record["target"] == 1 and record["category"] == "likely_human"
    ]
    uncertain = [record for record in predictions if record["category"] == "uncertain"]
    selections = []
    for role, records in (
        (
            "human_false_machine",
            _grouped_selection(
                human_false_machine,
                group_key=lambda record: record["domain"],
                rows_per_group=costly_rows_per_group,
                score_descending=True,
            ),
        ),
        (
            "machine_false_human",
            _grouped_selection(
                machine_false_human,
                group_key=lambda record: record["generator"],
                rows_per_group=costly_rows_per_group,
                score_descending=False,
            ),
        ),
    ):
        for record in records:
            selections.append({**record, "sample_role": role})

    uncertain_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in uncertain:
        uncertain_groups[f"{record['target']}|{record['domain']}"].append(record)
    for group in sorted(uncertain_groups):
        target = int(group.split("|", maxsplit=1)[0])
        ordered = sorted(
            uncertain_groups[group],
            key=lambda record: (
                -record["calibrated_score"] if target == 0 else record["calibrated_score"],
                record["record_id"],
            ),
        )
        if len(ordered) < uncertain_rows_per_group:
            raise ErrorReviewError(f"Uncertain review group {group!r} is too small")
        for record in ordered[:uncertain_rows_per_group]:
            selections.append({**record, "sample_role": "uncertain_boundary"})
    return sorted(
        selections,
        key=lambda record: (
            ROLE_ORDER.index(record["sample_role"]),
            record["domain"],
            record["generator"] or "human",
            record["target"],
            record["record_id"],
        ),
    )


def _excerpt(text: str, first_characters: int, last_characters: int) -> dict[str, str]:
    collapsed = " ".join(text.split())
    return {
        "opening_excerpt": collapsed[:first_characters],
        "closing_excerpt": (
            collapsed[-last_characters:] if len(collapsed) > first_characters else ""
        ),
    }


def _surface_profile(text: str) -> dict[str, int]:
    lines = text.splitlines() or [text]
    nonblank_lines = [line for line in lines if line.strip()]
    return {
        "characters": len(text),
        "whitespace_tokens": len(text.split()),
        "nonblank_lines": len(nonblank_lines),
        "paragraphs": len([part for part in re.split(r"\n\s*\n", text) if part.strip()]),
        "bullet_or_numbered_lines": sum(bool(BULLET_PATTERN.match(line)) for line in lines),
        "heading_like_lines": sum(
            bool(line.strip())
            and len(line.split()) <= 10
            and (line.strip().endswith(":") or line.strip().isupper())
            for line in lines
        ),
        "first_person_markers": len(FIRST_PERSON_PATTERN.findall(text)),
        "citation_markers": len(CITATION_PATTERN.findall(text)),
        "digit_characters": sum(character.isdigit() for character in text),
    }


def _write_packet(
    path: Path,
    selected: list[dict[str, Any]],
    texts: dict[str, str],
    *,
    first_excerpt_characters: int,
    last_excerpt_characters: int,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with open_deterministic_gzip(temporary) as handle:
            for record in selected:
                text = texts[record["record_id"]]
                handle.write(
                    json.dumps(
                        {
                            "calibrated_score": record["calibrated_score"],
                            "category": record["category"],
                            "domain": record["domain"],
                            "generator": record["generator"],
                            "record_id": record["record_id"],
                            "sample_role": record["sample_role"],
                            "strategy": record["strategy"],
                            "surface_profile": _surface_profile(text),
                            "target": record["target"],
                            **_excerpt(
                                text,
                                first_excerpt_characters,
                                last_excerpt_characters,
                            ),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "relative_path": path.name,
        "rows": len(selected),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_selected_texts(external_path: Path, selected_ids: set[str]) -> dict[str, str]:
    texts = {}
    with gzip.open(external_path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ErrorReviewError(f"{external_path}:{line_number} is invalid JSON") from error
            if not isinstance(record, dict):
                raise ErrorReviewError(f"{external_path}:{line_number} is not an object")
            record_id = record.get("record_id")
            if record_id in selected_ids:
                if not isinstance(record.get("text"), str):
                    raise ErrorReviewError(f"{external_path}:{line_number} has invalid text")
                texts[record_id] = record["text"]
    if texts.keys() != selected_ids:
        raise ErrorReviewError("Not every selected error-review ID exists in external data")
    return texts


def _validate_annotations(
    annotations: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_by_id = {record["record_id"]: record for record in selected}
    if not isinstance(annotations, list) or len(annotations) != len(selected):
        raise ErrorReviewError("Every selected review record must have one annotation")
    annotations_by_id = {}
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ErrorReviewError("Review annotation must be an object")
        record_id = annotation.get("record_id")
        cues = annotation.get("observed_cues")
        note = annotation.get("reviewer_note")
        if (
            not isinstance(record_id, str)
            or record_id in annotations_by_id
            or not isinstance(cues, list)
            or not cues
            or any(cue not in ALLOWED_CUES for cue in cues)
            or cues != sorted(set(cues))
            or not isinstance(note, str)
            or not note.strip()
            or len(note) > 160
            or '"' in note
        ):
            raise ErrorReviewError(f"Invalid text-free annotation for record {record_id!r}")
        annotations_by_id[record_id] = {"observed_cues": cues, "reviewer_note": note}
    if annotations_by_id.keys() != selected_by_id.keys():
        raise ErrorReviewError("Annotation IDs do not match the selected review sample")
    return [
        {
            "record_id": record["record_id"],
            "sample_role": record["sample_role"],
            "observed_cues": annotations_by_id[record["record_id"]]["observed_cues"],
            "reviewer_note": annotations_by_id[record["record_id"]]["reviewer_note"],
        }
        for record in selected
    ]


def build_error_review(
    *,
    evaluation_report: dict[str, Any],
    evaluation_report_identity: dict[str, Any],
    external_path: Path,
    prediction_path: Path,
    packet_path: Path,
    annotations: list[dict[str, Any]] | None = None,
    costly_rows_per_group: int = 3,
    uncertain_rows_per_group: int = 1,
    first_excerpt_characters: int = 240,
    last_excerpt_characters: int = 120,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select real errors, write a local packet, and optionally finalize annotations."""
    if evaluation_report.get("validation", {}).get("status") != "pass":
        raise ErrorReviewError("External evaluation status is not pass")
    if evaluation_report["validation"].get("external_outcomes_used_for_retuning") is not False:
        raise ErrorReviewError("External outcomes must remain evaluation-only")
    external_identity = evaluation_report["input"]
    if (
        external_path.stat().st_size != external_identity["bytes"]
        or sha256_file(external_path) != external_identity["sha256"]
    ):
        raise ErrorReviewError("External input identity does not match evaluation report")
    predictions = _load_predictions(prediction_path, evaluation_report["predictions"])
    selected = _select_records(
        predictions,
        costly_rows_per_group=costly_rows_per_group,
        uncertain_rows_per_group=uncertain_rows_per_group,
    )
    selected_ids = {record["record_id"] for record in selected}
    if len(selected_ids) != len(selected):
        raise ErrorReviewError("Error-review selection contains duplicate records")
    texts = _load_selected_texts(external_path, selected_ids)
    packet_identity = _write_packet(
        packet_path,
        selected,
        texts,
        first_excerpt_characters=first_excerpt_characters,
        last_excerpt_characters=last_excerpt_characters,
    )
    role_counts = Counter(record["sample_role"] for record in selected)
    domain_counts = Counter(record["domain"] for record in selected)
    generator_counts = Counter(record["generator"] or "human" for record in selected)
    template = [
        {"record_id": record["record_id"], "observed_cues": [], "reviewer_note": ""}
        for record in selected
    ]
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "review_version": REVIEW_VERSION,
        "dataset_id": evaluation_report["dataset_id"],
        "revision": evaluation_report["revision"],
        "inputs": {
            "evaluation_report": evaluation_report_identity,
            "external_data": external_identity,
            "predictions": evaluation_report["predictions"],
        },
        "configuration": {
            "selection": {
                "human_false_machine": ("highest calibrated scores within each external domain"),
                "machine_false_human": ("lowest calibrated scores within each external generator"),
                "uncertain_boundary": (
                    "human scores nearest machine boundary and machine scores nearest human "
                    "boundary within each target/domain"
                ),
                "costly_rows_per_group": costly_rows_per_group,
                "uncertain_rows_per_group": uncertain_rows_per_group,
                "tie_break": "lowest record_id",
            },
            "local_packet_opening_excerpt_characters": first_excerpt_characters,
            "local_packet_closing_excerpt_characters": last_excerpt_characters,
            "allowed_cues": sorted(ALLOWED_CUES),
            "annotation_note_policy": "generic observation, 160 characters maximum, no quotes",
            "model_or_policy_changes_allowed": False,
        },
        "sample": {
            "rows": len(selected),
            "role_counts": {key: role_counts[key] for key in ROLE_ORDER},
            "domain_counts": {key: domain_counts[key] for key in sorted(domain_counts)},
            "generator_counts": {key: generator_counts[key] for key in sorted(generator_counts)},
            "records": [
                {
                    "calibrated_score": record["calibrated_score"],
                    "category": record["category"],
                    "domain": record["domain"],
                    "generator": record["generator"],
                    "record_id": record["record_id"],
                    "sample_role": record["sample_role"],
                    "strategy": record["strategy"],
                    "target": record["target"],
                    "whitespace_tokens": record["whitespace_tokens"],
                }
                for record in selected
            ],
        },
        "local_review_packet": packet_identity,
        "validation": {
            "status": "awaiting_review" if annotations is None else "pass",
            "sample_is_deterministic": True,
            "source_text_in_committed_report": False,
            "external_outcomes_used_for_retuning": False,
            "all_selected_records_annotated": annotations is not None,
        },
    }
    if annotations is not None:
        validated = _validate_annotations(annotations, selected)
        cue_counts = Counter(cue for annotation in validated for cue in annotation["observed_cues"])
        role_cue_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        for annotation in validated:
            role_cue_counts[annotation["sample_role"]].update(annotation["observed_cues"])
        report["annotations"] = {
            "records": validated,
            "cue_counts": {key: cue_counts[key] for key in sorted(cue_counts)},
            "role_cue_counts": {
                role: {key: role_cue_counts[role][key] for key in sorted(role_cue_counts[role])}
                for role in ROLE_ORDER
            },
        }
    return report, template


def render_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
