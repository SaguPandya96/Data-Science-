"""Check committed experiment metadata without requiring ignored artifacts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from authentitext.data.cleaning import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_DIR = REPO_ROOT / "data" / "metadata"

REQUIRED_FILES = {
    "clean_room_reproduction_report.json",
    "experiment_registry.json",
    "ghostbuster_main_manifest.json",
    "ghostbuster_evaluation_report.json",
    "ghostbuster_error_review_report.json",
    "ghostbuster_overlap_report.json",
    "mage_baseline_training_report.json",
    "mage_baseline_validation_report.json",
    "mage_calibration_report.json",
    "mage_cleaning_report.json",
    "mage_development_manifest.json",
    "mage_development_profile.json",
    "mage_domain_holdout_report.json",
    "mage_drift_reference.json",
    "mage_eda_report.json",
    "mage_frozen_test_report.json",
    "mage_generator_holdout_report.json",
    "mage_id_split_report.json",
    "mage_leakage_report.json",
    "mage_near_duplicate_report.json",
    "mage_ood_evaluation_report.json",
    "mage_ood_manifest.json",
    "mage_ood_profile.json",
    "mage_source_regimes.json",
    "mage_truncation_robustness_report.json",
}
MANIFEST_FILES = {"mage_development_manifest.json", "mage_ood_manifest.json"}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_report(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if not raw.endswith("\n"):
        raise ValueError("file must end with a newline")
    payload = json.loads(
        raw,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON value must be an object")
    return payload


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _find_non_finite(value: Any, location: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{location} is not finite")
    elif isinstance(value, dict):
        for key, child in value.items():
            errors.extend(_find_non_finite(child, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_find_non_finite(child, f"{location}[{index}]"))
    return errors


def check_metadata(metadata_dir: Path = DEFAULT_METADATA_DIR) -> list[str]:
    """Return validation errors for the committed metadata directory."""
    errors: list[str] = []
    paths = {path.name: path for path in metadata_dir.glob("*.json")}
    missing = sorted(REQUIRED_FILES - paths.keys())
    if missing:
        errors.append(f"missing required files: {', '.join(missing)}")

    reports: dict[str, dict[str, Any]] = {}
    for name, path in sorted(paths.items()):
        try:
            payload = _load_report(path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{name}: {exc}")
            continue
        reports[name] = payload
        if payload.get("schema_version") != 1:
            errors.append(f"{name}: schema_version must equal 1")
        errors.extend(f"{name}: {message}" for message in _find_non_finite(payload))
        if name not in MANIFEST_FILES and _nested(payload, "validation", "status") != "pass":
            errors.append(f"{name}: validation.status must equal 'pass'")

    calibration = reports.get("mage_calibration_report.json")
    frozen = reports.get("mage_frozen_test_report.json")
    ood = reports.get("mage_ood_evaluation_report.json")
    drift = reports.get("mage_drift_reference.json")
    if all(report is not None for report in (calibration, frozen, ood, drift)):
        assert calibration is not None
        assert frozen is not None
        assert ood is not None
        assert drift is not None
        expected_model = _nested(calibration, "base_model", "sha256")
        expected_calibration = _nested(calibration, "artifact", "sha256")
        linked_hashes = {
            "frozen test base model": _nested(frozen, "base_model", "sha256"),
            "frozen test calibrator": _nested(frozen, "calibration_artifact", "sha256"),
            "OOD base model": _nested(ood, "base_model", "sha256"),
            "OOD calibrator": _nested(ood, "calibration_artifact", "sha256"),
            "drift base model": _nested(drift, "identity", "model", "base_model_sha256"),
            "drift calibrator": _nested(drift, "identity", "model", "calibration_sha256"),
        }
        for label, actual in linked_hashes.items():
            expected = expected_calibration if "calibrator" in label else expected_model
            if not isinstance(expected, str) or actual != expected:
                errors.append(f"{label}: hash does not match the calibration report")

        if _nested(ood, "validation", "published_test_data_used") is not False:
            errors.append("OOD report must state that published test data was not used")
        if _nested(drift, "validation", "published_test_data_used") is not False:
            errors.append("drift reference must state that published test data was not used")

    source_regimes = reports.get("mage_source_regimes.json")
    domain_holdouts = reports.get("mage_domain_holdout_report.json")
    if domain_holdouts is not None:
        folds = domain_holdouts.get("folds")
        if not isinstance(folds, list) or len(folds) != 9:
            errors.append("domain holdout report must contain nine folds")
        elif len({fold.get("held_out_domain") for fold in folds}) != 9:
            errors.append("domain holdout report must contain nine unique domains")
        elif source_regimes is not None:
            declared = {fold.get("held_out_domain") for fold in source_regimes["domain_holdouts"]}
            observed = {fold.get("held_out_domain") for fold in folds}
            if observed != declared:
                errors.append("domain holdout folds must match the declared regimes")
        if isinstance(folds, list) and any(
            fold.get("validation", {}).get("status") != "pass" for fold in folds
        ):
            errors.append("every domain holdout fold must have validation status pass")
        if (
            _nested(
                domain_holdouts,
                "configuration",
                "test_data_used_for_training_calibration_or_thresholds",
            )
            is not False
        ):
            errors.append("domain holdout report must prohibit test use before evaluation")
        if _nested(domain_holdouts, "validation", "folds_completed") != 9:
            errors.append("domain holdout report must mark all nine folds complete")

    generator_holdouts = reports.get("mage_generator_holdout_report.json")
    if generator_holdouts is not None:
        folds = generator_holdouts.get("folds")
        if not isinstance(folds, list) or len(folds) != 27:
            errors.append("generator holdout report must contain 27 folds")
        elif len({fold.get("held_out_generator") for fold in folds}) != 27:
            errors.append("generator holdout report must contain 27 unique generators")
        elif source_regimes is not None:
            declared = {
                fold.get("held_out_generator") for fold in source_regimes["generator_holdouts"]
            }
            observed = {fold.get("held_out_generator") for fold in folds}
            if observed != declared:
                errors.append("generator holdout folds must match the declared regimes")
        if isinstance(folds, list) and any(
            fold.get("validation", {}).get("status") != "pass" for fold in folds
        ):
            errors.append("every generator holdout fold must have validation status pass")
        if (
            _nested(
                generator_holdouts,
                "configuration",
                "test_data_used_for_training_calibration_or_thresholds",
            )
            is not False
        ):
            errors.append("generator holdout report must prohibit test use before evaluation")
        if _nested(generator_holdouts, "validation", "folds_completed") != 27:
            errors.append("generator holdout report must mark all 27 folds complete")

    truncation = reports.get("mage_truncation_robustness_report.json")
    calibration = reports.get("mage_calibration_report.json")
    id_split = reports.get("mage_id_split_report.json")
    if truncation is not None:
        if calibration is not None:
            if _nested(truncation, "base_model", "sha256") != _nested(
                calibration, "base_model", "sha256"
            ):
                errors.append("truncation robustness base model is not frozen")
            if _nested(truncation, "calibration_artifact", "sha256") != _nested(
                calibration, "artifact", "sha256"
            ):
                errors.append("truncation robustness calibrator is not frozen")
        if id_split is not None:
            test_input = next(
                (item for item in id_split.get("outputs", []) if item.get("partition") == "test"),
                None,
            )
            if not isinstance(test_input, dict) or _nested(
                truncation, "input", "sha256"
            ) != test_input.get("sha256"):
                errors.append("truncation robustness input does not match the sanitized test")
        budgets = _nested(truncation, "configuration", "budgets_whitespace_tokens")
        conditions = truncation.get("conditions")
        if budgets != [50, 100, 200]:
            errors.append("truncation robustness must retain the prespecified budgets")
        if (
            not isinstance(conditions, list)
            or [item.get("budget_whitespace_tokens") for item in conditions] != budgets
        ):
            errors.append("truncation robustness conditions do not match the budgets")
        else:
            condition_rows = [
                _nested(item, "selection", "rows") for item in conditions if isinstance(item, dict)
            ]
            if not all(isinstance(value, int) and value > 0 for value in condition_rows):
                errors.append("truncation robustness condition rows must be positive integers")
            elif sum(condition_rows) != _nested(truncation, "predictions", "rows"):
                errors.append("truncation robustness paired rows do not reconcile")
        for key in (
            "prediction_file_verified",
            "metrics_recomputed_from_predictions",
            "all_selected_records_strictly_exceed_budget",
            "all_transformed_records_match_budget",
        ):
            if _nested(truncation, "validation", key) is not True:
                errors.append(f"truncation robustness validation.{key} must be true")
        for key in (
            "source_text_in_report_or_predictions",
            "model_calibration_or_threshold_retuning_performed",
        ):
            if _nested(truncation, "validation", key) is not False:
                errors.append(f"truncation robustness validation.{key} must be false")
        if (
            _nested(truncation, "configuration", "test_outcomes_used_for_budget_selection")
            is not False
        ):
            errors.append("truncation budgets must not be selected from test outcomes")
        if (
            _nested(
                truncation,
                "configuration",
                "retuning_after_robustness_evaluation_allowed",
            )
            is not False
        ):
            errors.append("truncation robustness outcomes must not permit retuning")
        serialized = json.dumps(truncation, sort_keys=True)
        if '"text":' in serialized:
            errors.append("truncation robustness report contains source text")

    ghostbuster = reports.get("ghostbuster_main_manifest.json")
    if ghostbuster is not None:
        selected = _nested(ghostbuster, "selection", "paper_main_corpus_rows")
        excluded = _nested(ghostbuster, "validation", "blank_documents_excluded")
        output_rows = _nested(ghostbuster, "output", "rows")
        target_counts = _nested(ghostbuster, "profile", "target_counts")
        if not all(isinstance(value, int) for value in (selected, excluded, output_rows)):
            errors.append("Ghostbuster manifest row counts must be integers")
        elif selected - excluded != output_rows:
            errors.append("Ghostbuster selected, blank, and output rows do not reconcile")
        if (
            not isinstance(target_counts, dict)
            or not all(isinstance(value, int) for value in target_counts.values())
            or sum(target_counts.values()) != output_rows
        ):
            errors.append("Ghostbuster target counts do not reconcile to output rows")
        if _nested(ghostbuster, "validation", "raw_text_in_report") is not False:
            errors.append("Ghostbuster preparation report must not contain raw text")
        if _nested(ghostbuster, "validation", "test_scoring_performed") is not False:
            errors.append("Ghostbuster preparation must not score the external test")

    external_overlap = reports.get("ghostbuster_overlap_report.json")
    id_split = reports.get("mage_id_split_report.json")
    if external_overlap is not None:
        exclusions = _nested(external_overlap, "exclusions", "records")
        excluded_rows = _nested(external_overlap, "exclusions", "external_records")
        remaining_rows = _nested(external_overlap, "exclusions", "external_rows_after_exclusion")
        external_rows = _nested(external_overlap, "datasets", "external", "rows")
        if not isinstance(exclusions, list) or len(exclusions) != excluded_rows:
            errors.append("Ghostbuster overlap exclusions do not reconcile")
        if not all(
            isinstance(value, int) for value in (excluded_rows, remaining_rows, external_rows)
        ):
            errors.append("Ghostbuster overlap row counts must be integers")
        elif external_rows - excluded_rows != remaining_rows:
            errors.append("Ghostbuster overlap remaining rows do not reconcile")
        if ghostbuster is not None and (
            _nested(external_overlap, "datasets", "external", "sha256")
            != _nested(ghostbuster, "output", "sha256")
        ):
            errors.append("Ghostbuster overlap input does not match its preparation manifest")
        if id_split is not None:
            overlap_inputs = _nested(external_overlap, "datasets", "development", "inputs")
            if not isinstance(overlap_inputs, list) or {
                (item.get("partition"), item.get("sha256")) for item in overlap_inputs
            } != {(item.get("partition"), item.get("sha256")) for item in id_split["outputs"]}:
                errors.append("Ghostbuster overlap MAGE inputs do not match the ID split")
        required_false = (
            "raw_text_in_report",
            "model_scoring_performed",
            "outcome_data_used_to_define_exclusions",
        )
        for key in required_false:
            if _nested(external_overlap, "validation", key) is not False:
                errors.append(f"Ghostbuster overlap validation.{key} must be false")
        if (
            _nested(
                external_overlap,
                "validation",
                "all_confirmed_overlap_resolved_by_external_exclusion",
            )
            is not True
        ):
            errors.append("Ghostbuster overlap findings must be resolved before scoring")

    external_evaluation = reports.get("ghostbuster_evaluation_report.json")
    calibration = reports.get("mage_calibration_report.json")
    if external_evaluation is not None:
        if ghostbuster is not None and (
            _nested(external_evaluation, "input", "sha256")
            != _nested(ghostbuster, "output", "sha256")
        ):
            errors.append("Ghostbuster evaluation input does not match its manifest")
        if external_overlap is not None:
            overlap_path = metadata_dir / "ghostbuster_overlap_report.json"
            if _nested(external_evaluation, "overlap_gate", "sha256") != sha256_file(overlap_path):
                errors.append("Ghostbuster evaluation does not match the overlap report")
        if calibration is not None:
            if _nested(external_evaluation, "base_model", "sha256") != _nested(
                calibration, "base_model", "sha256"
            ):
                errors.append("Ghostbuster evaluation base model is not frozen")
            if _nested(external_evaluation, "calibration_artifact", "sha256") != _nested(
                calibration, "artifact", "sha256"
            ):
                errors.append("Ghostbuster evaluation calibrator is not frozen")
        input_rows = _nested(external_evaluation, "selection", "input_rows")
        excluded_rows = _nested(external_evaluation, "selection", "excluded_rows")
        scored_rows = _nested(external_evaluation, "selection", "scored_rows")
        prediction_rows = _nested(external_evaluation, "predictions", "rows")
        if not all(
            isinstance(value, int)
            for value in (input_rows, excluded_rows, scored_rows, prediction_rows)
        ):
            errors.append("Ghostbuster evaluation selection rows must be integers")
        elif input_rows - excluded_rows != scored_rows or scored_rows != prediction_rows:
            errors.append("Ghostbuster evaluation selection rows do not reconcile")
        required_true = (
            "prediction_file_verified",
            "metrics_recomputed_from_predictions",
            "prediction_ids_match_overlap_gated_external_ids",
            "overlap_gate_verified",
            "external_exclusions_applied",
        )
        for key in required_true:
            if _nested(external_evaluation, "validation", key) is not True:
                errors.append(f"Ghostbuster evaluation validation.{key} must be true")
        if (
            _nested(external_evaluation, "validation", "external_outcomes_used_for_retuning")
            is not False
        ):
            errors.append("Ghostbuster outcomes must not be used for retuning")

    error_review = reports.get("ghostbuster_error_review_report.json")
    if error_review is not None:
        if external_evaluation is not None:
            evaluation_path = metadata_dir / "ghostbuster_evaluation_report.json"
            if _nested(error_review, "inputs", "evaluation_report", "sha256") != sha256_file(
                evaluation_path
            ):
                errors.append("Ghostbuster error review does not match the evaluation report")
        sample_rows = _nested(error_review, "sample", "rows")
        sample_records = _nested(error_review, "sample", "records")
        annotations = _nested(error_review, "annotations", "records")
        role_counts = _nested(error_review, "sample", "role_counts")
        if (
            not isinstance(sample_rows, int)
            or not isinstance(sample_records, list)
            or not isinstance(annotations, list)
            or len(sample_records) != sample_rows
            or len(annotations) != sample_rows
        ):
            errors.append("Ghostbuster error-review sample and annotations do not reconcile")
        if not isinstance(role_counts, dict) or sum(role_counts.values()) != sample_rows:
            errors.append("Ghostbuster error-review role counts do not reconcile")
        if _nested(error_review, "validation", "all_selected_records_annotated") is not True:
            errors.append("Ghostbuster error-review records must all be annotated")
        if _nested(error_review, "validation", "source_text_in_committed_report") is not False:
            errors.append("Ghostbuster error-review report must not contain source text")
        if _nested(error_review, "validation", "external_outcomes_used_for_retuning") is not False:
            errors.append("Ghostbuster error review must not retune the frozen policy")
        serialized = json.dumps(error_review, sort_keys=True)
        for forbidden_key in ("opening_excerpt", "closing_excerpt", "text"):
            if f'"{forbidden_key}":' in serialized:
                errors.append(f"Ghostbuster error-review report contains {forbidden_key}")

    clean_room = reports.get("clean_room_reproduction_report.json")
    if clean_room is not None:
        commit = _nested(clean_room, "source", "commit")
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            errors.append("Clean-room reproduction must identify a full source commit")
        for key in ("clone_files_excluding_git", "clone_bytes_excluding_git"):
            value = _nested(clean_room, "source", key)
            if not isinstance(value, int) or value <= 0:
                errors.append(f"Clean-room reproduction source.{key} must be positive")
        if _nested(clean_room, "source", "evidence_worktree_line_endings") != "lf":
            errors.append("Clean-room reproduction must verify LF evidence checkouts")
        if _nested(clean_room, "source", "git_status_clean_after_validation") is not True:
            errors.append("Clean-room reproduction checkout must remain clean")
        if _nested(clean_room, "installation", "pip_check") != "pass":
            errors.append("Clean-room reproduction dependency graph must pass pip check")
        required_gates = {
            "documentation",
            "experiment_registry",
            "format",
            "lint",
            "metadata",
            "model_card",
            "tests",
            "wheel",
        }
        verification = clean_room.get("verification")
        if not isinstance(verification, dict) or not required_gates <= verification.keys():
            errors.append("Clean-room reproduction is missing a required verification gate")
        else:
            for gate in sorted(required_gates):
                if _nested(clean_room, "verification", gate, "status") != "pass":
                    errors.append(f"Clean-room reproduction {gate} gate must pass")
        wheel_sha = _nested(clean_room, "verification", "wheel", "sha256")
        if not isinstance(wheel_sha, str) or re.fullmatch(r"[0-9a-f]{64}", wheel_sha) is None:
            errors.append("Clean-room reproduction wheel must have a SHA-256 identity")
        for key in (
            "external_links_rechecked",
            "frozen_experiment_replay_performed",
            "hosted_ci_run_observed",
            "ignored_model_artifacts_present",
            "ignored_predictions_present",
            "raw_or_processed_datasets_present",
        ):
            if _nested(clean_room, "limitations", key) is not False:
                errors.append(f"Clean-room reproduction limitations.{key} must be false")
        for key in (
            "all_ci_equivalent_checks_passed",
            "checkout_clean_after_validation",
            "dependency_graph_consistent",
            "no_claim_of_experiment_replay",
        ):
            if _nested(clean_room, "validation", key) is not True:
                errors.append(f"Clean-room reproduction validation.{key} must be true")

    return errors


def main() -> int:
    errors = check_metadata()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        f"Checked {len(list(DEFAULT_METADATA_DIR.glob('*.json')))} committed metadata files; "
        "schemas, validation markers, and frozen model links are consistent."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
