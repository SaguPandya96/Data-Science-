"""Aggregate drift checks against a development-only reference."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from authentitext.monitoring import (
    CHARACTER_BUCKETS,
    KNOWN_CATEGORIES,
    SCORE_BUCKETS,
    TOKEN_BUCKETS,
    bucket_index,
    render_histogram,
)

DRIFT_VERSION = 1
SIGNALS = (
    "character_histogram",
    "token_histogram",
    "calibrated_score_histogram",
    "categories",
)


class DriftError(RuntimeError):
    """Raised when a drift reference or snapshot violates the contract."""


def _counts_from_histogram(value: object) -> tuple[list[float | int | None], list[int]]:
    if not isinstance(value, list) or not value:
        raise DriftError("Histogram must be a non-empty list")
    bounds = []
    counts = []
    for row in value:
        if (
            not isinstance(row, Mapping)
            or row.get("upper_bound") is not None
            and not isinstance(row.get("upper_bound"), int | float)
            or not isinstance(row.get("count"), int)
            or row["count"] < 0
        ):
            raise DriftError("Histogram row is invalid")
        bounds.append(row.get("upper_bound"))
        counts.append(row["count"])
    return bounds, counts


def total_variation(reference_counts: Iterable[int], observed_counts: Iterable[int]) -> float:
    """Return total-variation distance between two count distributions."""
    reference = list(reference_counts)
    observed = list(observed_counts)
    if len(reference) != len(observed) or not reference:
        raise DriftError("Distributions must have the same non-zero width")
    if any(not isinstance(count, int) or count < 0 for count in (*reference, *observed)):
        raise DriftError("Distribution counts must be non-negative integers")
    reference_total = sum(reference)
    observed_total = sum(observed)
    if not reference_total or not observed_total:
        raise DriftError("Distributions must have positive totals")
    distance = (
        sum(
            abs(reference_count / reference_total - observed_count / observed_total)
            for reference_count, observed_count in zip(reference, observed, strict=True)
        )
        / 2
    )
    return float(distance)


def aggregate_distribution(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate drift signals using the exact operational bucket definitions."""
    characters: Counter[int] = Counter()
    tokens: Counter[int] = Counter()
    scores: Counter[int] = Counter()
    categories: Counter[str] = Counter()
    rows = 0
    for record in records:
        try:
            character_count = int(record["characters"])
            token_count = int(record["whitespace_tokens"])
            score = float(record["calibrated_machine_likelihood"])
            category = record["category"]
        except (KeyError, TypeError, ValueError) as error:
            raise DriftError("Drift record has invalid fields") from error
        if (
            character_count < 0
            or token_count < 0
            or not 0 <= score <= 1
            or category not in KNOWN_CATEGORIES
        ):
            raise DriftError("Drift record has out-of-range fields")
        rows += 1
        characters[bucket_index(character_count, CHARACTER_BUCKETS)] += 1
        tokens[bucket_index(token_count, TOKEN_BUCKETS)] += 1
        scores[bucket_index(score, SCORE_BUCKETS)] += 1
        categories[category] += 1
    if not rows:
        raise DriftError("Drift distribution cannot be empty")
    return {
        "rows": rows,
        "character_histogram": render_histogram(CHARACTER_BUCKETS, characters),
        "token_histogram": render_histogram(TOKEN_BUCKETS, tokens),
        "calibrated_score_histogram": render_histogram(SCORE_BUCKETS, scores),
        "categories": {category: categories[category] for category in KNOWN_CATEGORIES},
    }


def signal_distances(observed: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, float]:
    """Compare the four fixed aggregate signals with total-variation distance."""
    distances = {}
    for signal in SIGNALS:
        if signal == "categories":
            observed_value = observed.get(signal)
            reference_value = reference.get(signal)
            if not isinstance(observed_value, Mapping) or not isinstance(reference_value, Mapping):
                raise DriftError("Category distribution is invalid")
            observed_counts = [observed_value.get(category, 0) for category in KNOWN_CATEGORIES]
            reference_counts = [reference_value.get(category, 0) for category in KNOWN_CATEGORIES]
        else:
            observed_bounds, observed_counts = _counts_from_histogram(observed.get(signal))
            reference_bounds, reference_counts = _counts_from_histogram(reference.get(signal))
            if observed_bounds != reference_bounds:
                raise DriftError(f"{signal} boundaries do not match the reference")
        distances[signal] = total_variation(reference_counts, observed_counts)
    return distances


def load_drift_reference(path: Path) -> dict[str, Any]:
    """Load and structurally validate a development drift reference."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DriftError(f"Cannot load drift reference: {error}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("drift_version") != DRIFT_VERSION
        or not isinstance(payload.get("identity"), Mapping)
        or not isinstance(payload.get("reference_distribution"), Mapping)
        or not isinstance(payload.get("thresholds"), Mapping)
    ):
        raise DriftError("Drift reference has an invalid schema")
    identity = payload["identity"]
    model = identity.get("model")
    validation = identity.get("validation")
    minimum = payload["thresholds"].get("minimum_observations")
    signal_thresholds = payload["thresholds"].get("total_variation")
    if (
        not isinstance(model, Mapping)
        or not isinstance(model.get("base_model_sha256"), str)
        or not isinstance(model.get("calibration_sha256"), str)
        or not isinstance(validation, Mapping)
        or not isinstance(validation.get("rows"), int)
        or not isinstance(payload["reference_distribution"].get("rows"), int)
        or not 0 < payload["reference_distribution"]["rows"] <= validation["rows"]
        or not isinstance(minimum, int)
        or minimum < 1
        or not isinstance(signal_thresholds, Mapping)
        or any(
            not isinstance(signal_thresholds.get(signal), int | float)
            or not 0 <= signal_thresholds[signal] <= 1
            for signal in SIGNALS
        )
    ):
        raise DriftError("Drift thresholds are invalid")
    signal_distances(payload["reference_distribution"], payload["reference_distribution"])
    return payload


def evaluate_drift(snapshot: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate a monitoring snapshot; flags require human investigation only."""
    predictions = snapshot.get("predictions")
    thresholds = reference.get("thresholds")
    if not isinstance(predictions, Mapping) or not isinstance(thresholds, Mapping):
        raise DriftError("Monitoring snapshot or reference is invalid")
    observations = predictions.get("successful_items")
    minimum = thresholds.get("minimum_observations")
    if not isinstance(observations, int) or observations < 0 or not isinstance(minimum, int):
        raise DriftError("Observation counts are invalid")
    base = {
        "schema_version": 1,
        "drift_version": DRIFT_VERSION,
        "observations": observations,
        "minimum_observations": minimum,
        "reference": reference.get("identity", {}),
        "automatic_action": False,
    }
    if observations < minimum:
        return {**base, "status": "insufficient_data", "investigation_required": False}

    distances = signal_distances(predictions, reference["reference_distribution"])
    configured = thresholds.get("total_variation")
    if not isinstance(configured, Mapping):
        raise DriftError("Drift signal thresholds are invalid")
    signals = {
        signal: {
            "total_variation": round(distance, 6),
            "threshold": round(float(configured[signal]), 6),
            "flagged": distance > float(configured[signal]),
        }
        for signal, distance in distances.items()
    }
    investigation_required = any(signal["flagged"] for signal in signals.values())
    return {
        **base,
        "status": "investigate" if investigation_required else "within_reference",
        "investigation_required": investigation_required,
        "signals": signals,
    }
