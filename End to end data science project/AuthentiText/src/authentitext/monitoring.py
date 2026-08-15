"""Bounded, privacy-safe operational metrics for local inference."""

from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Iterable, Mapping
from threading import Lock
from time import monotonic
from typing import Any

LATENCY_WINDOW_CAPACITY = 2_048
CHARACTER_BUCKETS = (49, 199, 499, 999, 4_999, 19_999, 100_000)
TOKEN_BUCKETS = (49, 99, 199, 511, 1_023, 4_095, 100_000)
SCORE_BUCKETS = tuple(round(index / 10, 1) for index in range(1, 11))
OBSERVED_ENDPOINTS = ("/v1/predict", "/v1/predict/batch")
KNOWN_CATEGORIES = ("likely_human", "uncertain", "likely_machine")
KNOWN_EVIDENCE = ("standard", "low")
KNOWN_WARNINGS = (
    "short_text_low_evidence",
    "length_outside_development_range",
    "format_out_of_profile",
)
KNOWN_ERRORS = (
    "request_validation",
    "text_type",
    "text_blank",
    "text_nul",
    "text_too_long",
    "model_not_ready",
    "batch_too_large",
    "batch_item_invalid",
    "unhandled_exception",
)


def _safe_label(value: object, known: tuple[str, ...]) -> str:
    return value if isinstance(value, str) and value in known else "other"


def bucket_index(value: float, boundaries: tuple[float | int, ...]) -> int:
    for index, boundary in enumerate(boundaries):
        if value <= boundary:
            return index
    return len(boundaries)


def render_histogram(
    boundaries: tuple[float | int, ...], counts: Counter[int]
) -> list[dict[str, float | int | None]]:
    rows: list[dict[str, float | int | None]] = [
        {"upper_bound": boundary, "count": counts[index]}
        for index, boundary in enumerate(boundaries)
    ]
    rows.append({"upper_bound": None, "count": counts[len(boundaries)]})
    return rows


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    rank = max(1, math.ceil(percentile * len(values)))
    return round(values[rank - 1], 3)


class OperationalMetrics:
    """Aggregate process-local service signals without retaining input records."""

    def __init__(self, *, latency_window_capacity: int = LATENCY_WINDOW_CAPACITY) -> None:
        if latency_window_capacity < 1:
            raise ValueError("latency_window_capacity must be positive")
        self._started = monotonic()
        self._lock = Lock()
        self._latencies_ms: deque[float] = deque(maxlen=latency_window_capacity)
        self._request_endpoints: Counter[str] = Counter()
        self._status_codes: Counter[int] = Counter()
        self._errors: Counter[str] = Counter()
        self._categories: Counter[str] = Counter()
        self._evidence: Counter[str] = Counter()
        self._warnings: Counter[str] = Counter()
        self._characters: Counter[int] = Counter()
        self._tokens: Counter[int] = Counter()
        self._scores: Counter[int] = Counter()
        self._successful_items = 0

    def record_http_request(self, endpoint: str, status_code: int, duration_ms: float) -> None:
        """Record one prediction HTTP request using fixed, non-input labels."""
        safe_endpoint = endpoint if endpoint in OBSERVED_ENDPOINTS else "other"
        safe_duration = max(0.0, float(duration_ms))
        with self._lock:
            self._request_endpoints[safe_endpoint] += 1
            self._status_codes[int(status_code)] += 1
            self._latencies_ms.append(safe_duration)

    def record_error(self, code: object) -> None:
        """Increment a bounded-cardinality error code."""
        with self._lock:
            self._errors[_safe_label(code, KNOWN_ERRORS)] += 1

    def record_predictions(self, results: Iterable[Mapping[str, Any]]) -> None:
        """Aggregate result distributions without retaining individual results."""
        aggregates: list[tuple[str, str, int, int, float, tuple[str, ...]]] = []
        for result in results:
            input_summary = result.get("input_summary", {})
            if not isinstance(input_summary, Mapping):
                input_summary = {}
            warnings = result.get("warnings", ())
            warning_codes = (
                tuple(
                    _safe_label(warning.get("code"), KNOWN_WARNINGS)
                    for warning in warnings
                    if isinstance(warning, Mapping)
                )
                if isinstance(warnings, Iterable) and not isinstance(warnings, (str, bytes))
                else ()
            )
            aggregates.append(
                (
                    _safe_label(result.get("category"), KNOWN_CATEGORIES),
                    _safe_label(result.get("evidence_quality"), KNOWN_EVIDENCE),
                    max(0, int(input_summary.get("characters", 0))),
                    max(0, int(input_summary.get("whitespace_tokens", 0))),
                    min(1.0, max(0.0, float(result.get("calibrated_machine_likelihood", 0.0)))),
                    warning_codes,
                )
            )

        with self._lock:
            for category, evidence, characters, tokens, score, warning_codes in aggregates:
                self._successful_items += 1
                self._categories[category] += 1
                self._evidence[evidence] += 1
                self._characters[bucket_index(characters, CHARACTER_BUCKETS)] += 1
                self._tokens[bucket_index(tokens, TOKEN_BUCKETS)] += 1
                self._scores[bucket_index(score, SCORE_BUCKETS)] += 1
                self._warnings.update(warning_codes)

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable aggregate snapshot."""
        with self._lock:
            latencies = sorted(self._latencies_ms)
            request_endpoints = dict(self._request_endpoints)
            status_codes = dict(self._status_codes)
            errors = dict(self._errors)
            categories = dict(self._categories)
            evidence = dict(self._evidence)
            warnings = dict(self._warnings)
            characters = self._characters.copy()
            tokens = self._tokens.copy()
            scores = self._scores.copy()
            successful_items = self._successful_items
            latency_capacity = self._latencies_ms.maxlen

        return {
            "schema_version": 1,
            "scope": "current_process",
            "uptime_seconds": round(max(0.0, monotonic() - self._started), 3),
            "retention": {
                "persistent": False,
                "per_request_records": False,
                "raw_text": False,
                "text_hashes": False,
                "latency_window_capacity": latency_capacity,
            },
            "requests": {
                "total": sum(request_endpoints.values()),
                "by_endpoint": {
                    endpoint: request_endpoints.get(endpoint, 0) for endpoint in OBSERVED_ENDPOINTS
                },
                "by_status_code": {
                    str(code): count for code, count in sorted(status_codes.items())
                },
                "errors_by_code": {code: errors.get(code, 0) for code in (*KNOWN_ERRORS, "other")},
                "latency_ms": {
                    "sample_size": len(latencies),
                    "p50": _nearest_rank(latencies, 0.50),
                    "p95": _nearest_rank(latencies, 0.95),
                    "p99": _nearest_rank(latencies, 0.99),
                    "maximum": round(latencies[-1], 3) if latencies else None,
                },
            },
            "predictions": {
                "successful_items": successful_items,
                "categories": {
                    category: categories.get(category, 0)
                    for category in (*KNOWN_CATEGORIES, "other")
                },
                "uncertain_rate": (
                    round(categories.get("uncertain", 0) / successful_items, 6)
                    if successful_items
                    else None
                ),
                "evidence_quality": {
                    quality: evidence.get(quality, 0) for quality in (*KNOWN_EVIDENCE, "other")
                },
                "warnings_by_code": {
                    code: warnings.get(code, 0) for code in (*KNOWN_WARNINGS, "other")
                },
                "character_histogram": render_histogram(CHARACTER_BUCKETS, characters),
                "token_histogram": render_histogram(TOKEN_BUCKETS, tokens),
                "calibrated_score_histogram": render_histogram(SCORE_BUCKETS, scores),
            },
        }
