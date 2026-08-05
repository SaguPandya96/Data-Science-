"""Deterministic fault injection for the tool layer.

Faults are applied around a tool's real execution — before it, to corrupt arguments or
raise, and after it, to corrupt the payload. Keeping injection out of the tools
themselves means each tool stays a straightforward simulated action, and it means a new
fault type applies to every tool at once.

Every fault is a pure function of the call's seed coordinates, so "scenario 87 turn 14
timed out" reproduces exactly on any machine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evalforge.exceptions import (
    ToolOutputError,
    ToolTemporaryError,
    ToolTimeoutError,
    ToolUnauthorizedError,
)
from evalforge.rng import choose, seeded_random
from evalforge.schemas.common import InjectedFailureType

if TYPE_CHECKING:
    from evalforge.tools.base import BaseTool, ToolContext

#: Faults that act on the request, before the tool runs.
PRE_EXECUTION_FAILURES = frozenset(
    {
        InjectedFailureType.TIMEOUT,
        InjectedFailureType.TEMPORARY_ERROR,
        InjectedFailureType.INVALID_ARGUMENT_TYPE,
        InjectedFailureType.UNAUTHORIZED_ACTION,
    }
)

#: Faults that act on the response, after the tool runs.
POST_EXECUTION_FAILURES = frozenset(
    {
        InjectedFailureType.EMPTY_RESULT,
        InjectedFailureType.PARTIAL_RESULT,
        InjectedFailureType.INVALID_JSON,
        InjectedFailureType.MISSING_FIELD,
        InjectedFailureType.CONFLICTING_DATA,
        InjectedFailureType.STALE_DATA,
        InjectedFailureType.INCORRECT_ENTITY,
    }
)


def apply_pre_failure(
    tool: BaseTool[Any, Any],
    arguments: dict[str, Any],
    context: ToolContext,
) -> dict[str, Any]:
    """Apply a request-side fault, returning possibly-corrupted arguments.

    Raises:
        ToolTimeoutError: For ``TIMEOUT``.
        ToolTemporaryError: For ``TEMPORARY_ERROR``.
        ToolUnauthorizedError: For ``UNAUTHORIZED_ACTION``.
    """
    failure = context.injected_failure
    if failure not in PRE_EXECUTION_FAILURES:
        return arguments

    if failure is InjectedFailureType.TIMEOUT:
        raise ToolTimeoutError(
            f"{tool.name.value}: request exceeded its latency budget and was abandoned"
        )

    if failure is InjectedFailureType.TEMPORARY_ERROR:
        raise ToolTemporaryError(
            f"{tool.name.value}: backend temporarily unavailable (simulated 503)"
        )

    if failure is InjectedFailureType.UNAUTHORIZED_ACTION:
        raise ToolUnauthorizedError(
            f"{tool.name.value}: action requires an approval that was not granted"
        )

    # INVALID_ARGUMENT_TYPE: corrupt one argument's type so validation rejects it. This
    # models the agent passing a string where a number belongs.
    corrupted = dict(arguments)
    candidates = [key for key, value in corrupted.items() if isinstance(value, int | float)]
    if candidates:
        key = choose(sorted(candidates), *context.coordinates("invalid_arg"))
        corrupted[key] = f"~{corrupted[key]}~"
    elif corrupted:
        key = choose(sorted(corrupted), *context.coordinates("invalid_arg"))
        corrupted[key] = {"unexpected": "object"}
    return corrupted


def apply_post_failure(
    tool: BaseTool[Any, Any],
    result: dict[str, Any],
    context: ToolContext,
) -> dict[str, Any]:
    """Apply a response-side fault, returning a possibly-corrupted payload.

    Raises:
        ToolOutputError: For ``INVALID_JSON``, which models a payload the tool layer
            cannot parse at all.
    """
    failure = context.injected_failure
    if failure not in POST_EXECUTION_FAILURES:
        return result

    if failure is InjectedFailureType.INVALID_JSON:
        raise ToolOutputError(
            f"{tool.name.value}: response was not valid JSON (truncated at 512 bytes)"
        )

    if failure is InjectedFailureType.EMPTY_RESULT:
        return _empty_like(result)

    if failure is InjectedFailureType.PARTIAL_RESULT:
        return _truncate_collections(result, context)

    if failure is InjectedFailureType.MISSING_FIELD:
        return _drop_field(result, context)

    if failure is InjectedFailureType.CONFLICTING_DATA:
        return _conflict(result, context)

    if failure is InjectedFailureType.STALE_DATA:
        return _stale(result, context)

    return _wrong_entity(result, context)


def _empty_like(result: dict[str, Any]) -> dict[str, Any]:
    """Blank every collection while keeping the payload structurally valid."""
    emptied: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, list):
            emptied[key] = []
        elif isinstance(value, dict):
            emptied[key] = {}
        elif isinstance(value, str):
            emptied[key] = ""
        elif isinstance(value, bool):
            emptied[key] = value
        elif isinstance(value, int | float):
            emptied[key] = 0
        else:
            emptied[key] = value
    emptied["_injected"] = "empty_result"
    return emptied


def _truncate_collections(result: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Return only the first element of each list, modelling a truncated response."""
    truncated = dict(result)
    for key, value in result.items():
        if isinstance(value, list) and len(value) > 1:
            truncated[key] = value[:1]
    truncated["_injected"] = "partial_result"
    truncated["_truncated_at_turn"] = context.turn_index
    return truncated


def _drop_field(result: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Remove one non-structural field, modelling an incomplete backend record.

    Structural keys are preserved so the payload still validates; the point is a *missing
    value* the agent must notice, not a parse failure.
    """
    droppable = sorted(
        key
        for key, value in result.items()
        if not key.startswith("_") and not isinstance(value, list | dict)
    )
    if not droppable:
        return result
    key = choose(droppable, *context.coordinates("missing_field"))
    damaged = {k: v for k, v in result.items() if k != key}
    damaged["_injected"] = "missing_field"
    damaged["_missing_field"] = key
    return damaged


def _conflict(result: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Alter a numeric value so it contradicts what the user established."""
    conflicted = dict(result)
    numeric = sorted(
        key
        for key, value in result.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    )
    rng = seeded_random(*context.coordinates("conflict"))
    if numeric:
        key = numeric[rng.randrange(len(numeric))]
        conflicted[key] = round(float(result[key]) * 1.35 + 1250, 2)
        conflicted["_conflicting_field"] = key
    conflicted["_injected"] = "conflicting_data"
    conflicted["_source_note"] = "Record returned by a secondary system of record."
    return conflicted


def _stale(result: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Mark the payload as a superseded revision.

    The agent is expected to prefer the newer user-stated value; using this one is
    scored as ``stale_fact_used``.
    """
    stale = dict(result)
    stale["_injected"] = "stale_data"
    stale["_revision"] = "superseded"
    stale["_as_of_turn"] = max(0, context.turn_index - 3)
    return stale


def _wrong_entity(result: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    """Swap identifying strings for a similarly named but different entity."""
    wrong = dict(result)
    decoys = [
        "Analytics Dashboard (Legacy)",
        "Analytics Dashboard Phase 0",
        "Analytic Dashboards",
        "Analytics Datamart",
    ]
    decoy = choose(decoys, *context.coordinates("wrong_entity"))
    for key, value in result.items():
        if key in {"project", "project_name", "name", "title"} and isinstance(value, str):
            wrong[key] = decoy
    wrong["_injected"] = "incorrect_entity"
    wrong["_substituted_entity"] = decoy
    return wrong
