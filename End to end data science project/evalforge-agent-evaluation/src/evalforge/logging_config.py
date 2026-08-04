"""Structured logging.

Logs are structured because evaluation debugging is a filtering problem: "show me every
tool error in run X on scenario Y" is trivial against key-value events and painful
against formatted strings. Console rendering is used interactively; JSON in CI.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_CONFIGURED = False


def configure_logging(level: str = "INFO", output_format: str = "console") -> None:
    """Configure structlog and the stdlib logging bridge.

    Args:
        level: Standard logging level name.
        output_format: ``console`` for human-readable output, ``json`` for CI.

    Calling this more than once is a no-op, so importing a module that logs at import
    time cannot clobber an explicit configuration.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=numeric_level)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if output_format == "json":
        processors.append(structlog.processors.format_exc_info)
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for ``name``, configuring logging on first use."""
    if not _CONFIGURED:
        configure_logging()
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger


def bind_run_context(run_id: str, scenario_id: str | None = None) -> None:
    """Attach run and scenario identifiers to every subsequent log line.

    Using context vars rather than passing ids through every call site keeps the
    tool and agent code readable while still producing filterable logs.
    """
    structlog.contextvars.bind_contextvars(run_id=run_id)
    if scenario_id is not None:
        structlog.contextvars.bind_contextvars(scenario_id=scenario_id)


def clear_run_context() -> None:
    """Remove any bound run context."""
    structlog.contextvars.clear_contextvars()


def reset_logging() -> None:
    """Allow reconfiguration. Intended for tests only."""
    global _CONFIGURED
    _CONFIGURED = False
