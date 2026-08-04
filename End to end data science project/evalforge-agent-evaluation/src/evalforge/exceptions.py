"""Explicit exception hierarchy.

Every failure mode EvalForge can hit has a named type. The CLI maps these onto exit
codes and renders them as readable messages rather than tracebacks, so an operator who
mistypes a run id sees a sentence, not a stack.
"""

from __future__ import annotations


class EvalForgeError(Exception):
    """Base class for every error raised by EvalForge."""


class ConfigurationError(EvalForgeError):
    """A configuration file is missing, malformed or internally inconsistent."""


class ScenarioError(EvalForgeError):
    """A scenario is invalid or cannot be generated."""


class ScenarioValidationError(ScenarioError):
    """A generated scenario failed the quality gate.

    Args:
        scenario_id: Identifier of the offending scenario.
        problems: Human-readable descriptions of every rule the scenario broke.
    """

    def __init__(self, scenario_id: str, problems: list[str]) -> None:
        self.scenario_id = scenario_id
        self.problems = problems
        joined = "; ".join(problems)
        super().__init__(f"Scenario {scenario_id} failed validation: {joined}")


class ToolError(EvalForgeError):
    """Base class for tool-layer failures."""

    retryable: bool = False


class ToolValidationError(ToolError):
    """Tool arguments failed schema validation. Never retryable."""

    retryable = False


class ToolTimeoutError(ToolError):
    """A tool exceeded its simulated latency budget. Retryable."""

    retryable = True


class ToolTemporaryError(ToolError):
    """A transient backend fault. Retryable."""

    retryable = True


class ToolPermanentError(ToolError):
    """A fault that will recur on retry, such as a missing document."""

    retryable = False


class ToolUnauthorizedError(ToolError):
    """The agent attempted an action requiring approval it had not obtained."""

    retryable = False


class ProviderError(EvalForgeError):
    """Base class for model-provider failures."""


class ProviderUnavailableError(ProviderError):
    """A provider was requested but its credentials or SDK are absent.

    This is the expected path when someone clones the repository without API keys;
    the message tells them to use the mock provider rather than failing obscurely.
    """


class ProviderResponseError(ProviderError):
    """A provider returned a response EvalForge could not parse."""


class EvaluationError(EvalForgeError):
    """An evaluator could not produce a result."""


class JudgeError(EvaluationError):
    """The LLM judge returned unusable output after all retries."""


class StorageError(EvalForgeError):
    """Persistence failed or a requested record does not exist."""


class RunNotFoundError(StorageError):
    """A run id does not exist in the store.

    Args:
        run_id: The identifier that was not found.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"Run {run_id!r} not found. List available runs with `evalforge runs`.")


class RegressionGateError(EvalForgeError):
    """A candidate run regressed beyond the configured tolerance.

    Args:
        violations: One message per metric that breached its tolerance.
    """

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("Regression gate failed:\n" + "\n".join(f"  - {v}" for v in violations))
