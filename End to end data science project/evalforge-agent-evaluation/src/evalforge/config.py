"""Configuration loading and validation.

Configuration is layered: YAML file, then environment variables, then explicit
overrides passed by the CLI. Every layer is validated by Pydantic, and the effective
configuration is hashed into a ``config_digest`` that is stored with each run. That
digest is what lets someone answer "which thresholds produced this decision?" months
later without guessing.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from evalforge.exceptions import ConfigurationError

#: Directory containing the installed package.
PACKAGE_ROOT = Path(__file__).resolve().parent

#: Project root: the directory holding ``configs/``, ``data/`` and ``reports/``.
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


class Paths(BaseModel):
    """Filesystem locations EvalForge reads and writes.

    All paths are absolute and constructed with :mod:`pathlib`, so the package behaves
    identically on Windows and POSIX.
    """

    model_config = ConfigDict(extra="forbid")

    project_root: Path = PROJECT_ROOT
    config_dir: Path = DEFAULT_CONFIG_DIR
    data_dir: Path = PROJECT_ROOT / "data"
    reports_dir: Path = PROJECT_ROOT / "reports"

    @property
    def seed_scenarios(self) -> Path:
        """Hand-authored seed scenarios."""
        return self.data_dir / "seed_scenarios"

    @property
    def generated_scenarios(self) -> Path:
        """Programmatically generated scenario suites."""
        return self.data_dir / "generated_scenarios"

    @property
    def sample_documents(self) -> Path:
        """Fictional corpus the productivity tools search."""
        return self.data_dir / "sample_documents"

    @property
    def runs(self) -> Path:
        """Persisted evaluation runs and their traces."""
        return self.data_dir / "demonstration_runs"

    @property
    def annotations(self) -> Path:
        """Human annotation files."""
        return self.data_dir / "human_annotations"

    def run_dir(self, run_id: str) -> Path:
        """Directory holding everything belonging to ``run_id``."""
        return self.runs / run_id

    def ensure(self) -> None:
        """Create every directory EvalForge writes to."""
        for path in (
            self.data_dir,
            self.reports_dir,
            self.seed_scenarios,
            self.generated_scenarios,
            self.sample_documents,
            self.runs,
            self.annotations,
        ):
            path.mkdir(parents=True, exist_ok=True)


class ProviderConfig(BaseModel):
    """Model provider selection and request parameters."""

    model_config = ConfigDict(extra="forbid")

    name: str = "mock"
    model: str = "mock-productivity-agent-v1"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    timeout_seconds: int = Field(default=60, gt=0)
    max_retries: int = Field(default=2, ge=0)


class AgentConfig(BaseModel):
    """Behaviour of the agent under test."""

    model_config = ConfigDict(extra="forbid")

    version: str = "v1"
    prompt_version: str = "v1"
    max_tool_retries: int = Field(default=2, ge=0)
    max_tool_calls_per_turn: int = Field(default=6, gt=0)
    approval_required_tools: list[str] = Field(default_factory=list)
    treat_tool_output_as_untrusted: bool = True


class ScenarioConfig(BaseModel):
    """Scenario generation defaults."""

    model_config = ConfigDict(extra="forbid")

    default_count: int = Field(default=150, gt=0)
    default_seed: int = 42
    conversation_lengths: list[int] = Field(default_factory=lambda: [5, 10, 15, 20, 30])
    category_weights: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalise_weights(self) -> ScenarioConfig:
        total = sum(self.category_weights.values())
        if self.category_weights and abs(total - 1.0) > 1e-6:
            self.category_weights = {k: v / total for k, v in self.category_weights.items()}
        return self


class EvaluationConfig(BaseModel):
    """Which evaluator families run and how the judge is controlled."""

    model_config = ConfigDict(extra="forbid")

    run_deterministic: bool = True
    run_semantic: bool = True
    run_judge: bool = True
    judge_provider: str = "mock"
    judge_model: str = "mock-judge-v1"
    judge_prompt_version: str = "v1"
    judge_samples: int = Field(default=3, ge=1)
    judge_aggregation: str = "median"
    judge_max_retries: int = Field(default=2, ge=0)
    low_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    latency_threshold_ms: float = Field(default=4000.0, gt=0)
    max_allowed_retries: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def _check_aggregation(self) -> EvaluationConfig:
        allowed = {"median", "mean", "majority"}
        if self.judge_aggregation not in allowed:
            raise ValueError(f"judge_aggregation must be one of {sorted(allowed)}")
        return self


class StorageConfig(BaseModel):
    """Persistence backend selection."""

    model_config = ConfigDict(extra="forbid")

    backend: str = "sqlite"
    database_name: str = "evalforge.db"
    trace_format: str = "jsonl"


class LoggingConfig(BaseModel):
    """Structured logging setup."""

    model_config = ConfigDict(extra="forbid")

    level: str = "INFO"
    format: str = "console"


class CostConfig(BaseModel):
    """Token price table used for cost estimation."""

    model_config = ConfigDict(extra="forbid")

    input_per_million: float = Field(default=0.0, ge=0.0)
    output_per_million: float = Field(default=0.0, ge=0.0)

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate USD cost for a call. Returns 0.0 for the mock provider."""
        return (
            input_tokens * self.input_per_million + output_tokens * self.output_per_million
        ) / 1_000_000


class RubricConfig(BaseModel):
    """Dimension weights, pass thresholds and severity penalties."""

    model_config = ConfigDict(extra="forbid")

    version: str = "1.0.0"
    weights: dict[str, float] = Field(default_factory=dict)
    pass_thresholds: dict[str, float] = Field(default_factory=dict)
    session_pass_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    critical_failures: dict[str, str] = Field(default_factory=dict)
    severity_penalties: dict[str, float] = Field(default_factory=dict)
    judge_rubric: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> RubricConfig:
        if not self.weights:
            return self
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"rubric weights must sum to 1.0, got {total:.6f}")
        return self


class ThresholdSpec(BaseModel):
    """A single release gate."""

    model_config = ConfigDict(extra="forbid")

    value: float
    comparison: str = "gte"
    blocking: bool = True
    rationale: str = ""

    @model_validator(mode="after")
    def _check_comparison(self) -> ThresholdSpec:
        if self.comparison not in {"gte", "lte"}:
            raise ValueError("comparison must be 'gte' or 'lte'")
        return self

    def satisfied_by(self, observed: float) -> bool:
        """Whether ``observed`` clears this gate."""
        if self.comparison == "gte":
            return observed >= self.value
        return observed <= self.value


class ReleaseConfig(BaseModel):
    """Release gates plus regression tolerances."""

    model_config = ConfigDict(extra="forbid")

    version: str = "1.0.0"
    thresholds: dict[str, ThresholdSpec] = Field(default_factory=dict)
    regression_tolerances: dict[str, float] = Field(default_factory=dict)
    lower_is_better: list[str] = Field(default_factory=list)
    regression_tolerances_inverse: dict[str, float] = Field(default_factory=dict)

    def tolerance_for(self, metric: str) -> float | None:
        """Allowed signed change for ``metric``, or ``None`` if it is not gated."""
        if metric in self.regression_tolerances:
            return self.regression_tolerances[metric]
        return self.regression_tolerances_inverse.get(metric)

    def higher_is_better(self, metric: str) -> bool:
        """Whether an increase in ``metric`` is an improvement."""
        return metric not in self.lower_is_better


class BehaviorProfileConfig(BaseModel):
    """Degradation rates for one mock-model behaviour profile.

    Every rate is a probability in 0..1 applied against a seeded RNG, so a profile
    describes a reproducible agent, not a random one.
    """

    model_config = ConfigDict(extra="forbid")

    description: str = ""
    context_loss_after_turn: int = Field(default=1000, ge=0)
    context_loss_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    instruction_forget_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    goal_drift_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    wrong_tool_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    wrong_argument_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    excessive_tool_call_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    recovery_failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Probability of asserting a tool result that was never received. Kept separate
    #: from ``recovery_failure_rate`` because abandoning a step and inventing its output
    #: are different severities: major and critical respectively.
    fabrication_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    unsupported_claim_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    injection_compliance_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    cascade_propagation_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    unauthorized_action_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    clarification_rate: float = Field(default=1.0, ge=0.0, le=1.0)


class ToolFailureSpec(BaseModel):
    """Documentation for one injectable tool fault."""

    model_config = ConfigDict(extra="forbid")

    description: str = ""
    retryable: bool = False
    expected_recovery: str = ""
    latency_multiplier: float = Field(default=1.0, gt=0)


class FailureInjectionConfig(BaseModel):
    """Tool fault catalogue, behaviour profiles and latency model."""

    model_config = ConfigDict(extra="forbid")

    version: str = "1.0.0"
    tool_failures: dict[str, ToolFailureSpec] = Field(default_factory=dict)
    behavior_profiles: dict[str, BehaviorProfileConfig] = Field(default_factory=dict)
    tool_latency_ms: dict[str, float] = Field(default_factory=dict)
    latency_jitter_ratio: float = Field(default=0.25, ge=0.0, le=1.0)

    def profile(self, name: str) -> BehaviorProfileConfig:
        """Look up a behaviour profile by name.

        Raises:
            ConfigurationError: If the profile is not defined.
        """
        if name not in self.behavior_profiles:
            known = ", ".join(sorted(self.behavior_profiles))
            raise ConfigurationError(f"Unknown behavior profile {name!r}. Known profiles: {known}")
        return self.behavior_profiles[name]


class EvalForgeConfig(BaseModel):
    """The complete effective configuration for a run."""

    model_config = ConfigDict(extra="forbid")

    version: str = "1.0.0"
    paths: Paths = Field(default_factory=Paths)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    scenarios: ScenarioConfig = Field(default_factory=ScenarioConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    rubric: RubricConfig = Field(default_factory=RubricConfig)
    release: ReleaseConfig = Field(default_factory=ReleaseConfig)
    failure_injection: FailureInjectionConfig = Field(default_factory=FailureInjectionConfig)

    @property
    def digest(self) -> str:
        """Stable hash of the scoring-relevant configuration.

        Paths are excluded: where files live does not change what a score means, and
        including them would make digests differ between machines.
        """
        payload = self.model_dump(mode="json", exclude={"paths", "logging"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(encoded.encode("utf-8"), digest_size=8).hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping, raising a typed error on any problem."""
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Malformed YAML in {path}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigurationError(f"{path} must contain a mapping at the top level")
    return raw


def _env_overrides() -> dict[str, Any]:
    """Collect configuration overrides from the environment.

    Only a small, explicit set of variables is honoured. Blanket ``EVALFORGE_*``
    mapping was rejected because it makes misconfiguration silent.
    """
    overrides: dict[str, Any] = {}
    provider: dict[str, Any] = {}
    evaluation: dict[str, Any] = {}
    logging_cfg: dict[str, Any] = {}

    if value := os.environ.get("EVALFORGE_PROVIDER"):
        provider["name"] = value
    if value := os.environ.get("EVALFORGE_MODEL"):
        provider["model"] = value
    if value := os.environ.get("EVALFORGE_JUDGE_PROVIDER"):
        evaluation["judge_provider"] = value
    if value := os.environ.get("EVALFORGE_LOG_LEVEL"):
        logging_cfg["level"] = value.upper()
    if value := os.environ.get("EVALFORGE_LOG_FORMAT"):
        logging_cfg["format"] = value

    if provider:
        overrides["provider"] = provider
    if evaluation:
        overrides["evaluation"] = evaluation
    if logging_cfg:
        overrides["logging"] = logging_cfg
    return overrides


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` without mutating either."""
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def load_config(
    config_dir: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> EvalForgeConfig:
    """Build the effective configuration.

    Args:
        config_dir: Directory containing the four YAML files. Defaults to the
            project's ``configs/``.
        overrides: Explicit overrides applied last, typically from CLI flags.

    Returns:
        A fully validated :class:`EvalForgeConfig`.

    Raises:
        ConfigurationError: If a file is missing, malformed or fails validation.
    """
    directory = config_dir or DEFAULT_CONFIG_DIR
    base = _read_yaml(directory / "default.yaml")
    base["rubric"] = _read_yaml(directory / "evaluation_rubrics.yaml")
    base["release"] = _read_yaml(directory / "release_thresholds.yaml")
    base["failure_injection"] = _read_yaml(directory / "failure_injection.yaml")

    paths_payload: dict[str, Any] = {"config_dir": directory}
    data_dir_env = os.environ.get("EVALFORGE_DATA_DIR")
    reports_dir_env = os.environ.get("EVALFORGE_REPORTS_DIR")
    if data_dir_env:
        paths_payload["data_dir"] = Path(data_dir_env)
        # Reports follow the data directory unless overridden. Without this, redirecting
        # only the data directory leaves reports writing to the real project folder —
        # which meant CLI tests were quietly polluting committed artifacts.
        paths_payload["reports_dir"] = Path(data_dir_env).parent / "reports"
    if reports_dir_env:
        paths_payload["reports_dir"] = Path(reports_dir_env)
    base["paths"] = paths_payload

    merged = _deep_merge(base, _env_overrides())
    if overrides:
        merged = _deep_merge(merged, overrides)

    try:
        return EvalForgeConfig.model_validate(merged)
    except Exception as exc:
        raise ConfigurationError(f"Invalid EvalForge configuration: {exc}") from exc


@lru_cache(maxsize=1)
def get_config() -> EvalForgeConfig:
    """Return the process-wide configuration, loaded once.

    Tests that need a different configuration should call :func:`load_config`
    directly and inject the result rather than mutating this cache.
    """
    return load_config()


def clear_config_cache() -> None:
    """Drop the cached configuration. Used by tests that alter the environment."""
    get_config.cache_clear()
