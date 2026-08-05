"""Configuration loading, validation and seeded randomness."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from evalforge.config import EvalForgeConfig, load_config
from evalforge.exceptions import ConfigurationError
from evalforge.rng import chance, choose, derive_seed, jitter, sample, seeded_random, shuffled
from evalforge.schemas.common import Dimension


class TestConfigLoading:
    """The shipped configuration must load and be internally consistent."""

    def test_loads_all_four_files(self, config: EvalForgeConfig) -> None:
        assert config.rubric.weights
        assert config.release.thresholds
        assert config.failure_injection.behavior_profiles
        assert config.scenarios.category_weights

    def test_rubric_weights_sum_to_one(self, config: EvalForgeConfig) -> None:
        assert sum(config.rubric.weights.values()) == pytest.approx(1.0)

    def test_every_weight_names_a_real_dimension(self, config: EvalForgeConfig) -> None:
        known = {item.value for item in Dimension}
        unknown = set(config.rubric.weights) - known
        assert unknown == set(), f"rubric weights reference unknown dimensions: {unknown}"

    def test_every_threshold_has_a_rationale(self, config: EvalForgeConfig) -> None:
        """ADR-005 requires a stated reason next to every number."""
        missing = [name for name, spec in config.release.thresholds.items() if not spec.rationale]
        assert missing == [], f"thresholds without a rationale: {missing}"

    def test_required_profiles_exist(self, config: EvalForgeConfig) -> None:
        for name in ("baseline", "candidate", "perfect", "broken"):
            assert config.failure_injection.profile(name) is not None

    def test_unknown_profile_raises_with_a_helpful_message(self, config: EvalForgeConfig) -> None:
        with pytest.raises(ConfigurationError, match="Known profiles"):
            config.failure_injection.profile("does-not-exist")

    def test_perfect_profile_has_no_degradation(self, config: EvalForgeConfig) -> None:
        """The false-positive control must be genuinely clean."""
        profile = config.failure_injection.profile("perfect")
        rates = [
            profile.context_loss_rate,
            profile.instruction_forget_rate,
            profile.goal_drift_rate,
            profile.wrong_tool_rate,
            profile.wrong_argument_rate,
            profile.fabrication_rate,
            profile.injection_compliance_rate,
            profile.unauthorized_action_rate,
        ]
        assert all(rate == 0.0 for rate in rates)

    def test_baseline_never_fabricates(self, config: EvalForgeConfig) -> None:
        """The reference agent may recover poorly but must not invent tool results."""
        assert config.failure_injection.profile("baseline").fabrication_rate == 0.0

    def test_candidate_is_worse_than_baseline(self, config: EvalForgeConfig) -> None:
        baseline = config.failure_injection.profile("baseline")
        candidate = config.failure_injection.profile("candidate")
        assert candidate.context_loss_rate > baseline.context_loss_rate
        assert candidate.instruction_forget_rate > baseline.instruction_forget_rate
        assert candidate.injection_compliance_rate > baseline.injection_compliance_rate


class TestConfigValidation:
    """Malformed configuration must fail loudly rather than silently."""

    def _write(self, directory: Path, overrides: dict) -> None:
        source = Path(__file__).resolve().parents[2] / "configs"
        for name in (
            "default.yaml",
            "evaluation_rubrics.yaml",
            "release_thresholds.yaml",
            "failure_injection.yaml",
        ):
            payload = yaml.safe_load((source / name).read_text(encoding="utf-8"))
            if name in overrides:
                payload.update(overrides[name])
            (directory / name).write_text(yaml.safe_dump(payload), encoding="utf-8")

    def test_weights_not_summing_to_one_is_rejected(self, tmp_path: Path) -> None:
        self._write(tmp_path, {"evaluation_rubrics.yaml": {"weights": {"safety": 0.5}}})
        with pytest.raises(ConfigurationError, match=r"sum to 1\.0"):
            load_config(tmp_path)

    def test_missing_file_is_reported_by_name(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            load_config(tmp_path)

    def test_unknown_key_is_rejected(self, tmp_path: Path) -> None:
        """``extra="forbid"`` means a typo is an error, not a silently ignored setting."""
        self._write(tmp_path, {"default.yaml": {"provider": {"nmae": "mock"}}})
        with pytest.raises(ConfigurationError):
            load_config(tmp_path)


class TestConfigDigest:
    """The digest ties a stored run to the policy that scored it."""

    def test_digest_is_stable(self, config: EvalForgeConfig) -> None:
        assert config.digest == load_config().digest

    def test_digest_changes_when_a_threshold_changes(self, config: EvalForgeConfig) -> None:
        altered = config.model_copy(deep=True)
        altered.rubric.session_pass_threshold = 0.5
        assert altered.digest != config.digest

    def test_digest_ignores_paths(self, config: EvalForgeConfig) -> None:
        """Where files live does not change what a score means."""
        altered = config.model_copy(deep=True)
        altered.paths.data_dir = Path("/somewhere/else")
        assert altered.digest == config.digest


class TestSeededRandomness:
    """Determinism is the property the regression gate rests on."""

    def test_derive_seed_is_stable_across_calls(self) -> None:
        assert derive_seed("run", 42, "scn_1") == derive_seed("run", 42, "scn_1")

    def test_different_coordinates_give_different_seeds(self) -> None:
        assert derive_seed("run", 42, "scn_1") != derive_seed("run", 42, "scn_2")

    def test_chance_is_reproducible(self) -> None:
        first = [chance(0.5, "run", i, "aspect") for i in range(200)]
        second = [chance(0.5, "run", i, "aspect") for i in range(200)]
        assert first == second

    def test_chance_respects_its_bounds(self) -> None:
        assert chance(0.0, "x") is False
        assert chance(1.0, "x") is True

    def test_chance_is_roughly_calibrated(self) -> None:
        hits = sum(chance(0.3, "calibration", i) for i in range(4000))
        assert 0.26 < hits / 4000 < 0.34

    def test_choose_and_sample_are_reproducible(self) -> None:
        options = list("abcdefghij")
        assert choose(options, "s", 1) == choose(options, "s", 1)
        assert sample(options, 4, "s", 1) == sample(options, 4, "s", 1)
        assert len(sample(options, 4, "s", 1)) == 4

    def test_sample_larger_than_population_returns_everything(self) -> None:
        options = list("abc")
        assert sorted(sample(options, 10, "s")) == ["a", "b", "c"]

    def test_choose_rejects_an_empty_sequence(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            choose([], "s")

    def test_shuffled_preserves_membership(self) -> None:
        options = list(range(20))
        assert sorted(shuffled(options, "s")) == options

    def test_jitter_stays_within_bounds(self) -> None:
        for index in range(200):
            value = jitter(100.0, 0.25, "latency", index)
            assert 75.0 <= value <= 125.0

    def test_jitter_of_zero_ratio_is_identity(self) -> None:
        assert jitter(100.0, 0.0, "x") == 100.0

    def test_seeded_random_is_independent_of_call_order(self) -> None:
        """A decision must depend only on its own coordinates.

        This is what lets scenarios run in any order, or in parallel, and still produce
        identical faults.
        """
        direct = seeded_random("a", 1).random()
        seeded_random("b", 2).random()
        seeded_random("c", 3).random()
        assert seeded_random("a", 1).random() == direct
