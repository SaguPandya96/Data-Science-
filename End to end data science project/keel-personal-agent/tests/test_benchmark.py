from __future__ import annotations

from pathlib import Path

import numpy as np

from keel.evaluation.benchmark import bootstrap_mean, load_config, paired_difference, score_arm
from keel.evaluation.live import mentions, run_live
from keel.evaluation.scenarios import SLOTS, build_personas
from keel.memory.retrieval import KeelMemory, LexicalMemory, RecentMemory
from keel.model import text

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "eval.toml"


def small_config(personas: int = 8) -> dict:
    return dict(load_config(CONFIG)["benchmark"], personas=personas)


def test_personas_are_deterministic_and_well_formed():
    first = build_personas(small_config(3))
    second = build_personas(small_config(3))
    assert [m.text for m in first[0].memories] == [m.text for m in second[0].memories]
    for persona in first:
        for split in ("dev", "test"):
            probes = persona.probes[split]
            assert len(probes) == len(SLOTS)
            for probe in probes:
                assert persona.store.get(probe.current_id).active
                assert all(not persona.store.get(i).active for i in probe.stale_ids)
                assert len(probe.stale_values) == len(probe.stale_ids)


def test_dev_and_test_templates_never_overlap():
    for slot in SLOTS:
        assert not set(slot.dev) & set(slot.test), slot.key


def test_key_noise_changes_keys_not_histories():
    clean = build_personas(small_config(3))
    noisy = build_personas(small_config(3), key_noise=1.0)
    for a, b in zip(clean, noisy, strict=True):
        assert [m.text for m in a.memories] == [m.text for m in b.memories]
    assert sum(len(p.store.all()) for p in noisy) > sum(len(p.store.all()) for p in clean)


def test_keel_beats_baselines_on_a_small_sample():
    personas = build_personas(small_config())
    scores = {
        arm.name: score_arm(arm, personas, "dev", 5).all["clean_hit"].mean()
        for arm in (RecentMemory(), LexicalMemory(), KeelMemory())
    }
    assert scores["keel"] > scores["lexical"] > scores["recent"]


def test_bootstrap_interval_contains_mean():
    stat = bootstrap_mean(np.array([0.2, 0.4, 0.6, 0.8]), 500, 0.95, 1)
    assert stat["low"] <= stat["mean"] <= stat["high"]
    diff = paired_difference(np.array([1.0, 1.0]), np.array([0.0, np.nan]), 100, 0.95, 1)
    assert diff["n"] == 1 and diff["mean"] == 1.0


def test_answer_grading():
    assert mentions("You live in Austin.", "Austin")
    assert mentions("Your dog Biscuit, a beagle.", "a beagle named Biscuit")
    assert not mentions("A cat.", "a beagle named Biscuit")
    assert mentions("You wake at 5:30 am.", "5:30 am")
    assert mentions("You're vegetarian.", "vegetarian")
    assert mentions("About $300 a month.", "$300 a month")


class EchoModel:
    """Answers with whatever the first memory line says."""

    def reply(self, system, messages, tools):  # type: ignore[no-untyped-def]
        prompt = messages[0]["content"]
        lines = [line for line in prompt.splitlines() if line.startswith("- #")]
        return text(" ".join(lines) or "I don't know.")


def test_live_runner_with_a_fake_model():
    results = run_live(CONFIG, EchoModel(), personas=2, arms=["recent", "keel"])
    assert set(results["accuracy"]) == {"recent", "keel"}
    assert results["accuracy"]["keel"]["mean"] > results["accuracy"]["recent"]["mean"]
