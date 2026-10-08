from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from keel.evaluation import scenarios
from keel.evaluation.benchmark import load_config
from keel.evaluation.round2 import run_round2, to_markdown
from keel.evaluation.scenarios import HOLDOUT, SLOTS, build_personas
from keel.memory.embeddings import HashingEmbedder
from keel.memory.retrieval import EMBEDDING_WEIGHT, EmbeddingMemory, KeelMemory, default_retriever

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "eval.toml"


def test_hashing_embedder_is_unit_norm_deterministic_and_cached():
    embedder = HashingEmbedder()
    first = embedder.embed(["hello world", "goodbye"])
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
    assert np.array_equal(first, HashingEmbedder().embed(["hello world", "goodbye"]))
    assert len(embedder._cache) == 2
    embedder.embed(["hello world"])
    assert len(embedder._cache) == 2


def test_embedding_arm_ranks_by_similarity(store, clock):
    store.add("Commutes by bicycle every day.", kind="fact")
    store.add("Owns a sailboat.", kind="fact")
    chosen = EmbeddingMemory(HashingEmbedder()).retrieve(store.all(), "bicycles", clock.now(), 1)
    assert chosen[0].text == "Commutes by bicycle every day."


def test_embeddings_find_a_memory_with_no_shared_word(store, clock):
    # "cycling" and "bicycle" stem differently, so BM25 scores both memories zero.
    # The bicycle memory is the oldest, so recency alone would never pick it.
    store.add("Commutes by bicycle.", kind="fact")
    clock.advance(days=1)
    store.add("Owns a sailboat.", kind="fact")
    store.add("Keeps the sailboat at the marina.", kind="fact")
    memories = store.all()
    plain = KeelMemory(core_max=0).retrieve(memories, "cycling", clock.now(), 1)
    hybrid = KeelMemory(core_max=0, embedder=HashingEmbedder(), embedding_weight=1.0)
    assert plain[0].text != "Commutes by bicycle."
    assert hybrid.retrieve(memories, "cycling", clock.now(), 1)[0].text == "Commutes by bicycle."


def test_zero_weight_matches_plain_keel(store, clock):
    for i in range(6):
        clock.advance(hours=1)
        store.add(f"Memory number {i} about topic {i % 3}.", kind="fact")
    memories = store.all()
    plain = KeelMemory().retrieve(memories, "topic 1", clock.now(), 4)
    zero = KeelMemory(embedder=HashingEmbedder(), embedding_weight=0.0)
    assert plain == zero.retrieve(memories, "topic 1", clock.now(), 4)


def test_holdout_has_one_direct_and_one_indirect_wording_per_detail():
    persona = build_personas(dict(load_config(CONFIG)["benchmark"], personas=1))[0]
    assert len(persona.probes["holdout_direct"]) == len(SLOTS)
    assert len(persona.probes["holdout_indirect"]) == len(SLOTS)
    assert len(persona.probes["holdout"]) == 2 * len(SLOTS)
    for slot in SLOTS:
        assert not set(HOLDOUT[slot.key]) & (set(slot.dev) | set(slot.test))


def test_every_held_out_set_is_new_and_complete():
    """Each held-out set covers every detail and repeats no wording used anywhere else."""
    names = ["HOLDOUT"] + [f"HOLDOUT{i}" for i in range(2, 12)]
    sets = [getattr(scenarios, name) for name in names]
    keys = {slot.key for slot in SLOTS}
    seen = {q.lower() for slot in SLOTS for q in (*slot.dev, *slot.test)}
    for name, wordings in zip(names, sets, strict=True):
        assert set(wordings) == keys, name
        for key in keys:
            new = {q.lower() for q in wordings[key]}
            assert len(new) == 2, (name, key)
            assert not new & seen, (name, key, new & seen)
            seen |= new
    persona = build_personas(dict(load_config(CONFIG)["benchmark"], personas=1))[0]
    assert len(persona.probes["holdout10"]) == 2 * len(SLOTS)
    assert len(persona.probes["holdout11"]) == 2 * len(SLOTS)


def test_round2_runner_with_a_stand_in_embedder(tmp_path):
    small = tmp_path / "eval.toml"
    small.write_text(CONFIG.read_text().replace("personas = 200", "personas = 4"))
    results = run_round2(small, HashingEmbedder())
    assert results["chosen_weight"] in load_config(CONFIG)["round2"]["weight_grid"]
    assert set(results["splits"]) == {"holdout", "holdout_direct", "holdout_indirect", "test"}
    assert "keel+embed" in results["splits"]["holdout"]
    assert "Round 2: embedding search" in to_markdown(results)


def test_wordllama_loads_offline_and_is_used_by_default(monkeypatch):
    pytest.importorskip("wordllama")
    import requests

    def no_network(*args, **kwargs):
        raise AssertionError("tried to download")

    monkeypatch.setattr(requests, "get", no_network)
    from keel.memory.embeddings import WordLlamaEmbedder

    embedder = WordLlamaEmbedder()
    vectors = embedder.embed(
        ["Who is the vet appointment for?", "Has a beagle named Biscuit.", "Works at a bank."]
    )
    assert vectors.shape == (3, 256)
    assert vectors[1] @ vectors[0] > vectors[2] @ vectors[0]

    retriever = default_retriever()
    assert retriever.embedder is not None
    assert retriever.embedding_weight == EMBEDDING_WEIGHT
