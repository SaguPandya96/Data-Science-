from __future__ import annotations

import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from keel.evaluation.benchmark import load_config
from keel.evaluation.round7 import load_rerankers, run_round7, to_markdown
from keel.evaluation.round8 import run_round8
from keel.evaluation.round8 import to_markdown as round8_markdown
from keel.evaluation.round9 import run_round9
from keel.evaluation.round9 import to_markdown as round9_markdown
from keel.evaluation.round11 import run_round11
from keel.evaluation.round11 import to_markdown as round11_markdown
from keel.evaluation.round17 import run_round17
from keel.evaluation.round17 import to_markdown as round17_markdown
from keel.memory.embeddings import DEFAULT_ENCODER, HashingEmbedder, download_files
from keel.memory.rerank import (
    DEFAULT_RERANKER,
    RERANK_CANDIDATES,
    RERANK_WEIGHT,
    OverlapReranker,
    RerankedMemory,
    default_reranker,
)
from keel.memory.retrieval import TRANSFORMER_WEIGHT, KeelMemory, default_retriever

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "eval.toml"


class TableReranker:
    """Scores from a fixed table, so a test can say exactly what the reranker prefers."""

    name = "table"

    def __init__(self, table: dict[str, float]) -> None:
        self.table = table
        self.calls: list[list[str]] = []

    def score(self, query, docs):  # type: ignore[no-untyped-def]
        self.calls.append(list(docs))
        return np.array([self.table.get(d, 0.0) for d in docs], dtype=np.float32)


def _memories(store, clock):  # type: ignore[no-untyped-def]
    texts = [
        ("Allergic to peanuts.", "constraint", "allergy"),
        ("Moved to Lisbon.", "fact", "home_city"),
        ("Works at Acme.", "fact", "employer"),
        ("Asked how long to boil an egg.", "episode", None),
        ("Wanted a podcast about history.", "episode", None),
    ]
    for text, kind, key in texts:
        clock.set(clock.now() + timedelta(hours=1))
        store.add(text, kind=kind, key=key)
    return store.all(include_inactive=True)


def test_reranker_alone_reorders_the_short_list(store, clock):
    memories = _memories(store, clock)
    base = KeelMemory(core_max=1)
    rest = base.score_pool(memories, "where do I live", clock.now(), 3)[1]
    favourite = base.document(rest[-1])
    reranker = TableReranker({favourite: 10.0})
    arm = RerankedMemory(base=base, reranker=reranker, candidates=len(rest))
    got = arm.retrieve(memories, "where do I live", clock.now(), 3)
    assert got[0].kind == "constraint"  # constraints still go first, untouched
    assert got[1].id == rest[-1].id
    assert len(got) == 3


def test_memories_outside_the_short_list_are_never_scored_or_shown(store, clock):
    memories = _memories(store, clock)
    base = KeelMemory(core_max=1)
    _, rest, scores = base.score_pool(memories, "Lisbon", clock.now(), 3)
    order = sorted(range(len(rest)), key=lambda i: (scores[i], rest[i].id), reverse=True)
    outside = base.document(rest[order[-1]])
    reranker = TableReranker({outside: 100.0})
    arm = RerankedMemory(base=base, reranker=reranker, candidates=2)
    got = arm.retrieve(memories, "Lisbon", clock.now(), 3)
    assert all(base.document(m) != outside for m in got)
    assert all(outside not in call for call in reranker.calls)
    assert len(reranker.calls[0]) == 2


def test_blend_with_zero_weight_matches_keel(store, clock):
    memories = _memories(store, clock)
    base = KeelMemory(core_max=1)
    arm = RerankedMemory(base=base, reranker=OverlapReranker(), candidates=10, rerank_weight=0.0)
    for query in ("where do I live", "who do I work for", "podcast"):
        expected = base.retrieve(memories, query, clock.now(), 3)
        assert arm.retrieve(memories, query, clock.now(), 3) == expected


def test_overlap_reranker_caches_and_scores_shared_text_higher():
    reranker = OverlapReranker()
    scores = reranker.score("Lisbon apartment", ["Moved to Lisbon.", "Works at Acme."])
    assert scores[0] > scores[1]
    assert (reranker.score("Lisbon apartment", ["Moved to Lisbon."]) == scores[:1]).all()


def _files(tmp_path: Path) -> dict[str, dict[str, str]]:
    files = {}
    for name, data in (("model.onnx", b"onnx"), ("tokenizer.json", b"{}")):
        source = tmp_path / f"src-{name}"
        source.write_bytes(data)
        files[name] = {"url": source.as_uri(), "sha256": hashlib.sha256(data).hexdigest()}
    return files


def test_file_download_verifies_and_caches(tmp_path, monkeypatch):
    monkeypatch.delenv("KEEL_OFFLINE")
    files = _files(tmp_path)
    folder = download_files(files, tmp_path / "cache")
    assert (folder / "model.onnx").read_bytes() == b"onnx"
    assert (folder / "tokenizer.json").read_bytes() == b"{}"
    monkeypatch.setenv("KEEL_OFFLINE", "1")
    assert download_files(files, tmp_path / "cache") == folder


def test_file_download_rejects_a_bad_hash_and_keeps_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("KEEL_OFFLINE")
    files = _files(tmp_path)
    files["model.onnx"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_files(files, tmp_path / "cache")
    assert not list((tmp_path / "cache").rglob("model.onnx"))
    assert not list((tmp_path / "cache").rglob(".verified"))


def test_file_download_refuses_unpinned_files_and_offline_fetches(tmp_path):
    files = _files(tmp_path)
    with pytest.raises(FileNotFoundError, match="KEEL_OFFLINE"):
        download_files(files, tmp_path / "cache")
    files["tokenizer.json"]["sha256"] = ""
    with pytest.raises(ValueError, match="pinned"):
        download_files(files, tmp_path / "cache")


def test_round7_runner_with_stand_ins(tmp_path):
    small = tmp_path / "eval.toml"
    small.write_text(CONFIG.read_text().replace("personas = 200", "personas = 3"))
    ticks = iter(range(100))
    plan = load_config(CONFIG)["round7"]
    rerankers = {key: OverlapReranker() for key in plan["rerankers"]}
    results, timings = run_round7(
        small,
        first_stage=HashingEmbedder(64),
        rerankers=rerankers,
        clock=lambda: float(next(ticks)),
    )
    assert set(results["tuning"]) == set(plan["rerankers"])
    for by_n in results["tuning"].values():
        assert set(by_n) == {str(n) for n in plan["candidates_grid"]}
        for row in by_n.values():
            assert set(row) == {*(str(w) for w in plan["weight_grid"]), "only"}
    chosen = results["chosen"]
    assert chosen["candidates"] in plan["candidates_grid"]
    assert chosen["weight"] in [*plan["weight_grid"], None]
    if results["pass_rule"]["passed"] is None:
        assert results["splits"] == {}
    else:
        assert set(results["splits"]) == {"holdout5", "holdout5_direct", "holdout5_indirect"}
        assert set(results["splits"]["holdout5"]) == {plan["must_beat"], plan["candidate"]}
    assert set(timings) == set(plan["rerankers"])
    assert "Round 7: cross-encoder reranking" in to_markdown(results, timings)


@pytest.mark.skipif(not os.environ.get("KEEL_TEST_MODEL_DIR"), reason="needs downloaded models")
def test_real_rerankers_are_deterministic_and_semantic(monkeypatch):
    pytest.importorskip("onnxruntime")
    monkeypatch.setenv("KEEL_MODEL_DIR", os.environ["KEEL_TEST_MODEL_DIR"])
    plan = load_config(CONFIG)["round7"]
    first = load_rerankers(plan)
    second = load_rerankers(plan)
    query = "How many people live in Berlin?"
    docs = ["Berlin has a population of 3.5 million people.", "The cat sat on the mat."]
    for key in plan["rerankers"]:
        a, b = first[key].score(query, docs), second[key].score(query, docs)
        assert (a == b).all()
        assert a[0] > a[1] + 3


def test_shipped_reranker_matches_the_benchmark_results():
    """The agent ships what rounds 7 and 8 chose, and nothing else."""
    metrics = ROOT / "reports" / "metrics"
    round7 = json.loads((metrics / "round7.json").read_text())
    round8 = json.loads((metrics / "round8.json").read_text())
    config = load_config(CONFIG)
    plan7, plan8 = config["round7"], config["round8"]
    assert round7["pass_rule"]["passed"]
    chosen = round7["chosen"]
    assert chosen["candidates"] == RERANK_CANDIDATES == plan8["candidates"]
    assert chosen["weight"] == RERANK_WEIGHT == plan8["rerank_weight"]
    # Round 8 started from round 7's model, then chose which precision ships.
    assert plan8["rerankers"]["fp32"]["files"] == plan7["rerankers"][chosen["reranker"]]["files"]
    precision = round8["chosen"]["reranker"] if round8["pass_rule"]["passed"] else "fp32"
    spec = plan8["rerankers"][precision]
    assert {"name": spec["name"], "files": spec["files"]} == DEFAULT_RERANKER
    # Round 8 kept the encoder at full precision; tests/test_transformer.py checks which
    # encoder ships.
    encoder = round8["chosen"]["encoder"] if round8["pass_rule"]["passed"] else "fp32"
    assert encoder == "fp32"


def test_offline_agent_runs_without_a_reranker():
    assert default_reranker() is None
    assert isinstance(default_retriever(), KeelMemory)


def test_recall_with_a_reranked_retriever_skips_constraints(toolbox):
    for text, kind, key in (
        ("Allergic to peanuts.", "constraint", "allergy"),
        ("Moved to Lisbon.", "fact", "home_city"),
    ):
        assert not toolbox.run("t", "remember", {"text": text, "kind": kind, "key": key}).is_error
    toolbox.context.retriever = RerankedMemory(
        base=KeelMemory(core_max=2), reranker=OverlapReranker(), candidates=5, rerank_weight=1.0
    )
    found = json.loads(toolbox.run("t", "recall", {"query": "Lisbon", "limit": 1}).output)
    assert [m["text"] for m in found] == ["Moved to Lisbon."]
    assert toolbox.context.retriever.base.core_max == 2  # the agent's own copy is unchanged


def test_round8_runner_with_stand_ins(tmp_path):
    small = tmp_path / "eval.toml"
    small.write_text(CONFIG.read_text().replace("personas = 200", "personas = 3"))
    plan = load_config(CONFIG)["round8"]
    ticks = iter(range(100))
    sizes = {
        "encoder:fp32": 436.0,
        "encoder:int8": 110.0,
        "reranker:fp32": 91.0,
        "reranker:int8": 23.0,
    }
    results, timings = run_round8(
        small,
        encoders={"fp32": HashingEmbedder(64), "int8": HashingEmbedder(48)},
        rerankers={"fp32": OverlapReranker(), "int8": OverlapReranker()},
        sizes=sizes,
        clock=lambda: float(next(ticks)),
    )
    assert set(results["tuning"]) == {
        "encoder int8, reranker fp32",
        "encoder fp32, reranker int8",
        "encoder int8, reranker int8",
    }
    assert set(results["tuning"]["encoder int8, reranker int8"]) == {
        str(w) for w in plan["int8_encoder_weights"]
    }
    # The two rerankers are identical stand-ins, so the int8 reranker ties the current one
    # on every set and must qualify; the smallest qualifying variant is chosen.
    assert "encoder fp32, reranker int8" in results["eligible"]
    assert results["chosen"] is not None
    assert results["pass_rule"]["passed"] is not None
    assert set(results["splits"]) == {"holdout6", "holdout6_direct", "holdout6_indirect"}
    assert set(timings) == {"encoder:fp32", "encoder:int8", "reranker:fp32", "reranker:int8"}
    assert "Round 8: quantized models" in round8_markdown(results, timings)


def test_round9_runner_with_stand_ins(tmp_path):
    small = tmp_path / "eval.toml"
    small.write_text(CONFIG.read_text().replace("personas = 200", "personas = 3"))
    plan = load_config(CONFIG)["round9"]
    ticks = iter(range(100))
    encoders = {key: HashingEmbedder(32 + 16 * i) for i, key in enumerate(plan["encoders"])}
    # The current encoder is also offered as a stand-in "smaller" one, so one must qualify.
    encoders["minilm"] = encoders[plan["current"]["encoder"]]
    sizes = {key: 100.0 + i for i, key in enumerate(plan["encoders"])}
    results, timings = run_round9(
        small,
        encoders=encoders,
        reranker=OverlapReranker(),
        sizes=sizes,
        clock=lambda: float(next(ticks)),
    )
    smaller = set(plan["encoders"]) - {plan["current"]["encoder"]}
    assert set(results["tuning"]) == smaller
    for row in results["tuning"].values():
        assert set(row) == {str(w) for w in plan["weight_grid"]}
    assert "minilm" in results["eligible"]
    assert results["chosen"] is not None
    assert results["pass_rule"]["passed"] is not None
    assert set(results["splits"]) == {"holdout7", "holdout7_direct", "holdout7_indirect"}
    assert set(timings) == set(plan["encoders"])
    assert "Round 9: smaller encoders" in round9_markdown(results, timings)


def test_round10_reuses_round9_with_its_own_grid(tmp_path):
    small = tmp_path / "eval.toml"
    small.write_text(CONFIG.read_text().replace("personas = 200", "personas = 3"))
    plan = load_config(CONFIG)["round10"]
    ticks = iter(range(100))
    encoders = {key: HashingEmbedder(32 + 16 * i) for i, key in enumerate(plan["encoders"])}
    sizes = {key: 100.0 + i for i, key in enumerate(plan["encoders"])}
    results, timings = run_round9(
        small,
        encoders=encoders,
        reranker=OverlapReranker(),
        sizes=sizes,
        clock=lambda: float(next(ticks)),
        section="round10",
    )
    assert set(results["tuning"]) == {"minilm", "minilm_l12"}
    for row in results["tuning"].values():
        assert set(row) == {str(w) for w in plan["weight_grid"]}
    assert all(w < 4.5 for w in results["best_weight"].values())
    assert "Round 10: MiniLM below weight 4" in round9_markdown(results, timings, section="round10")


class RecordingEmbedder(HashingEmbedder):
    def __init__(self, query_prefix: str = "", doc_prefix: str = "") -> None:
        super().__init__(32)
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix
        self.seen: list[str] = []

    def embed(self, texts):  # type: ignore[no-untyped-def]
        self.seen.extend(texts)
        return super().embed(texts)


def test_prefixes_are_added_to_questions_and_memories(store, clock):
    memories = _memories(store, clock)
    plain, prefixed = RecordingEmbedder(), RecordingEmbedder("query: ", "passage: ")
    for embedder in (plain, prefixed):
        KeelMemory(embedder=embedder, embedding_weight=1.0).retrieve(
            memories, "where do I live", clock.now(), 3
        )
    assert plain.seen[0] == "where do I live"
    assert not any(t.startswith(("query: ", "passage: ")) for t in plain.seen)
    assert prefixed.seen[0] == "query: where do I live"
    assert all(t.startswith("passage: ") for t in prefixed.seen[1:])


def test_round11_runner_with_stand_ins(tmp_path):
    small = tmp_path / "eval.toml"
    small.write_text(CONFIG.read_text().replace("personas = 200", "personas = 3"))
    plan = load_config(CONFIG)["round11"]
    ticks = iter(range(100))
    encoders = {key: HashingEmbedder(32 + 16 * i) for i, key in enumerate(plan["encoders"])}
    sizes = dict.fromkeys(plan["encoders"], 436.0)
    results, timings = run_round11(
        small,
        encoders=encoders,
        reranker=OverlapReranker(),
        sizes=sizes,
        clock=lambda: float(next(ticks)),
    )
    others = set(plan["encoders"]) - {plan["current"]["encoder"]}
    assert set(results["tuning"]) == others
    assert results["chosen"]["encoder"] in others
    tested = results["pass_rule"]["passed"] is not None
    assert bool(results["splits"]) == tested
    assert set(timings) == set(plan["encoders"])
    assert "Round 11: other encoders" in round11_markdown(results, timings)


def test_round12_reuses_round11_with_a_one_point_bar(tmp_path):
    small = tmp_path / "eval.toml"
    small.write_text(CONFIG.read_text().replace("personas = 200", "personas = 3"))
    plan = load_config(CONFIG)["round12"]
    assert plan["min_gain"] == 0.01
    ticks = iter(range(100))
    encoders = {key: HashingEmbedder(32 + 16 * i) for i, key in enumerate(plan["encoders"])}
    sizes = dict.fromkeys(plan["encoders"], 1337.0)
    results, timings = run_round11(
        small,
        encoders=encoders,
        reranker=OverlapReranker(),
        sizes=sizes,
        clock=lambda: float(next(ticks)),
        section="round12",
    )
    others = set(plan["encoders"]) - {plan["current"]["encoder"]}
    assert set(results["tuning"]) == others
    assert all(
        set(row) == {str(w) for w in plan["weight_grid"]} for row in results["tuning"].values()
    )
    for key, row in results["tuning"].items():
        top = max(r["mean"] for r in row.values())
        tied = [float(w) for w, r in row.items() if r["mean"] > top - plan["tie_margin"]]
        assert results["best_weight"][key] == min(tied)
    rule = results["pass_rule"]
    assert rule["split"] == "holdout8"
    assert rule["min_gain"] == 0.01
    if rule["passed"] is not None:
        assert set(results["splits"]) == {"holdout8", "holdout8_direct", "holdout8_indirect"}
        low = results["comparisons"]["holdout8"][plan["must_beat"]]["low"]
        assert rule["passed"] == (low >= 0.01)
    assert "Round 12: larger encoders" in round11_markdown(results, timings, section="round12")


def test_round13_reuses_round11_with_the_tie_rule_set_in_advance(tmp_path):
    small = tmp_path / "eval.toml"
    small.write_text(CONFIG.read_text().replace("personas = 200", "personas = 3"))
    config = load_config(CONFIG)
    plan = config["round13"]
    assert "min_gain" not in plan
    assert plan["tie_margin"] == config["round12"]["tie_margin"]
    # The baseline is what round 12 shipped.
    assert plan["current"] == {"encoder": "e5_large_v2", "weight": 64.0}
    assert plan["encoders"]["e5_large_v2"] == config["round12"]["encoders"]["e5_large"]
    ticks = iter(range(100))
    encoders = {key: HashingEmbedder(32 + 16 * i) for i, key in enumerate(plan["encoders"])}
    results, timings = run_round11(
        small,
        encoders=encoders,
        reranker=OverlapReranker(),
        sizes=dict.fromkeys(plan["encoders"], 1337.0),
        clock=lambda: float(next(ticks)),
        section="round13",
    )
    assert set(results["tuning"]) == set(plan["encoders"]) - {"e5_large_v2"}
    rule = results["pass_rule"]
    assert rule["split"] == "holdout9"
    if rule["passed"] is not None:
        low = results["comparisons"]["holdout9"][plan["must_beat"]]["low"]
        assert rule["passed"] == (low > 0)
    assert "Round 13: other encoders" in round11_markdown(results, timings, section="round13")


def test_round14_tests_on_the_set_round13_left_unseen(tmp_path):
    small = tmp_path / "eval.toml"
    small.write_text(CONFIG.read_text().replace("personas = 200", "personas = 3"))
    config = load_config(CONFIG)
    plan = config["round14"]
    assert plan["split"] == config["round13"]["split"] == "holdout9"
    assert plan["min_gain"] == 0.01
    assert plan["current"] == config["round13"]["current"]
    assert all(
        "model.onnx_data" in spec["files"]
        for spec in plan["encoders"].values()
        if spec is not plan["encoders"]["e5_large_v2"]
    )
    ticks = iter(range(100))
    encoders = {key: HashingEmbedder(32 + 16 * i) for i, key in enumerate(plan["encoders"])}
    results, timings = run_round11(
        small,
        encoders=encoders,
        reranker=OverlapReranker(),
        sizes=dict.fromkeys(plan["encoders"], 2236.0),
        clock=lambda: float(next(ticks)),
        section="round14",
    )
    assert set(results["tuning"]) == set(plan["encoders"]) - {"e5_large_v2"}
    rule = results["pass_rule"]
    if rule["passed"] is not None:
        low = results["comparisons"]["holdout9"][plan["must_beat"]]["low"]
        assert rule["passed"] == (low >= 0.01)
    assert "Round 14: encoders of about 2.2 GB" in round11_markdown(
        results, timings, section="round14"
    )


def test_round15_keeps_holdout9_and_the_round11_rule(tmp_path):
    small = tmp_path / "eval.toml"
    small.write_text(CONFIG.read_text().replace("personas = 200", "personas = 3"))
    config = load_config(CONFIG)
    plan = config["round15"]
    assert plan["split"] == config["round14"]["split"] == "holdout9"
    assert "min_gain" not in plan
    assert plan["current"] == config["round14"]["current"]
    ticks = iter(range(100))
    encoders = {key: HashingEmbedder(32 + 16 * i) for i, key in enumerate(plan["encoders"])}
    results, timings = run_round11(
        small,
        encoders=encoders,
        reranker=OverlapReranker(),
        sizes=dict.fromkeys(plan["encoders"], 596.0),
        clock=lambda: float(next(ticks)),
        section="round15",
    )
    assert set(results["tuning"]) == set(plan["encoders"]) - {"e5_large_v2"}
    rule = results["pass_rule"]
    if rule["passed"] is not None:
        low = results["comparisons"]["holdout9"][plan["must_beat"]]["low"]
        assert rule["passed"] == (low > 0)
    assert "Round 15: newer encoder architectures" in round11_markdown(
        results, timings, section="round15"
    )


def test_round16_tries_the_int8_model_if_it_is_within_half_a_point(tmp_path):
    small = tmp_path / "eval.toml"
    small.write_text(CONFIG.read_text().replace("personas = 200", "personas = 3"))
    config = load_config(CONFIG)
    plan = config["round16"]
    assert plan["split"] == config["round15"]["split"] == "holdout9"
    assert plan["current"] == config["round15"]["current"]
    assert plan["encoders"]["e5_large_v2"] == config["round15"]["encoders"]["e5_large_v2"]
    assert plan["min_gain"] == -0.01 and plan["max_tuning_drop"] == 0.005
    # The same stand-in for both: its best weight is at least as good on tuning as the
    # shipped weight (64 is in the grid), so it always reaches the held-out set.
    same = HashingEmbedder(64)
    ticks = iter(range(100))
    results, timings = run_round11(
        small,
        encoders=dict.fromkeys(plan["encoders"], same),
        reranker=OverlapReranker(),
        sizes={"e5_large_v2": 1336.9, "e5_large_v2_int8": 337.0},
        clock=lambda: float(next(ticks)),
        section="round16",
    )
    assert set(results["tuning"]) == {"e5_large_v2_int8"}
    rule = results["pass_rule"]
    assert rule["max_tuning_drop"] == 0.005
    assert set(results["splits"]) == {"holdout9", "holdout9_direct", "holdout9_indirect"}
    low = results["comparisons"]["holdout9"][plan["must_beat"]]["low"]
    assert rule["passed"] == (low >= -0.01)
    assert "Round 16: an int8 e5-large-v2" in round11_markdown(results, timings, section="round16")


def test_round17_retunes_the_reranker_on_holdout10(tmp_path):
    small = tmp_path / "eval.toml"
    small.write_text(CONFIG.read_text().replace("personas = 200", "personas = 3"))
    config = load_config(CONFIG)
    plan = config["round17"]
    assert plan["split"] == "holdout10"
    assert plan["current"] == {"candidates": RERANK_CANDIDATES, "weight": RERANK_WEIGHT}
    assert plan["encoder_weight"] == TRANSFORMER_WEIGHT
    assert {"name": plan["encoder"]["name"], "files": plan["encoder"]["files"]} == {
        "name": DEFAULT_ENCODER["name"],
        "files": DEFAULT_ENCODER["files"],
    }
    assert plan["reranker"] == DEFAULT_RERANKER
    ticks = iter(range(100))
    results, timings = run_round17(
        small,
        encoder=HashingEmbedder(64),
        reranker=OverlapReranker(),
        clock=lambda: float(next(ticks)),
    )
    grid = {f"n={n} w={w}" for n in plan["candidates_grid"] for w in plan["weight_grid"]}
    assert set(results["tuning"]) == grid
    chosen = results["chosen"]
    rule = results["pass_rule"]
    if chosen == results["current"]:
        assert rule["passed"] is None and not results["splits"]
    else:
        assert set(results["splits"]) == {"holdout10", "holdout10_direct", "holdout10_indirect"}
        low = results["comparisons"]["holdout10"][plan["must_beat"]]["low"]
        bar = plan["min_gain_costlier"] if chosen["candidates"] > 20 else plan["min_gain"]
        assert rule["passed"] == (low >= bar if bar else low > 0)
    assert set(timings) == {str(n) for n in plan["candidates_grid"]}
    assert "Round 17" in round17_markdown(results, timings)
