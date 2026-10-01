from __future__ import annotations

import threading
from pathlib import Path

import pytest

from keel.cli import main
from keel.evaluation.benchmark import load_config
from keel.evaluation.live import (
    AnswerCache,
    grade,
    live_arms,
    planned_calls,
    run_live,
    to_markdown,
)
from keel.evaluation.scenarios import Probe
from keel.memory.embeddings import HashingEmbedder
from keel.memory.rerank import OverlapReranker
from keel.model import text

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "eval.toml"


class EchoModel:
    """Repeats every memory line it is shown, and counts its calls."""

    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def reply(self, system, messages, tools):  # type: ignore[no-untyped-def]
        with self._lock:
            self.calls += 1
        lines = [line for line in messages[0]["content"].splitlines() if line.startswith("- #")]
        return text(" ".join(lines) or "I don't know.")


def _run(model: EchoModel, cache: Path, arms: list[str]) -> dict:
    return run_live(
        CONFIG,
        model,
        model_name="echo",
        personas=2,
        arms=arms,
        embedder=HashingEmbedder(),
        transformer=HashingEmbedder(64),
        reranker=OverlapReranker(),
        cache_path=cache,
        workers=4,
    )


def test_live_arms_match_the_plan():
    names = [
        a.name for a in live_arms(2, HashingEmbedder(), HashingEmbedder(64), OverlapReranker())
    ]
    plan = load_config(CONFIG)["live"]
    assert set(plan["arms"]) <= set(names)
    assert plan["candidate"] == "keel+rerank"  # the retriever the agent ships


def test_grading_separates_current_stale_and_both():
    probe = Probe("home_city", "q", 2, (1,), "Austin", True, ("Denver",))
    assert grade("You live in Austin.", probe) == {"correct": True, "stale": False}
    assert grade("You live in Denver.", probe) == {"correct": False, "stale": True}
    assert grade("Denver or Austin.", probe) == {"correct": False, "stale": False}
    assert grade("I don't know.", probe) == {"correct": False, "stale": False}


def test_run_scores_arms_and_resumes_from_cache(tmp_path):
    cache = tmp_path / "answers.jsonl"
    model = EchoModel()
    results = _run(model, cache, ["recent", "keel+rerank", "transformer"])
    assert model.calls == planned_calls(2, ["recent", "keel+rerank", "transformer"]) == 192
    assert set(results["accuracy"]) == {"recent", "keel+rerank", "transformer"}
    keel = results["accuracy"]["keel+rerank"]["all"]["correct"]["mean"]
    assert keel > results["accuracy"]["recent"]["all"]["correct"]["mean"]
    assert set(results["comparisons"]) == {"recent", "transformer"}
    assert results["pass_rule"]["passed"] in (True, False)
    assert "Round 3: end-to-end check" in to_markdown(results, "echo")

    # A second run pays for nothing and reproduces the same numbers.
    again = EchoModel()
    assert _run(again, cache, ["recent", "keel+rerank", "transformer"]) == results
    assert again.calls == 0
    assert len(AnswerCache(cache)) == 192


def test_rule_is_not_run_without_its_arms(tmp_path):
    results = _run(EchoModel(), tmp_path / "a.jsonl", ["recent"])
    assert results["pass_rule"]["passed"] is None
    assert "NOT RUN" in to_markdown(results, "echo")


def test_unknown_arm_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown arms"):
        _run(EchoModel(), tmp_path / "a.jsonl", ["recent", "oracle"])


def test_cli_states_the_spend_and_needs_confirmation(capsys):
    assert main(["eval-live", "--personas", "30"]) == 1
    out = capsys.readouterr().out
    assert "Up to 5,760 API calls to claude-opus-5-5" in out
    assert "--yes" in out


def test_transformer_arms_need_the_encoder(tmp_path):
    with pytest.raises(RuntimeError, match="default encoder"):
        run_live(
            CONFIG,
            EchoModel(),
            model_name="echo",
            personas=1,
            arms=["keel+transformer"],
            embedder=HashingEmbedder(),
            cache_path=tmp_path / "a.jsonl",
        )


def test_rerank_arm_needs_the_reranker(tmp_path):
    with pytest.raises(RuntimeError, match="default reranker"):
        run_live(
            CONFIG,
            EchoModel(),
            model_name="echo",
            personas=1,
            arms=["keel+rerank"],
            embedder=HashingEmbedder(),
            transformer=HashingEmbedder(64),
            cache_path=tmp_path / "a.jsonl",
        )
