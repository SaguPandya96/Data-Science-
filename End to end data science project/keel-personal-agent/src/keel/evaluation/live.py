"""Round 3: end-to-end check with a real model (docs/ANALYSIS_PLAN.md).

For a sample of benchmark personas, each arm picks memories for each held-out question and
the model answers from those memories alone. An answer is correct when it states the
current value and no earlier one.

This costs money (one API call per question per arm), so it never runs in CI and the CLI
asks for ``--yes``. Every answer is appended to a JSONL cache as it arrives, so an
interrupted run resumes where it stopped instead of paying for the same calls again.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from keel.evaluation.benchmark import bootstrap_mean, load_config, paired_difference
from keel.evaluation.scenarios import SLOTS, Persona, Probe, build_personas
from keel.memory.embeddings import Embedder, WordLlamaEmbedder, default_transformer
from keel.memory.retrieval import (
    EMBEDDING_WEIGHT,
    TRANSFORMER_WEIGHT,
    EmbeddingMemory,
    FullMemory,
    KeelMemory,
    LexicalMemory,
    RecentMemory,
    Retriever,
    render_context,
)
from keel.memory.text import estimate_tokens
from keel.model import Model

ANSWER_SYSTEM = (
    "You answer questions about the user from their saved memories. Use only the memories "
    "given. If they don't contain the answer, say you don't know. Answer in one sentence."
)

_FILLER = {"a", "an", "the", "named", "and", "month"}
STYLES = ("direct", "indirect")


def mentions(answer: str, value: str) -> bool:
    """Does the answer state this value? Names and numbers must all appear; other values
    must appear as a phrase (ignoring articles)."""
    answer_l = answer.lower()
    anchors = re.findall(r"\b(?:[A-Z][\w'-]+|\d[\d:$,.]*)", value)
    anchors = [a for a in anchors if a.lower() not in _FILLER]
    if anchors:
        return all(a.lower() in answer_l for a in anchors)
    phrase = " ".join(w for w in value.lower().split() if w not in _FILLER)
    return phrase in answer_l


def grade(answer: str, probe: Probe) -> dict[str, bool]:
    current = mentions(answer, probe.expected)
    earlier = any(mentions(answer, v) for v in probe.stale_values)
    return {"correct": current and not earlier, "stale": earlier and not current}


def live_arms(core_max: int, wordllama: Embedder, transformer: Embedder) -> list[Retriever]:
    """Every method the live check knows; configs/eval.toml picks which ones run."""
    return [
        RecentMemory(),
        LexicalMemory(),
        EmbeddingMemory(wordllama),
        EmbeddingMemory(transformer, name="transformer"),
        KeelMemory(core_max=core_max),
        KeelMemory(
            core_max=core_max,
            embedder=wordllama,
            embedding_weight=EMBEDDING_WEIGHT,
            name="keel+embed",
        ),
        KeelMemory(
            core_max=core_max,
            embedder=transformer,
            embedding_weight=TRANSFORMER_WEIGHT,
            name="keel+transformer",
        ),
        FullMemory(),
    ]


def planned_calls(personas: int, arms: Sequence[str]) -> int:
    """API calls a run makes before any cache hits: one per question per arm."""
    return personas * len(SLOTS) * len(STYLES) * len(arms)


class AnswerCache:
    """Answers already paid for, keyed by persona, arm and question. Thread-safe."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._answers: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        if path is not None and path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._answers[row["key"]] = row

    @staticmethod
    def key(persona_id: int, arm: str, question: str, model: str) -> str:
        return f"{model}|{persona_id}|{arm}|{question}"

    def get(self, key: str) -> dict[str, Any] | None:
        return self._answers.get(key)

    def put(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._answers[row["key"]] = row
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")

    def __len__(self) -> int:
        return len(self._answers)


def run_live(
    config_path: str | Path,
    model: Model,
    *,
    model_name: str,
    personas: int | None = None,
    arms: Sequence[str] | None = None,
    embedder: Embedder | None = None,
    transformer: Embedder | None = None,
    cache_path: Path | None = None,
    workers: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    plan = config["live"]
    personas = personas or plan["personas"]
    arm_names = list(arms or plan["arms"])
    k, core_max = config["retrieval"]["k"], config["retrieval"]["core_max"]
    seed = config["benchmark"]["seed"]
    embedder = embedder or WordLlamaEmbedder()
    if transformer is None and any(a in ("transformer", "keel+transformer") for a in arm_names):
        transformer = default_transformer()
        if transformer is None:
            raise RuntimeError(
                "the transformer arms need the MiniLM encoder: install the 'transformer' "
                "extra and allow one download (unset KEEL_OFFLINE)"
            )
    chosen = [
        a for a in live_arms(core_max, embedder, transformer or embedder) if a.name in arm_names
    ]
    missing = set(arm_names) - {a.name for a in chosen}
    if missing:
        raise ValueError(f"unknown arms: {', '.join(sorted(missing))}")
    population = build_personas(dict(config["benchmark"], personas=personas))
    cache = AnswerCache(cache_path)

    # Build every job first: retrieval is cheap and deterministic, the API call is not.
    jobs: list[tuple[Persona, Retriever, str, Probe, str]] = []
    for persona in population:
        memories = persona.memories
        for arm in chosen:
            for style in STYLES:
                for probe in persona.probes[f"holdout_{style}"]:
                    picked = arm.retrieve(memories, probe.question, persona.now, k)
                    prompt = (
                        f"Memories:\n{render_context(picked)}\n\nToday is "
                        f"{persona.now:%Y-%m-%d}.\nQuestion: {probe.question}"
                    )
                    jobs.append((persona, arm, style, probe, prompt))

    done = 0
    lock = threading.Lock()

    def answer(job: tuple[Persona, Retriever, str, Probe, str]) -> dict[str, Any]:
        nonlocal done
        persona, arm, style, probe, prompt = job
        key = AnswerCache.key(persona.persona_id, arm.name, probe.question, model_name)
        row = cache.get(key)
        if row is None:
            reply = model.reply(ANSWER_SYSTEM, [{"role": "user", "content": prompt}], [])
            text = " ".join(b["text"] for b in reply.content if b.get("type") == "text")
            row = {
                "key": key,
                "persona": persona.persona_id,
                "arm": arm.name,
                "style": style,
                "detail": probe.slot,
                "question": probe.question,
                "expected": probe.expected,
                "answer": text.strip(),
                "prompt_tokens": estimate_tokens(prompt),
                "stop_reason": reply.stop_reason,
            }
            cache.put(row)
        with lock:
            done += 1
            if progress is not None:
                progress(done, len(jobs))
        return {**row, **grade(row["answer"], probe), "style": style}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(answer, jobs))

    return summarize_live(rows, population, [a.name for a in chosen], plan, seed, k)


def summarize_live(
    rows: list[dict[str, Any]],
    population: Sequence[Persona],
    arm_names: Sequence[str],
    plan: dict[str, Any],
    seed: int,
    k: int,
) -> dict[str, Any]:
    ids = [p.persona_id for p in population]

    def per_persona(arm: str, metric: str, style: str | None) -> np.ndarray:
        values = []
        for pid in ids:
            chosen = [
                r[metric]
                for r in rows
                if r["arm"] == arm
                and r["persona"] == pid
                and (style is None or r["style"] == style)
            ]
            values.append(float(np.mean(chosen)) if chosen else np.nan)
        return np.array(values)

    subsets = {"all": None, "direct": "direct", "indirect": "indirect"}
    accuracy = {
        arm: {
            name: {
                "correct": bootstrap_mean(per_persona(arm, "correct", style), 2000, 0.95, seed),
                "stale": bootstrap_mean(per_persona(arm, "stale", style), 2000, 0.95, seed),
            }
            for name, style in subsets.items()
        }
        for arm in arm_names
    }
    tokens = {
        arm: float(np.mean([r["prompt_tokens"] for r in rows if r["arm"] == arm]))
        for arm in arm_names
    }

    candidate = plan["candidate"]
    comparisons: dict[str, Any] = {}
    if candidate in arm_names:
        for other in arm_names:
            if other == candidate:
                continue
            comparisons[other] = {
                name: paired_difference(
                    per_persona(candidate, "correct", style),
                    per_persona(other, "correct", style),
                    2000,
                    0.95,
                    seed,
                )
                for name, style in subsets.items()
            }

    rule = comparisons.get(plan["must_beat"], {}).get("all")
    # A fixed, spread-out sample for checking the grader by eye: every 97th answer.
    ordered = sorted(rows, key=lambda r: r["key"])
    examples = [
        {
            key: r[key]
            for key in ("arm", "style", "question", "expected", "answer", "correct", "stale")
        }
        for r in ordered[::97]
    ]
    return {
        "personas": len(ids),
        "questions_per_arm": len(ids) * len(STYLES) * len(SLOTS),
        "k": k,
        "accuracy": accuracy,
        "prompt_tokens": tokens,
        "comparisons": comparisons,
        "pass_rule": {
            "candidate": candidate,
            "must_beat": plan["must_beat"],
            "passed": None if rule is None else rule["low"] > 0,
        },
        "examples": examples,
    }


def to_markdown(results: dict[str, Any], model_name: str) -> str:
    def pct(stat: dict) -> str:
        return f"{100 * stat['mean']:.1f}% ({100 * stat['low']:.1f} to {100 * stat['high']:.1f})"

    def pts(stat: dict) -> str:
        return (
            f"{100 * stat['mean']:+.1f} pts ({100 * stat['low']:+.1f} to {100 * stat['high']:+.1f})"
        )

    rule = results["pass_rule"]
    verdict = {True: "PASSED", False: "FAILED", None: "NOT RUN (arm missing)"}[rule["passed"]]
    lines = [
        "# Round 3: end-to-end check",
        "",
        f"Model `{model_name}`. {results['personas']} personas, "
        f"{results['questions_per_arm']} questions per arm, k = {results['k']}. "
        "Intervals are 95% persona bootstrap.",
        "",
        f"**Pass rule (`{rule['candidate']}` beats `{rule['must_beat']}` on answer "
        f"accuracy): {verdict}**",
        "",
        "| Arm | Correct | Direct | Indirect | Stale answers | Prompt tokens |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm, stats in results["accuracy"].items():
        lines.append(
            f"| `{arm}` | {pct(stats['all']['correct'])} | "
            f"{100 * stats['direct']['correct']['mean']:.1f}% | "
            f"{100 * stats['indirect']['correct']['mean']:.1f}% | "
            f"{100 * stats['all']['stale']['mean']:.1f}% | "
            f"{results['prompt_tokens'][arm]:.0f} |"
        )
    if results["comparisons"]:
        lines += ["", f"`{rule['candidate']}` minus, on correct answers:", ""]
        for other, diff in results["comparisons"].items():
            lines.append(
                f"- `{other}`: {pts(diff['all'])} (direct {pts(diff['direct'])}, "
                f"indirect {pts(diff['indirect'])})"
            )
    lines += [
        "",
        "## Sample of graded answers",
        "",
        "| Arm | Question | Expected | Answer | Correct |",
        "| --- | --- | --- | --- | --- |",
    ]
    for ex in results["examples"]:
        answer = ex["answer"].replace("|", "/").replace("\n", " ")
        lines.append(
            f"| `{ex['arm']}` | {ex['question']} | {ex['expected']} | {answer} | "
            f"{'yes' if ex['correct'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"
