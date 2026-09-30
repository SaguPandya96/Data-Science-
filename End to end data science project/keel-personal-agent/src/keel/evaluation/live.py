"""End-to-end check with a real model: does better retrieval give better answers?

For a sample of benchmark personas, each arm picks memories for each test question and the
model answers from those memories alone. An answer counts as correct when it contains the
current value and no earlier one. This costs money (one API call per question per arm), so
it never runs in CI and needs ``--yes`` on the command line.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from keel.evaluation.benchmark import bootstrap_mean, load_config, main_arms
from keel.evaluation.scenarios import build_personas
from keel.memory.retrieval import render_context
from keel.model import Model

ANSWER_SYSTEM = (
    "You answer questions about the user from their saved memories. Use only the memories "
    "given. If they don't contain the answer, say you don't know. Answer in one sentence."
)

_FILLER = {"a", "an", "the", "named", "and", "month"}


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


def run_live(config_path: str | Path, model: Model, personas: int, arms: list[str]) -> dict:
    config = load_config(config_path)
    bench = dict(config["benchmark"], personas=personas)
    k, core_max = config["retrieval"]["k"], config["retrieval"]["core_max"]
    chosen_arms = [a for a in main_arms(core_max) if a.name in arms]
    population = build_personas(bench)

    per_arm: dict[str, list[float]] = {a.name: [] for a in chosen_arms}
    examples: list[dict[str, Any]] = []
    for persona in population:
        memories = persona.memories
        for arm in chosen_arms:
            correct = []
            for probe in persona.probes["test"]:
                picked = arm.retrieve(memories, probe.question, persona.now, k)
                prompt = (
                    f"Memories:\n{render_context(picked)}\n\nToday is "
                    f"{persona.now:%Y-%m-%d}.\nQuestion: {probe.question}"
                )
                reply = model.reply(ANSWER_SYSTEM, [{"role": "user", "content": prompt}], [])
                answer = " ".join(b["text"] for b in reply.content if b.get("type") == "text")
                current = mentions(answer, probe.expected)
                stale = any(mentions(answer, v) for v in probe.stale_values)
                correct.append(float(current and not stale))
                if len(examples) < 40:
                    examples.append(
                        {
                            "arm": arm.name,
                            "q": probe.question,
                            "expected": probe.expected,
                            "answer": answer,
                            "correct": current and not stale,
                        }
                    )
            per_arm[arm.name].append(float(np.mean(correct)))

    seed = config["benchmark"]["seed"]
    summary = {
        name: bootstrap_mean(np.array(values), 2000, 0.95, seed) for name, values in per_arm.items()
    }
    return {"personas": personas, "k": k, "accuracy": summary, "examples": examples}
