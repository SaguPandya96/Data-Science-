"""Scoring retrievers on the synthetic personas, with persona-level bootstrap intervals."""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from keel.evaluation.scenarios import Persona, build_personas
from keel.memory.retrieval import (
    FullMemory,
    KeelMemory,
    LexicalMemory,
    NoMemory,
    RecentMemory,
    Retriever,
)
from keel.memory.text import estimate_tokens

METRICS = ("hit", "stale", "clean_hit", "allergy_shown", "tokens")


def load_config(path: str | Path) -> dict:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def main_arms(core_max: int) -> list[Retriever]:
    return [
        NoMemory(),
        RecentMemory(),
        LexicalMemory(),
        KeelMemory(core_max=core_max),
        FullMemory(),
    ]


def ablation_arms(core_max: int) -> list[Retriever]:
    return [
        KeelMemory(core_max=core_max),
        KeelMemory(core_max=core_max, use_supersession=False, name="keel - supersession"),
        KeelMemory(
            core_max=core_max,
            recency_weight=0.0,
            importance_weight=0.0,
            name="keel - recency & importance",
        ),
        KeelMemory(core_max=core_max, use_expansion=False, name="keel - synonyms"),
        KeelMemory(core_max=core_max, use_keys=False, name="keel - key indexing"),
        KeelMemory(core_max=core_max, use_dedupe=False, name="keel - duplicate suppression"),
        KeelMemory(core_max=core_max, use_core=False, name="keel - always-on constraints"),
    ]


def _mean(values: list[float]) -> float:
    kept = [v for v in values if not np.isnan(v)]
    return float(np.mean(kept)) if kept else np.nan


@dataclass
class ArmScores:
    """Per-persona averages: arrays of shape (personas,) for each metric and subset."""

    name: str
    all: dict[str, np.ndarray]
    updated: dict[str, np.ndarray]


def score_arm(arm: Retriever, personas: Sequence[Persona], split: str, k: int) -> ArmScores:
    rows_all: dict[str, list[float]] = {m: [] for m in METRICS}
    rows_upd: dict[str, list[float]] = {m: [] for m in METRICS}
    for persona in personas:
        memories = persona.memories
        per_all: dict[str, list[float]] = {m: [] for m in METRICS}
        per_upd: dict[str, list[float]] = {m: [] for m in METRICS}
        allergy_id = next(p.current_id for p in persona.probes[split] if p.slot == "allergy")
        for probe in persona.probes[split]:
            chosen = arm.retrieve(memories, probe.question, persona.now, k)
            ids = {m.id for m in chosen}
            hit = float(probe.current_id in ids)
            stale = float(any(i in ids for i in probe.stale_ids))
            result = {
                "hit": hit,
                "stale": stale,
                "clean_hit": hit * (1.0 - stale),
                # Is the allergy in the prompt when the question is about something else?
                "allergy_shown": np.nan if probe.slot == "allergy" else float(allergy_id in ids),
                "tokens": float(sum(estimate_tokens(m.render()) for m in chosen)),
            }
            for metric, value in result.items():
                per_all[metric].append(value)
                if probe.updated:
                    per_upd[metric].append(value)
        for metric in METRICS:
            rows_all[metric].append(_mean(per_all[metric]))
            # A persona with no changed details contributes nothing to the updated subset.
            rows_upd[metric].append(_mean(per_upd[metric]))
    return ArmScores(
        name=arm.name,
        all={m: np.array(v) for m, v in rows_all.items()},
        updated={m: np.array(v) for m, v in rows_upd.items()},
    )


def bootstrap_mean(
    values: np.ndarray, resamples: int, confidence: float, seed: int
) -> dict[str, float]:
    """Mean and percentile interval, resampling personas with replacement."""
    values = values[~np.isnan(values)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(means, alpha)),
        "high": float(np.quantile(means, 1 - alpha)),
        "n": len(values),
    }


def paired_difference(
    a: np.ndarray, b: np.ndarray, resamples: int, confidence: float, seed: int
) -> dict[str, float]:
    """a minus b on the same personas."""
    mask = ~(np.isnan(a) | np.isnan(b))
    return bootstrap_mean(a[mask] - b[mask], resamples, confidence, seed)


def summarize(scores: Sequence[ArmScores], analysis: dict, seed: int) -> dict:
    out: dict = {}
    for arm in scores:
        out[arm.name] = {
            subset: {
                metric: bootstrap_mean(
                    getattr(arm, subset)[metric],
                    analysis["bootstrap_resamples"],
                    analysis["confidence"],
                    seed,
                )
                for metric in METRICS
            }
            for subset in ("all", "updated")
        }
    return out


def run(config_path: str | Path, split: str) -> dict:
    """Every table in the README for one template split."""
    config = load_config(config_path)
    bench, retrieval, analysis = config["benchmark"], config["retrieval"], config["analysis"]
    seed = bench["seed"]
    resamples, confidence = analysis["bootstrap_resamples"], analysis["confidence"]
    k, core_max = retrieval["k"], retrieval["core_max"]

    personas = build_personas(bench)
    main = [score_arm(a, personas, split, k) for a in main_arms(core_max)]
    by_name = {s.name: s for s in main}
    primary = analysis["primary_metric"]

    comparisons = {}
    for baseline in analysis["baselines_to_beat"]:
        comparisons[baseline] = {
            subset: paired_difference(
                getattr(by_name["keel"], subset)[primary],
                getattr(by_name[baseline], subset)[primary],
                resamples,
                confidence,
                seed,
            )
            for subset in ("all", "updated")
        }
    passed = all(comparisons[b]["all"]["low"] > 0 for b in analysis["baselines_to_beat"])

    ablations = summarize(
        [score_arm(a, personas, split, k) for a in ablation_arms(core_max)], analysis, seed
    )

    noise = {}
    for level in analysis["key_noise_levels"]:
        noisy = build_personas(bench, key_noise=level)
        arms = [score_arm(a, noisy, split, k) for a in main_arms(core_max)[1:4]]
        noise[str(level)] = summarize(arms, analysis, seed)

    sweep = {}
    for k_value in analysis["k_sweep"]:
        arms = [score_arm(a, personas, split, k_value) for a in main_arms(core_max)[1:4]]
        sweep[str(k_value)] = summarize(arms, analysis, seed)

    memories_per_persona = [len(p.memories) for p in personas]
    updated_share = float(np.mean([probe.updated for p in personas for probe in p.probes[split]]))
    return {
        "split": split,
        "k": k,
        "personas": len(personas),
        "questions": sum(len(p.probes[split]) for p in personas),
        "memories_per_persona": {
            "mean": float(np.mean(memories_per_persona)),
            "min": int(np.min(memories_per_persona)),
            "max": int(np.max(memories_per_persona)),
        },
        "updated_share": updated_share,
        "arms": summarize(main, analysis, seed),
        "comparisons": comparisons,
        "pass_rule": {
            "metric": primary,
            "baselines": analysis["baselines_to_beat"],
            "passed": passed,
        },
        "ablations": ablations,
        "key_noise": noise,
        "k_sweep": sweep,
    }
