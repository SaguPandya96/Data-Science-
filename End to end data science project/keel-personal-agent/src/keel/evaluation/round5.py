"""Round 5: a wider weight grid for the MiniLM encoder (docs/ANALYSIS_PLAN.md)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from keel.evaluation.benchmark import load_config, paired_difference, score_arm, summarize
from keel.evaluation.scenarios import build_personas
from keel.memory.embeddings import (
    DEFAULT_ENCODER,
    Embedder,
    OnnxSentenceEmbedder,
    WordLlamaEmbedder,
    download_model,
)
from keel.memory.retrieval import EMBEDDING_WEIGHT, EmbeddingMemory, KeelMemory, Retriever

SPLITS = ("holdout3", "holdout3_direct", "holdout3_indirect")


def load_minilm() -> Embedder:
    folder = download_model(DEFAULT_ENCODER["url"], DEFAULT_ENCODER["sha256"])
    return OnnxSentenceEmbedder(
        folder, pooling=DEFAULT_ENCODER["pooling"], name=DEFAULT_ENCODER["name"]
    )


def run_round5(
    config_path: str | Path,
    encoder: Embedder | None = None,
    wordllama: Embedder | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    bench, retrieval, analysis = config["benchmark"], config["retrieval"], config["analysis"]
    plan = config["round5"]
    seed, k, core_max = bench["seed"], retrieval["k"], retrieval["core_max"]
    resamples, confidence = analysis["bootstrap_resamples"], analysis["confidence"]
    encoder = encoder or load_minilm()
    wordllama = wordllama or WordLlamaEmbedder()
    personas = build_personas(bench)

    def keel(weight: float, name: str) -> KeelMemory:
        return KeelMemory(core_max=core_max, embedder=encoder, embedding_weight=weight, name=name)

    # 1. Choose the weight on question sets that have already been seen.
    tuning: dict[str, dict[str, float]] = {}
    for weight in plan["weight_grid"]:
        arm = keel(weight, f"w={weight}")
        per_split = {
            split: float(score_arm(arm, personas, split, k).all["clean_hit"].mean())
            for split in plan["tuning_splits"]
        }
        per_split["mean"] = float(np.mean(list(per_split.values())))
        tuning[str(weight)] = per_split
    best = max(plan["weight_grid"], key=lambda w: (tuning[str(w)]["mean"], -w))
    current = plan["current_weight"]

    arms: list[Retriever] = [
        EmbeddingMemory(encoder, name="transformer"),
        KeelMemory(
            core_max=core_max,
            embedder=wordllama,
            embedding_weight=EMBEDDING_WEIGHT,
            name="keel+embed",
        ),
        keel(current, plan["must_beat"]),
    ]
    if best != current:
        arms.append(keel(best, plan["candidate"]))

    # Every grid weight on the new held-out set too, reported but not used for the rule.
    grid_on_test: dict[str, dict[str, float]] = {}
    for weight in plan["weight_grid"]:
        arm = keel(weight, f"w={weight}")
        grid_on_test[str(weight)] = {
            split: float(score_arm(arm, personas, split, k).all["clean_hit"].mean())
            for split in SPLITS
        }

    # 2. Score the arms once on the new held-out set.
    results: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for split in SPLITS:
        scores = [score_arm(a, personas, split, k) for a in arms]
        by_name = {s.name: s for s in scores}
        results[split] = summarize(scores, analysis, seed)
        if best != current:
            candidate = by_name[plan["candidate"]].all["clean_hit"]
            comparisons[split] = {
                other: paired_difference(
                    candidate, by_name[other].all["clean_hit"], resamples, confidence, seed
                )
                for other in [plan["must_beat"], *plan["also_compare"]]
            }

    if best == current:
        verdict: bool | None = None
        outcome = "current weight confirmed"
    else:
        verdict = comparisons[plan["split"]][plan["must_beat"]]["low"] > 0
        outcome = f"w = {best} {'adopted' if verdict else 'not adopted'}"
    return {
        "encoder": encoder.name,
        "k": k,
        "personas": len(personas),
        "tuning_splits": plan["tuning_splits"],
        "tuning": tuning,
        "chosen_weight": best,
        "current_weight": current,
        "grid_on_holdout3": grid_on_test,
        "splits": results,
        "comparisons": comparisons,
        "pass_rule": {
            "candidate": plan["candidate"],
            "must_beat": plan["must_beat"],
            "split": plan["split"],
            "passed": verdict,
            "outcome": outcome,
        },
    }


def to_markdown(results: dict[str, Any]) -> str:
    def pct(stat: dict) -> str:
        return f"{100 * stat['mean']:.1f}% ({100 * stat['low']:.1f} to {100 * stat['high']:.1f})"

    def pts(stat: dict) -> str:
        return (
            f"{100 * stat['mean']:+.1f} pts ({100 * stat['low']:+.1f} to {100 * stat['high']:+.1f})"
        )

    rule = results["pass_rule"]
    splits = results["tuning_splits"]
    lines = [
        "# Round 5: a wider weight grid for MiniLM",
        "",
        f"Encoder `{results['encoder']}`. {results['personas']} personas, k = {results['k']}. "
        "Intervals are 95% persona bootstrap.",
        "",
        f"**Outcome: {rule['outcome']}.** Chosen weight {results['chosen_weight']}; "
        f"current weight {results['current_weight']}.",
        "",
        "## Choosing the weight (clean hit on already-seen question sets)",
        "",
        "| Weight | " + " | ".join(f"`{s}`" for s in splits) + " | Mean |",
        "| --- |" + " --- |" * (len(splits) + 1),
    ]
    for weight, row in results["tuning"].items():
        mark = " (chosen)" if float(weight) == results["chosen_weight"] else ""
        cells = " | ".join(f"{100 * row[s]:.1f}%" for s in splits)
        lines.append(f"| {weight}{mark} | {cells} | {100 * row['mean']:.1f}% |")
    lines += [
        "",
        "## Every weight on the new held-out set (reported, not used to choose)",
        "",
        "| Weight | All | Direct | Indirect |",
        "| --- | --- | --- | --- |",
    ]
    for weight, row in results["grid_on_holdout3"].items():
        lines.append(
            f"| {weight} | {100 * row['holdout3']:.1f}% | {100 * row['holdout3_direct']:.1f}% | "
            f"{100 * row['holdout3_indirect']:.1f}% |"
        )
    titles = {
        "holdout3": "New held-out set (all)",
        "holdout3_direct": "New held-out set: direct",
        "holdout3_indirect": "New held-out set: indirect",
    }
    for split, title in titles.items():
        lines += [
            "",
            f"## {title}",
            "",
            "| Arm | Clean hit | Stale shown | Allergy shown |",
            "| --- | --- | --- | --- |",
        ]
        for name, stats in results["splits"][split].items():
            s = stats["all"]
            lines.append(
                f"| `{name}` | {pct(s['clean_hit'])} | {pct(s['stale'])} | "
                f"{pct(s['allergy_shown'])} |"
            )
        if split in results["comparisons"]:
            lines += ["", f"`{rule['candidate']}` minus:", ""]
            for other, diff in results["comparisons"][split].items():
                lines.append(f"- `{other}`: {pts(diff)}")
    return "\n".join(lines) + "\n"
