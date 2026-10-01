"""Round 6: does a larger transformer encoder beat MiniLM? (docs/ANALYSIS_PLAN.md)"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from keel.evaluation.benchmark import load_config, paired_difference, score_arm, summarize
from keel.evaluation.round4 import load_encoders
from keel.evaluation.scenarios import build_personas
from keel.memory.embeddings import Embedder
from keel.memory.retrieval import EmbeddingMemory, KeelMemory, Retriever

SPLITS = ("holdout4", "holdout4_direct", "holdout4_indirect")

# Question sets added after round 6 are left out of the warm-up below, so its batches stay
# as they were when round 6's numbers were committed. On some CPUs, how a text is padded
# within its batch changes the last bits of its vector, and that can flip a near-tie.
LATER_SPLITS = ("holdout5",)


def run_round6(
    config_path: str | Path,
    encoders: Mapping[str, Embedder] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (results, timings). Timings vary by machine, so they are kept apart."""
    config = load_config(config_path)
    bench, retrieval, analysis = config["benchmark"], config["retrieval"], config["analysis"]
    plan = config["round6"]
    seed, k, core_max = bench["seed"], retrieval["k"], retrieval["core_max"]
    resamples, confidence = analysis["bootstrap_resamples"], analysis["confidence"]
    encoders = dict(encoders or load_encoders(plan))
    order = list(plan["encoders"])  # smallest model first, so it wins ties
    personas = build_personas(bench)

    texts = sorted(
        {m.text for p in personas for m in p.memories}
        | {f"{m.key.replace('_', ' ')}. {m.text}" for p in personas for m in p.memories if m.key}
        | {
            q.question
            for p in personas
            for split, probes in p.probes.items()
            if not split.startswith(LATER_SPLITS)
            for q in probes
        }
    )
    timings = {}
    for key in order:
        start = clock()
        encoders[key].embed(texts)
        elapsed = clock() - start
        timings[key] = {
            "texts": len(texts),
            "seconds": round(elapsed, 2),
            "per_second": round(len(texts) / elapsed, 1) if elapsed else None,
        }

    def keel(key: str, weight: float, name: str) -> KeelMemory:
        return KeelMemory(
            core_max=core_max, embedder=encoders[key], embedding_weight=weight, name=name
        )

    # 1. Choose the encoder and weight together, on question sets already seen.
    tuning: dict[str, dict[str, dict[str, float]]] = {}
    for key in order:
        tuning[key] = {}
        for weight in plan["weight_grid"]:
            arm = keel(key, weight, f"{key} w={weight}")
            row = {
                split: float(score_arm(arm, personas, split, k).all["clean_hit"].mean())
                for split in plan["tuning_splits"]
            }
            row["mean"] = float(np.mean(list(row.values())))
            tuning[key][str(weight)] = row
    best_key, best_weight = max(
        ((key, w) for key in order for w in plan["weight_grid"]),
        key=lambda kw: (tuning[kw[0]][str(kw[1])]["mean"], -order.index(kw[0]), -kw[1]),
    )
    current = plan["current"]
    unchanged = best_key == current["encoder"] and best_weight == current["weight"]

    arms: list[Retriever] = [
        EmbeddingMemory(encoders[current["encoder"]], name="transformer"),
        keel(current["encoder"], current["weight"], plan["must_beat"]),
    ]
    if not unchanged:
        arms += [
            EmbeddingMemory(encoders[best_key], name="larger"),
            keel(best_key, best_weight, plan["candidate"]),
        ]

    # 2. Score once on the new held-out set.
    results: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for split in SPLITS:
        scores = [score_arm(a, personas, split, k) for a in arms]
        by_name = {s.name: s for s in scores}
        results[split] = summarize(scores, analysis, seed)
        if not unchanged:
            candidate = by_name[plan["candidate"]].all["clean_hit"]
            comparisons[split] = {
                other: paired_difference(
                    candidate, by_name[other].all["clean_hit"], resamples, confidence, seed
                )
                for other in (plan["must_beat"], "transformer", "larger")
            }

    if unchanged:
        verdict: bool | None = None
        outcome = "MiniLM at the current weight kept"
    else:
        low = comparisons[plan["split"]][plan["must_beat"]]["low"]
        verdict = low >= plan["min_gain"]
        name = plan["encoders"][best_key]["name"]
        if verdict:
            outcome = f"{name} at w = {best_weight} adopted"
        elif low > 0:
            outcome = (
                f"{name} at w = {best_weight} is better, but below the 1-point bar; MiniLM kept"
            )
        else:
            outcome = f"{name} at w = {best_weight} not better on the held-out set; MiniLM kept"
    return (
        {
            "encoders": {key: plan["encoders"][key]["name"] for key in order},
            "k": k,
            "personas": len(personas),
            "tuning_splits": plan["tuning_splits"],
            "tuning": tuning,
            "chosen": {"encoder": best_key, "weight": best_weight},
            "current": current,
            "splits": results,
            "comparisons": comparisons,
            "pass_rule": {
                "candidate": plan["candidate"],
                "must_beat": plan["must_beat"],
                "split": plan["split"],
                "min_gain": plan["min_gain"],
                "passed": verdict,
                "outcome": outcome,
            },
        },
        timings,
    )


def to_markdown(results: dict[str, Any], timings: dict[str, Any] | None = None) -> str:
    def pct(stat: dict) -> str:
        return f"{100 * stat['mean']:.1f}% ({100 * stat['low']:.1f} to {100 * stat['high']:.1f})"

    def pts(stat: dict) -> str:
        return (
            f"{100 * stat['mean']:+.1f} pts ({100 * stat['low']:+.1f} to {100 * stat['high']:+.1f})"
        )

    rule = results["pass_rule"]
    chosen = results["chosen"]
    lines = [
        "# Round 6: larger transformer encoders",
        "",
        f"{results['personas']} personas, k = {results['k']}. Intervals are 95% persona bootstrap.",
        "",
        f"**Outcome: {rule['outcome']}.**",
        "",
        f"Chosen on already-seen sets: `{results['encoders'][chosen['encoder']]}` at weight "
        f"{chosen['weight']}.",
        "",
        "## Choosing the encoder and weight (mean clean hit over "
        + ", ".join(f"`{s}`" for s in results["tuning_splits"])
        + ")",
        "",
    ]
    weights = list(next(iter(results["tuning"].values())))
    lines += [
        "| Encoder | " + " | ".join(f"w = {w}" for w in weights) + " |",
        "| --- |" + " --- |" * len(weights),
    ]
    for key, row in results["tuning"].items():
        cells = " | ".join(f"{100 * row[w]['mean']:.1f}%" for w in weights)
        lines.append(f"| `{results['encoders'][key]}` | {cells} |")
    titles = {
        "holdout4": "New held-out set (all)",
        "holdout4_direct": "New held-out set: direct",
        "holdout4_indirect": "New held-out set: indirect",
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
    if timings:
        lines += [
            "",
            "## Embedding speed (this machine, one CPU thread)",
            "",
            "| Encoder | Texts | Seconds | Texts per second |",
            "| --- | --- | --- | --- |",
        ]
        for key, t in timings.items():
            lines.append(f"| {key} | {t['texts']:,} | {t['seconds']} | {t['per_second']} |")
    return "\n".join(lines) + "\n"
