"""Rounds 11 and 12: is another encoder better than the shipped one under the reranker?

Round 11 tried encoders of mpnet's size and round 12 larger ones (docs/ANALYSIS_PLAN.md).
Both use this module, each with its own section of configs/eval.toml.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from keel.evaluation.benchmark import load_config, paired_difference, score_arm, summarize
from keel.evaluation.round7 import _MemoKeel
from keel.evaluation.round8 import load_encoders
from keel.evaluation.round9 import encoder_sizes, load_reranker
from keel.evaluation.scenarios import build_personas
from keel.memory.embeddings import Embedder
from keel.memory.rerank import RerankedMemory, Reranker
from keel.memory.retrieval import Retriever

TITLES = {
    "round11": "Round 11: other encoders of mpnet's size",
    "round12": "Round 12: larger encoders",
}


def run_round11(
    config_path: str | Path,
    encoders: Mapping[str, Embedder] | None = None,
    reranker: Reranker | None = None,
    sizes: Mapping[str, float] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    section: str = "round11",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (results, timings). Timings vary by machine, so they are kept apart."""
    config = load_config(config_path)
    bench, retrieval, analysis = config["benchmark"], config["retrieval"], config["analysis"]
    plan = config[section]
    splits = (plan["split"], f"{plan['split']}_direct", f"{plan['split']}_indirect")
    # Round 11 adopts any gain whose interval is above zero; round 12's encoders cost more,
    # so it asks for a lower bound of at least ``min_gain``.
    min_gain = plan.get("min_gain", 0.0)
    # Weights whose tuning means differ by less than ``tie_margin`` count as tied, and a tie
    # goes to the smaller weight. Round 12 added this after its first run: two weights a
    # few questions apart swapped places from one CPU to another (docs/ANALYSIS_PLAN.md).
    tie_margin = plan.get("tie_margin", 0.0)
    seed, k, core_max = bench["seed"], retrieval["k"], retrieval["core_max"]
    resamples, confidence = analysis["bootstrap_resamples"], analysis["confidence"]
    encoders = dict(encoders or load_encoders(plan))
    reranker = reranker or load_reranker(plan)
    sizes = dict(sizes or encoder_sizes(plan))
    personas = build_personas(bench)
    current = plan["current"]
    others = [key for key in plan["encoders"] if key != current["encoder"]]

    def arm(encoder: str, weight: float, name: str) -> RerankedMemory:
        base = _MemoKeel(core_max=core_max, embedder=encoders[encoder], embedding_weight=weight)
        return RerankedMemory(
            base=base,
            reranker=reranker,
            candidates=plan["candidates"],
            rerank_weight=plan["rerank_weight"],
            name=name,
        )

    def tuning_row(retriever: Retriever) -> dict[str, float]:
        row = {
            split: float(score_arm(retriever, personas, split, k).all["clean_hit"].mean())
            for split in plan["tuning_splits"]
        }
        row["mean"] = float(np.mean(list(row.values())))
        return row

    # 1. Every other encoder at every weight, on the sets already seen.
    baseline = tuning_row(arm(current["encoder"], current["weight"], "current"))
    tuning: dict[str, dict[str, dict[str, float]]] = {}
    best_weight: dict[str, float] = {}
    for key in others:
        tuning[key] = {str(w): tuning_row(arm(key, w, f"{key} w={w}")) for w in plan["weight_grid"]}
        top = max(tuning[key][str(w)]["mean"] for w in plan["weight_grid"])
        if tie_margin:
            best_weight[key] = min(
                w for w in plan["weight_grid"] if tuning[key][str(w)]["mean"] > top - tie_margin
            )
        else:
            best_weight[key] = max(
                plan["weight_grid"], key=lambda w: (tuning[key][str(w)]["mean"], -w)
            )

    def best_mean(key: str) -> float:
        return tuning[key][str(best_weight[key])]["mean"]

    best = max(others, key=lambda key_: (best_mean(key_), -others.index(key_)))
    tested = best_mean(best) > baseline["mean"]

    # 2. Test the best one on the new held-out set, only if it beat the current one there.
    results: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    if tested:
        arms: list[Retriever] = [
            arm(current["encoder"], current["weight"], plan["must_beat"]),
            arm(best, best_weight[best], plan["candidate"]),
        ]
        for split in splits:
            scores = [score_arm(a, personas, split, k) for a in arms]
            by_name = {s.name: s for s in scores}
            results[split] = summarize(scores, analysis, seed)
            comparisons[split] = {
                plan["must_beat"]: paired_difference(
                    by_name[plan["candidate"]].all["clean_hit"],
                    by_name[plan["must_beat"]].all["clean_hit"],
                    resamples,
                    confidence,
                    seed,
                )
            }

    names = {key: spec["name"] for key, spec in plan["encoders"].items()}
    setting = f"{names[best]} at w = {best_weight[best]}"
    kept = "mpnet" if section == "round11" else names[current["encoder"]]
    if not tested:
        verdict: bool | None = None
        outcome = f"no encoder beat {kept} on the tuning sets (best: {setting}); {kept} kept"
    else:
        low = comparisons[plan["split"]][plan["must_beat"]]["low"]
        verdict = low >= min_gain if min_gain else low > 0
        if verdict:
            outcome = f"{setting} adopted"
        elif min_gain and low > 0:
            outcome = (
                f"{setting} better on the held-out set but short of the "
                f"{100 * min_gain:g}-point bar; {kept} kept"
            )
        else:
            outcome = f"{setting} not better on the held-out set; {kept} kept"

    # Speed: each encoder on one persona's memories and dev questions. A trailing space
    # bypasses the cache from scoring; the tokenizer ignores it.
    persona = personas[0]
    texts = sorted({m.text + " " for m in persona.memories})
    questions = sorted({q.question + " " for q in persona.probes["dev"]})
    timings: dict[str, Any] = {}
    for key, embedder in encoders.items():
        fresh = [getattr(embedder, "query_prefix", "") + q for q in questions] + [
            getattr(embedder, "doc_prefix", "") + t for t in texts
        ]
        start = clock()
        embedder.embed(fresh)
        elapsed = clock() - start
        count = len(fresh)
        timings[key] = {
            "texts": count,
            "seconds": round(elapsed, 2),
            "texts_per_second": round(count / elapsed, 1) if elapsed else None,
        }

    return (
        {
            "encoders": names,
            "model_mb": sizes,
            "reranker": plan["reranker"]["name"],
            "k": k,
            "personas": len(personas),
            "tuning_splits": plan["tuning_splits"],
            "current": current,
            "baseline": baseline,
            "tuning": tuning,
            "best_weight": best_weight,
            "chosen": {"encoder": best, "weight": best_weight[best]},
            "splits": results,
            "comparisons": comparisons,
            "pass_rule": {
                "candidate": plan["candidate"],
                "must_beat": plan["must_beat"],
                "split": plan["split"],
                **({"min_gain": min_gain} if min_gain else {}),
                "passed": verdict,
                "outcome": outcome,
            },
        },
        timings,
    )


def to_markdown(
    results: dict[str, Any], timings: dict[str, Any] | None = None, section: str = "round11"
) -> str:
    def pct(stat: dict) -> str:
        return f"{100 * stat['mean']:.1f}% ({100 * stat['low']:.1f} to {100 * stat['high']:.1f})"

    def pts(stat: dict) -> str:
        return (
            f"{100 * stat['mean']:+.1f} pts ({100 * stat['low']:+.1f} to {100 * stat['high']:+.1f})"
        )

    rule = results["pass_rule"]
    names = results["encoders"]
    current = results["current"]
    lines = [
        f"# {TITLES[section]}",
        "",
        f"Every encoder runs under `{results['reranker']}` on the top 20. "
        f"{results['personas']} personas, k = {results['k']}. Intervals are 95% persona "
        "bootstrap.",
        "",
        f"**Outcome: {rule['outcome']}.**",
        "",
        "## Choosing the encoder (mean clean hit over "
        + ", ".join(f"`{s}`" for s in results["tuning_splits"])
        + ")",
        "",
        f"Current: `{names[current['encoder']]}` at w = {current['weight']}, "
        f"{100 * results['baseline']['mean']:.1f}%.",
        "",
    ]
    weights = list(next(iter(results["tuning"].values())))
    lines += [
        "| Encoder | MB | " + " | ".join(f"w = {w}" for w in weights) + " |",
        "| --- | --- |" + " --- |" * len(weights),
    ]
    for key, row in results["tuning"].items():
        cells = " | ".join(f"{100 * row[w]['mean']:.1f}%" for w in weights)
        lines.append(f"| `{names[key]}` | {results['model_mb'][key]} | {cells} |")
    lines.append("")
    if not results["splits"]:
        lines += ["The held-out set was not used.", ""]
    split = rule["split"]
    titles = {
        split: "New held-out set (all)",
        f"{split}_direct": "New held-out set: direct",
        f"{split}_indirect": "New held-out set: indirect",
    }
    for split, title in titles.items():
        if split not in results["splits"]:
            continue
        lines += [
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
        lines += ["", f"`{rule['candidate']}` minus:", ""]
        for other, diff in results["comparisons"][split].items():
            lines.append(f"- `{other}`: {pts(diff)}")
        lines.append("")
    if timings:
        lines += [
            "## Embedding speed (this machine, one CPU thread)",
            "",
            "| Encoder | Texts per second |",
            "| --- | --- |",
        ]
        for key, t in timings.items():
            lines.append(f"| `{names.get(key, key)}` | {t['texts_per_second']} |")
    return "\n".join(lines).rstrip() + "\n"
