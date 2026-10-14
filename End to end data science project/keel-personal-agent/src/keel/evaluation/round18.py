"""Round 18: does another cross-encoder rerank better than the shipped MiniLM-L-6?

The shipped first stage (int8 e5-large-v2 at weight 64) and short-list size (20) stay
fixed; only the reranker and its blend weight change (docs/ANALYSIS_PLAN.md).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from keel.evaluation.benchmark import load_config, paired_difference, score_arm, summarize
from keel.evaluation.round7 import _MemoKeel, load_rerankers
from keel.evaluation.round8 import load_encoders
from keel.evaluation.scenarios import build_personas
from keel.memory.embeddings import Embedder
from keel.memory.rerank import RerankedMemory, Reranker
from keel.memory.retrieval import Retriever

TITLE = "Round 18: other cross-encoder rerankers"


def run_round18(
    config_path: str | Path,
    encoder: Embedder | None = None,
    rerankers: Mapping[str, Reranker] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (results, timings). Timings vary by machine, so they are kept apart."""
    config = load_config(config_path)
    bench, retrieval, analysis = config["benchmark"], config["retrieval"], config["analysis"]
    plan = config["round18"]
    split = plan["split"]
    splits = (split, f"{split}_direct", f"{split}_indirect")
    seed, k, core_max = bench["seed"], retrieval["k"], retrieval["core_max"]
    resamples, confidence = analysis["bootstrap_resamples"], analysis["confidence"]
    if encoder is None:
        encoder = load_encoders({"encoders": {"x": plan["encoder"]}})["x"]
    rerankers = dict(rerankers or load_rerankers(plan))
    order = list(plan["rerankers"])  # listed fastest first
    personas = build_personas(bench)
    base = _MemoKeel(core_max=core_max, embedder=encoder, embedding_weight=plan["encoder_weight"])
    current = (plan["current"]["reranker"], plan["current"]["weight"])

    def arm(key: str, w: float, name: str) -> RerankedMemory:
        return RerankedMemory(
            base=base,
            reranker=rerankers[key],
            candidates=plan["candidates"],
            rerank_weight=w,
            name=name,
        )

    def tuning_row(retriever: Retriever) -> dict[str, float]:
        row = {
            s: float(score_arm(retriever, personas, s, k).all["clean_hit"].mean())
            for s in plan["tuning_splits"]
        }
        row["mean"] = float(np.mean(list(row.values())))
        return row

    # 1. Every reranker at every weight, on the sets already seen.
    tuning = {
        key: {str(w): tuning_row(arm(key, w, f"{key} w={w}")) for w in plan["weight_grid"]}
        for key in order
    }
    grid = [(key, w) for key in order for w in plan["weight_grid"]]
    top = max(tuning[key][str(w)]["mean"] for key, w in grid)
    # A gap of up to the margin counts as a tie; the small tolerance absorbs float rounding
    # (round 17's amendment).
    tied = [
        (key, w)
        for key, w in grid
        if top - tuning[key][str(w)]["mean"] <= plan["tie_margin"] + 1e-9
    ]
    # Within the tie margin the current setting is kept; otherwise the fastest reranker,
    # then the smallest weight.
    chosen = current if current in tied else min(tied, key=lambda c: (order.index(c[0]), c[1]))
    tested = chosen != current

    # 2. Test the chosen one once on the new held-out set, if it differs.
    costlier = chosen[0] not in plan["same_cost"]
    min_gain = plan["min_gain_costlier"] if costlier else plan["min_gain"]
    results: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    if tested:
        arms: list[Retriever] = [
            arm(*current, plan["must_beat"]),
            arm(*chosen, plan["candidate"]),
        ]
        for s in splits:
            scores = [score_arm(a, personas, s, k) for a in arms]
            by_name = {x.name: x for x in scores}
            results[s] = summarize(scores, analysis, seed)
            comparisons[s] = {
                plan["must_beat"]: paired_difference(
                    by_name[plan["candidate"]].all["clean_hit"],
                    by_name[plan["must_beat"]].all["clean_hit"],
                    resamples,
                    confidence,
                    seed,
                )
            }

    names = {key: spec["name"] for key, spec in plan["rerankers"].items()}
    setting = f"{names[chosen[0]]} at weight {chosen[1]}"
    kept = f"{names[current[0]]} at weight {current[1]} kept"
    if not tested:
        verdict: bool | None = None
        outcome = (
            f"the current reranker is within the tie margin of the best on the tuning sets; {kept}"
        )
    else:
        low = comparisons[split][plan["must_beat"]]["low"]
        verdict = low >= min_gain if min_gain else low > 0
        if verdict:
            outcome = f"{setting} adopted"
        elif low > 0:
            outcome = (
                f"{setting} better on the held-out set but short of the "
                f"{100 * min_gain:g}-point bar for a slower reranker; {kept}"
            )
        else:
            outcome = f"{setting} not better on the held-out set; {kept}"

    # Speed: each reranker on one persona's dev questions, 20 candidates each. A suffix on
    # the question keeps the reranker's cache from answering.
    persona = personas[0]
    memories = persona.store.all()
    questions = [q.question for q in persona.probes["dev"]]
    lists = [
        [
            base.document(m)
            for m in base.score_pool(memories, q, persona.now, k)[1][: plan["candidates"]]
        ]
        for q in questions
    ]
    timings: dict[str, Any] = {}
    for key in order:
        start = clock()
        for q, docs in zip(questions, lists, strict=True):
            rerankers[key].score(f"{q} #timing", docs)
        elapsed = clock() - start
        timings[key] = {
            "questions": len(questions),
            "ms_per_question": round(1000 * elapsed / len(questions), 1),
        }

    return (
        {
            "rerankers": names,
            "encoder": plan["encoder"]["name"],
            "encoder_weight": plan["encoder_weight"],
            "candidates": plan["candidates"],
            "k": k,
            "personas": len(personas),
            "tuning_splits": plan["tuning_splits"],
            "current": {"reranker": current[0], "weight": current[1]},
            "tuning": tuning,
            "chosen": {"reranker": chosen[0], "weight": chosen[1]},
            "splits": results,
            "comparisons": comparisons,
            "pass_rule": {
                "candidate": plan["candidate"],
                "must_beat": plan["must_beat"],
                "split": split,
                "min_gain": min_gain,
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
    names = results["rerankers"]
    current = results["current"]
    lines = [
        f"# {TITLE}",
        "",
        f"First stage: `{results['encoder']}` at w = {results['encoder_weight']}, top "
        f"{results['candidates']} reranked. {results['personas']} personas, k = {results['k']}. "
        "Intervals are 95% persona bootstrap.",
        "",
        f"**Outcome: {rule['outcome']}.**",
        "",
        "## Choosing the reranker (mean clean hit over "
        + ", ".join(f"`{s}`" for s in results["tuning_splits"])
        + ")",
        "",
        f"Current: `{names[current['reranker']]}` at weight {current['weight']}.",
        "",
    ]
    weights = list(next(iter(results["tuning"].values())))
    lines += [
        "| Reranker | " + " | ".join(f"w = {w}" for w in weights) + " |",
        "| --- |" + " --- |" * len(weights),
    ]
    for key, row in results["tuning"].items():
        cells = " | ".join(f"{100 * row[w]['mean']:.2f}%" for w in weights)
        lines.append(f"| `{names[key]}` | {cells} |")
    lines.append("")
    if not results["splits"]:
        lines += ["The held-out set was not used.", ""]
    split = rule["split"]
    titles = {
        split: "New held-out set (all)",
        f"{split}_direct": "New held-out set: direct",
        f"{split}_indirect": "New held-out set: indirect",
    }
    for s, title in titles.items():
        if s not in results["splits"]:
            continue
        lines += [
            f"## {title}",
            "",
            "| Arm | Clean hit | Stale shown | Allergy shown |",
            "| --- | --- | --- | --- |",
        ]
        for name, stats in results["splits"][s].items():
            x = stats["all"]
            lines.append(
                f"| `{name}` | {pct(x['clean_hit'])} | {pct(x['stale'])} | "
                f"{pct(x['allergy_shown'])} |"
            )
        lines += ["", f"`{rule['candidate']}` minus:", ""]
        for other, diff in results["comparisons"][s].items():
            lines.append(f"- `{other}`: {pts(diff)}")
        lines.append("")
    if timings:
        lines += [
            "## Reranking time (this machine, one CPU thread, 20 candidates)",
            "",
            "| Reranker | ms per question |",
            "| --- | --- |",
        ]
        for key, t in timings.items():
            lines.append(f"| `{names.get(key, key)}` | {t['ms_per_question']} |")
    return "\n".join(lines).rstrip() + "\n"
