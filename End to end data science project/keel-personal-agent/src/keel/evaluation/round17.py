"""Round 17: is the reranker's setting still right for the current encoder?

Round 7 chose the short-list size (20) and the blend weight (2.0) when mpnet at weight 12
was the encoder. The encoder is now an int8 e5-large-v2 at weight 64, which changes the
scale of the hybrid score the reranker's logit is added to (docs/ANALYSIS_PLAN.md).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from keel.evaluation.benchmark import load_config, paired_difference, score_arm, summarize
from keel.evaluation.round7 import _MemoKeel
from keel.evaluation.round8 import load_encoders
from keel.evaluation.round9 import load_reranker
from keel.evaluation.scenarios import build_personas
from keel.memory.embeddings import Embedder
from keel.memory.rerank import RerankedMemory, Reranker
from keel.memory.retrieval import Retriever

TITLE = "Round 17: the reranker's setting for the current encoder"


def _setting(n: int, w: float) -> str:
    return f"n={n} w={w}"


def run_round17(
    config_path: str | Path,
    encoder: Embedder | None = None,
    reranker: Reranker | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (results, timings). Timings vary by machine, so they are kept apart."""
    config = load_config(config_path)
    bench, retrieval, analysis = config["benchmark"], config["retrieval"], config["analysis"]
    plan = config["round17"]
    split = plan["split"]
    splits = (split, f"{split}_direct", f"{split}_indirect")
    seed, k, core_max = bench["seed"], retrieval["k"], retrieval["core_max"]
    resamples, confidence = analysis["bootstrap_resamples"], analysis["confidence"]
    if encoder is None:
        encoder = load_encoders({"encoders": {"x": plan["encoder"]}})["x"]
    reranker = reranker or load_reranker(plan)
    personas = build_personas(bench)
    base = _MemoKeel(core_max=core_max, embedder=encoder, embedding_weight=plan["encoder_weight"])
    current = (plan["current"]["candidates"], plan["current"]["weight"])

    def arm(n: int, w: float, name: str) -> RerankedMemory:
        return RerankedMemory(
            base=base, reranker=reranker, candidates=n, rerank_weight=w, name=name
        )

    def tuning_row(retriever: Retriever) -> dict[str, float]:
        row = {
            s: float(score_arm(retriever, personas, s, k).all["clean_hit"].mean())
            for s in plan["tuning_splits"]
        }
        row["mean"] = float(np.mean(list(row.values())))
        return row

    # 1. Every setting on the sets already seen.
    grid = [(n, w) for n in plan["candidates_grid"] for w in plan["weight_grid"]]
    tuning = {_setting(n, w): tuning_row(arm(n, w, _setting(n, w))) for n, w in grid}
    top = max(row["mean"] for row in tuning.values())
    # A gap of exactly the margin counts as a tie. Means are whole questions over a fixed
    # total, so compare with a small tolerance rather than trusting float subtraction
    # (added after the first run; docs/ANALYSIS_PLAN.md).
    tied = [s for s in grid if top - tuning[_setting(*s)]["mean"] <= plan["tie_margin"] + 1e-9]
    # Within the tie margin the current setting is kept; otherwise the cheapest wins:
    # the shortest list, then the smallest weight.
    chosen = current if current in tied else min(tied)
    tested = chosen != current

    # 2. Test the chosen setting once on the new held-out set, if it differs.
    costlier = chosen[0] > current[0]
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

    setting = f"{chosen[0]} candidates at weight {chosen[1]}"
    kept = f"the current {current[0]} candidates at weight {current[1]} kept"
    if not tested:
        verdict: bool | None = None
        outcome = (
            f"the current setting is within the tie margin of the best on the tuning sets; {kept}"
        )
    else:
        low = comparisons[split][plan["must_beat"]]["low"]
        verdict = low >= min_gain if min_gain else low > 0
        if verdict:
            outcome = f"{setting} adopted"
        elif low > 0:
            outcome = (
                f"{setting} better on the held-out set but short of the "
                f"{100 * min_gain:g}-point bar for a longer list; {kept}"
            )
        else:
            outcome = f"{setting} not better on the held-out set; {kept}"

    # Speed: reranking one persona's dev questions at each list size. Fresh scores each
    # time, since the reranker caches pairs it has seen.
    persona = personas[0]
    memories = persona.store.all()
    questions = [q.question for q in persona.probes["dev"]]
    timings: dict[str, Any] = {}
    for n in plan["candidates_grid"]:
        docs_per_question = [
            [base.document(m) for m in base.score_pool(memories, q, persona.now, k)[1][:n]]
            for q in questions
        ]
        start = clock()
        for q, docs in zip(questions, docs_per_question, strict=True):
            reranker.score(f"{q} #{n}", docs)
        elapsed = clock() - start
        timings[str(n)] = {
            "questions": len(questions),
            "ms_per_question": round(1000 * elapsed / len(questions), 1),
        }

    return (
        {
            "encoder": plan["encoder"]["name"],
            "encoder_weight": plan["encoder_weight"],
            "reranker": plan["reranker"]["name"],
            "k": k,
            "personas": len(personas),
            "tuning_splits": plan["tuning_splits"],
            "current": {"candidates": current[0], "weight": current[1]},
            "tuning": tuning,
            "chosen": {"candidates": chosen[0], "weight": chosen[1]},
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
    current = results["current"]
    lines = [
        f"# {TITLE}",
        "",
        f"First stage: `{results['encoder']}` at w = {results['encoder_weight']}. Reranker: "
        f"`{results['reranker']}`. {results['personas']} personas, k = {results['k']}. "
        "Intervals are 95% persona bootstrap.",
        "",
        f"**Outcome: {rule['outcome']}.**",
        "",
        "## Choosing the setting (mean clean hit over "
        + ", ".join(f"`{s}`" for s in results["tuning_splits"])
        + ")",
        "",
        f"Current: {current['candidates']} candidates at weight {current['weight']}.",
        "",
    ]
    weights = sorted({float(s.split("w=")[1]) for s in results["tuning"]})
    sizes = sorted({int(s.split()[0][2:]) for s in results["tuning"]})
    lines += [
        "| Candidates | " + " | ".join(f"w = {w}" for w in weights) + " |",
        "| --- |" + " --- |" * len(weights),
    ]
    for n in sizes:
        cells = " | ".join(
            f"{100 * results['tuning'][_setting(n, w)]['mean']:.2f}%" for w in weights
        )
        lines.append(f"| {n} | {cells} |")
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
            "## Reranking time (this machine, one CPU thread)",
            "",
            "| Candidates | ms per question |",
            "| --- | --- |",
        ]
        for size, t in timings.items():
            lines.append(f"| {size} | {t['ms_per_question']} |")
    return "\n".join(lines).rstrip() + "\n"
