"""Round 7: does a cross-encoder reranker help? (docs/ANALYSIS_PLAN.md)"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from keel.evaluation.benchmark import load_config, paired_difference, score_arm, summarize
from keel.evaluation.round4 import load_encoders
from keel.evaluation.scenarios import build_personas
from keel.memory.embeddings import Embedder, download_files
from keel.memory.rerank import OnnxCrossEncoder, RerankedMemory, Reranker
from keel.memory.retrieval import KeelMemory, Retriever
from keel.memory.store import Memory

SPLITS = ("holdout5", "holdout5_direct", "holdout5_indirect")
ONLY = "only"  # the tuning-table label for ordering the short list by the logit alone


def load_rerankers(plan: dict[str, Any], cache_dir: Path | None = None) -> dict[str, Reranker]:
    rerankers: dict[str, Reranker] = {}
    for key, spec in plan["rerankers"].items():
        folder = download_files(spec["files"], cache_dir)
        rerankers[key] = OnnxCrossEncoder(folder, name=spec["name"])
    return rerankers


class _MemoKeel(KeelMemory):
    """Keel whose first-stage scores are computed once per question and memory list.

    Every grid point reranks the same short lists, so this only saves time; the scores
    are exactly those of ``KeelMemory``.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._memo: dict[Any, tuple[list[Memory], list[Memory], list[float]]] = {}

    def score_pool(
        self, memories: Sequence[Memory], query: str, now: datetime, k: int
    ) -> tuple[list[Memory], list[Memory], list[float]]:
        key = (query, now, k, tuple((m.id, m.text, m.superseded_by) for m in memories))
        if key not in self._memo:
            self._memo[key] = super().score_pool(memories, query, now, k)
        return self._memo[key]


def _label(weight: float | None) -> str:
    return ONLY if weight is None else str(weight)


def run_round7(
    config_path: str | Path,
    first_stage: Embedder | None = None,
    rerankers: Mapping[str, Reranker] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (results, timings). Timings vary by machine, so they are kept apart."""
    config = load_config(config_path)
    bench, retrieval, analysis = config["benchmark"], config["retrieval"], config["analysis"]
    plan = config["round7"]
    seed, k, core_max = bench["seed"], retrieval["k"], retrieval["core_max"]
    resamples, confidence = analysis["bootstrap_resamples"], analysis["confidence"]
    if first_stage is None:
        stage_plan = {"encoders": {"x": config["round6"]["encoders"][plan["first_stage"]]}}
        first_stage = load_encoders(stage_plan)["x"]
    rerankers = dict(rerankers or load_rerankers(plan))
    order = list(plan["rerankers"])  # smallest model first, so it wins ties
    personas = build_personas(bench)
    base = _MemoKeel(
        core_max=core_max,
        embedder=first_stage,
        embedding_weight=plan["first_stage_weight"],
        name=plan["must_beat"],
    )
    # Blends before ordering by the logit alone, and smaller weights first, so ties go to
    # the change closest to the current retriever.
    weights: list[float | None] = [*plan["weight_grid"], *([None] if plan["rerank_only"] else [])]

    def arm(key: str, n: int, weight: float | None, name: str) -> RerankedMemory:
        return RerankedMemory(
            base=base, reranker=rerankers[key], candidates=n, rerank_weight=weight, name=name
        )

    def tuning_row(retriever: Retriever) -> dict[str, float]:
        row = {
            split: float(score_arm(retriever, personas, split, k).all["clean_hit"].mean())
            for split in plan["tuning_splits"]
        }
        row["mean"] = float(np.mean(list(row.values())))
        return row

    # 1. Choose the reranker, short-list size and weight together, on sets already seen.
    baseline = tuning_row(base)
    tuning: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for key in order:
        tuning[key] = {}
        for n in plan["candidates_grid"]:
            tuning[key][str(n)] = {
                _label(w): tuning_row(arm(key, n, w, f"{key} n={n} w={_label(w)}")) for w in weights
            }
    grid = [(key, n, w) for key in order for n in plan["candidates_grid"] for w in weights]
    best_key, best_n, best_w = max(
        grid,
        key=lambda c: (
            tuning[c[0]][str(c[1])][_label(c[2])]["mean"],
            -order.index(c[0]),
            -c[1],
            -weights.index(c[2]),
        ),
    )
    best_mean = tuning[best_key][str(best_n)][_label(best_w)]["mean"]
    unchanged = best_mean <= baseline["mean"]

    # 2. Score once on the new held-out set, only if there is a change to test.
    arms: list[Retriever] = [base, arm(best_key, best_n, best_w, plan["candidate"])]
    results: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for split in SPLITS if not unchanged else ():
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

    if unchanged:
        verdict: bool | None = None
        outcome = "no reranking setting beat the current retriever on the tuning sets; kept"
    else:
        low = comparisons[plan["split"]][plan["must_beat"]]["low"]
        verdict = low >= plan["min_gain"]
        name = plan["rerankers"][best_key]["name"]
        setting = f"{name} on the top {best_n}, weight {_label(best_w)}"
        if verdict:
            outcome = f"{setting} adopted"
        elif low > 0:
            outcome = f"{setting} is better, but below the 1-point bar; not adopted"
        else:
            outcome = f"{setting} not better on the held-out set; not adopted"

    # Speed: each of one persona's dev questions against a 30-memory short list, scored in
    # one batch as the agent would. A trailing space on the question bypasses the cache from
    # tuning; the tokenizer ignores it.
    questions = [q.question + " " for q in personas[0].probes["dev"]]
    short_list = [base.document(m) for m in personas[0].memories][: max(plan["candidates_grid"])]
    timings = {}
    for key in order:
        start = clock()
        for question in questions:
            rerankers[key].score(question, short_list)
        elapsed = clock() - start
        timings[key] = {
            "questions": len(questions),
            "pairs": len(questions) * len(short_list),
            "seconds": round(elapsed, 2),
            "ms_per_question": round(1000 * elapsed / len(questions), 1),
        }

    return (
        {
            "first_stage": {"encoder": first_stage.name, "weight": plan["first_stage_weight"]},
            "rerankers": {key: plan["rerankers"][key]["name"] for key in order},
            "k": k,
            "personas": len(personas),
            "tuning_splits": plan["tuning_splits"],
            "baseline": baseline,
            "tuning": tuning,
            "chosen": {"reranker": best_key, "candidates": best_n, "weight": best_w},
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
    stage = results["first_stage"]
    splits = results["tuning_splits"]
    lines = [
        "# Round 7: cross-encoder reranking",
        "",
        f"First stage: Keel with `{stage['encoder']}` at weight {stage['weight']}. "
        f"{results['personas']} personas, k = {results['k']}. "
        "Intervals are 95% persona bootstrap.",
        "",
        f"**Outcome: {rule['outcome']}.**",
        "",
        f"Chosen on already-seen sets: `{results['rerankers'][chosen['reranker']]}` on the top "
        f"{chosen['candidates']}, weight {_label(chosen['weight'])}.",
        "",
        "## Choosing the setting (mean clean hit over " + ", ".join(f"`{s}`" for s in splits) + ")",
        "",
        f"Without reranking: {100 * results['baseline']['mean']:.1f}%.",
        "",
    ]
    for key, by_n in results["tuning"].items():
        labels = list(next(iter(by_n.values())))
        lines += [
            f"### `{results['rerankers'][key]}`",
            "",
            "| Short list | " + " | ".join(f"w = {w}" for w in labels) + " |",
            "| --- |" + " --- |" * len(labels),
        ]
        for n, row in by_n.items():
            cells = " | ".join(f"{100 * row[w]['mean']:.1f}%" for w in labels)
            lines.append(f"| top {n} | {cells} |")
        lines.append("")
    titles = {
        "holdout5": "New held-out set (all)",
        "holdout5_direct": "New held-out set: direct",
        "holdout5_indirect": "New held-out set: indirect",
    }
    if not results["splits"]:
        lines += ["The held-out set was not used.", ""]
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
        if split in results["comparisons"]:
            lines += ["", f"`{rule['candidate']}` minus:", ""]
            for other, diff in results["comparisons"][split].items():
                lines.append(f"- `{other}`: {pts(diff)}")
        lines.append("")
    if timings:
        lines += [
            "## Reranking speed (this machine, one CPU thread)",
            "",
            "| Reranker | Questions | Pairs | Seconds | Milliseconds per question |",
            "| --- | --- | --- | --- | --- |",
        ]
        for key, t in timings.items():
            lines.append(
                f"| {key} | {t['questions']} | {t['pairs']:,} | {t['seconds']} | "
                f"{t['ms_per_question']} |"
            )
    return "\n".join(lines).rstrip() + "\n"
