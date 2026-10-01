"""Rounds 9 and 10: can a smaller encoder replace mpnet under the reranker?

Round 10 reruns round 9's procedure with a different weight grid (docs/ANALYSIS_PLAN.md),
so both use this module, each with its own section of configs/eval.toml.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from keel.evaluation.benchmark import load_config, paired_difference, score_arm, summarize
from keel.evaluation.round7 import _MemoKeel
from keel.evaluation.round8 import _folder, load_encoders
from keel.evaluation.scenarios import build_personas
from keel.memory.embeddings import Embedder
from keel.memory.rerank import OnnxCrossEncoder, RerankedMemory, Reranker
from keel.memory.retrieval import Retriever

SPLITS = ("holdout7", "holdout7_direct", "holdout7_indirect")
TITLES = {"round9": "Round 9: smaller encoders", "round10": "Round 10: MiniLM below weight 4"}


def load_reranker(plan: dict[str, Any], cache_dir: Path | None = None) -> Reranker:
    spec = plan["reranker"]
    return OnnxCrossEncoder(_folder(spec, cache_dir), name=spec["name"])


def encoder_sizes(plan: dict[str, Any], cache_dir: Path | None = None) -> dict[str, float]:
    """Size of each encoder's ONNX model in MB, as downloaded."""
    sizes = {}
    for key, spec in plan["encoders"].items():
        folder = _folder(spec, cache_dir)
        onnx = next(p for p in folder.glob("*.onnx") if not p.name.startswith("._"))
        sizes[key] = round(onnx.stat().st_size / 1e6, 1)
    return sizes


def run_round9(
    config_path: str | Path,
    encoders: Mapping[str, Embedder] | None = None,
    reranker: Reranker | None = None,
    sizes: Mapping[str, float] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    section: str = "round9",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (results, timings). Timings vary by machine, so they are kept apart."""
    config = load_config(config_path)
    bench, retrieval, analysis = config["benchmark"], config["retrieval"], config["analysis"]
    plan = config[section]
    seed, k, core_max = bench["seed"], retrieval["k"], retrieval["core_max"]
    resamples, confidence = analysis["bootstrap_resamples"], analysis["confidence"]
    encoders = dict(encoders or load_encoders(plan))
    reranker = reranker or load_reranker(plan)
    sizes = dict(sizes or encoder_sizes(plan))
    personas = build_personas(bench)
    current = plan["current"]
    smaller = [key for key in plan["encoders"] if key != current["encoder"]]

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

    # 1. Score every smaller encoder at every weight on the sets already seen.
    baseline = tuning_row(arm(current["encoder"], current["weight"], "current"))
    tuning: dict[str, dict[str, dict[str, float]]] = {}
    best_weight: dict[str, float] = {}
    for key in smaller:
        tuning[key] = {str(w): tuning_row(arm(key, w, f"{key} w={w}")) for w in plan["weight_grid"]}
        best_weight[key] = max(plan["weight_grid"], key=lambda w: (tuning[key][str(w)]["mean"], -w))

    def best_mean(key: str) -> float:
        return tuning[key][str(best_weight[key])]["mean"]

    eligible = [k_ for k_ in smaller if best_mean(k_) >= baseline["mean"] - plan["max_tuning_drop"]]
    chosen = min(eligible, key=lambda k_: (sizes[k_], -best_mean(k_))) if eligible else None

    # 2. Score once on the new held-out set, only if an encoder qualified.
    results: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    if chosen is not None:
        arms: list[Retriever] = [
            arm(current["encoder"], current["weight"], plan["must_beat"]),
            arm(chosen, best_weight[chosen], plan["candidate"]),
        ]
        for split in SPLITS:
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
    if chosen is None:
        verdict: bool | None = None
        outcome = "no smaller encoder came within 0.5 points on the tuning sets; mpnet kept"
    else:
        low = comparisons[plan["split"]][plan["must_beat"]]["low"]
        verdict = low >= -plan["margin"]
        setting = f"{names[chosen]} at w = {best_weight[chosen]}"
        if verdict:
            outcome = f"{setting} adopted (no worse than mpnet)"
        else:
            outcome = f"{setting} not adopted (a loss of more than 1 point is not ruled out)"

    # Speed: each encoder on one persona's memories and dev questions. A trailing space
    # bypasses the cache from scoring; the tokenizer ignores it.
    persona = personas[0]
    sample = sorted(
        {m.text for m in persona.memories} | {q.question for q in persona.probes["dev"]}
    )
    timings: dict[str, Any] = {}
    for key, embedder in encoders.items():
        fresh = [f"{text} " for text in sample]
        start = clock()
        embedder.embed(fresh)
        elapsed = clock() - start
        timings[key] = {
            "texts": len(fresh),
            "seconds": round(elapsed, 2),
            "texts_per_second": round(len(fresh) / elapsed, 1) if elapsed else None,
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
            "eligible": eligible,
            "chosen": (
                None if chosen is None else {"encoder": chosen, "weight": best_weight[chosen]}
            ),
            "splits": results,
            "comparisons": comparisons,
            "pass_rule": {
                "candidate": plan["candidate"],
                "must_beat": plan["must_beat"],
                "split": plan["split"],
                "margin": plan["margin"],
                "passed": verdict,
                "outcome": outcome,
            },
        },
        timings,
    )


def to_markdown(
    results: dict[str, Any], timings: dict[str, Any] | None = None, section: str = "round9"
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
    eligible = ", ".join(f"`{names[k_]}`" for k_ in results["eligible"]) or "none"
    lines += ["", f"Within 0.5 points: {eligible}.", ""]
    if not results["splits"]:
        lines += ["The held-out set was not used.", ""]
    titles = {
        "holdout7": "New held-out set (all)",
        "holdout7_direct": "New held-out set: direct",
        "holdout7_indirect": "New held-out set: indirect",
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
