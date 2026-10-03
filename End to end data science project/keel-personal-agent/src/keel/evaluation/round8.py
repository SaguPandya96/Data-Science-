"""Round 8: can int8 models replace the full-precision ones? (docs/ANALYSIS_PLAN.md)"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from keel.evaluation.benchmark import load_config, paired_difference, score_arm, summarize
from keel.evaluation.round7 import _MemoKeel, load_rerankers
from keel.evaluation.scenarios import build_personas
from keel.memory.embeddings import (
    Embedder,
    OnnxSentenceEmbedder,
    fetch_model,
)
from keel.memory.rerank import RerankedMemory, Reranker
from keel.memory.retrieval import Retriever

SPLITS = ("holdout6", "holdout6_direct", "holdout6_indirect")
VARIANTS = (("int8", "fp32"), ("fp32", "int8"), ("int8", "int8"))  # (encoder, reranker)


def _folder(spec: Mapping[str, Any], cache_dir: Path | None) -> Path:
    return fetch_model(spec, cache_dir)


def load_encoders(plan: dict[str, Any], cache_dir: Path | None = None) -> dict[str, Embedder]:
    return {
        key: OnnxSentenceEmbedder(
            _folder(spec, cache_dir),
            pooling=spec["pooling"],
            name=spec["name"],
            query_prefix=spec.get("query_prefix", ""),
            doc_prefix=spec.get("doc_prefix", ""),
        )
        for key, spec in plan["encoders"].items()
    }


def model_sizes(plan: dict[str, Any], cache_dir: Path | None = None) -> dict[str, float]:
    """Size of each ONNX model in MB, as downloaded."""
    sizes = {}
    for kind in ("encoders", "rerankers"):
        for key, spec in plan[kind].items():
            folder = _folder(spec, cache_dir)
            onnx = next(p for p in folder.glob("*.onnx") if not p.name.startswith("._"))
            sizes[f"{kind[:-1]}:{key}"] = round(onnx.stat().st_size / 1e6, 1)
    return sizes


def _label(encoder: str, reranker: str) -> str:
    return f"encoder {encoder}, reranker {reranker}"


def run_round8(
    config_path: str | Path,
    encoders: Mapping[str, Embedder] | None = None,
    rerankers: Mapping[str, Reranker] | None = None,
    sizes: Mapping[str, float] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (results, timings). Timings vary by machine, so they are kept apart."""
    config = load_config(config_path)
    bench, retrieval, analysis = config["benchmark"], config["retrieval"], config["analysis"]
    plan = config["round8"]
    seed, k, core_max = bench["seed"], retrieval["k"], retrieval["core_max"]
    resamples, confidence = analysis["bootstrap_resamples"], analysis["confidence"]
    encoders = dict(encoders or load_encoders(plan))
    rerankers = dict(rerankers or load_rerankers(plan))
    sizes = dict(sizes or model_sizes(plan))
    personas = build_personas(bench)
    current = plan["current"]

    bases: dict[tuple[str, float], _MemoKeel] = {}

    def arm(encoder: str, weight: float, reranker: str, name: str) -> RerankedMemory:
        if (encoder, weight) not in bases:
            bases[encoder, weight] = _MemoKeel(
                core_max=core_max, embedder=encoders[encoder], embedding_weight=weight
            )
        return RerankedMemory(
            base=bases[encoder, weight],
            reranker=rerankers[reranker],
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

    def weights_for(encoder: str) -> list[float]:
        return plan["int8_encoder_weights"] if encoder == "int8" else [plan["encoder_weight"]]

    # 1. Score every variant on the sets already seen.
    baseline = tuning_row(
        arm(current["encoder"], plan["encoder_weight"], current["reranker"], "current")
    )
    tuning: dict[str, dict[str, dict[str, float]]] = {}
    best_weight: dict[str, float] = {}
    for encoder, reranker in VARIANTS:
        label = _label(encoder, reranker)
        tuning[label] = {
            str(w): tuning_row(arm(encoder, w, reranker, f"{label} w={w}"))
            for w in weights_for(encoder)
        }
        # Ties go to the shipped weight, then the smaller one.
        best_weight[label] = max(
            weights_for(encoder),
            key=lambda w: (tuning[label][str(w)]["mean"], w == plan["encoder_weight"], -w),
        )

    def size(encoder: str, reranker: str) -> float:
        return sizes[f"encoder:{encoder}"] + sizes[f"reranker:{reranker}"]

    eligible = [
        (encoder, reranker)
        for encoder, reranker in VARIANTS
        if tuning[_label(encoder, reranker)][str(best_weight[_label(encoder, reranker)])]["mean"]
        >= baseline["mean"] - plan["max_tuning_drop"]
    ]
    chosen = (
        min(
            eligible,
            key=lambda v: (
                size(*v),
                -tuning[_label(*v)][str(best_weight[_label(*v)])]["mean"],
            ),
        )
        if eligible
        else None
    )

    # 2. Score once on the new held-out set, only if a variant qualified.
    results: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    if chosen is not None:
        label = _label(*chosen)
        arms: list[Retriever] = [
            arm(current["encoder"], plan["encoder_weight"], current["reranker"], plan["must_beat"]),
            arm(chosen[0], best_weight[label], chosen[1], plan["candidate"]),
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

    if chosen is None:
        verdict: bool | None = None
        outcome = "no int8 variant came within 0.5 points on the tuning sets; nothing changes"
    else:
        low = comparisons[plan["split"]][plan["must_beat"]]["low"]
        verdict = low >= -plan["margin"]
        setting = _label(*chosen)
        if verdict:
            outcome = f"{setting} adopted (no worse than the current models)"
        else:
            outcome = f"{setting} not adopted (a loss of more than 1 point is not ruled out)"

    # Speed: each encoder on one persona's memories and dev questions, and each reranker on
    # 20 of those memories per dev question.
    persona = personas[0]
    sample = sorted(
        {m.text for m in persona.memories} | {q.question for q in persona.probes["dev"]}
    )
    questions = [q.question + " " for q in persona.probes["dev"]]
    short_list = [m.text for m in persona.memories][: plan["candidates"]]
    timings: dict[str, Any] = {}
    for key, embedder in encoders.items():
        fresh = [f"{text} " for text in sample]  # bypasses the cache; the tokenizer ignores it
        start = clock()
        embedder.embed(fresh)
        elapsed = clock() - start
        timings[f"encoder:{key}"] = {
            "texts": len(fresh),
            "seconds": round(elapsed, 2),
            "texts_per_second": round(len(fresh) / elapsed, 1) if elapsed else None,
        }
    for key, model in rerankers.items():
        start = clock()
        for question in questions:
            model.score(question, short_list)
        elapsed = clock() - start
        timings[f"reranker:{key}"] = {
            "questions": len(questions),
            "seconds": round(elapsed, 2),
            "ms_per_question": round(1000 * elapsed / len(questions), 1),
        }

    return (
        {
            "models": {
                **{f"encoder:{k_}": s["name"] for k_, s in plan["encoders"].items()},
                **{f"reranker:{k_}": s["name"] for k_, s in plan["rerankers"].items()},
            },
            "model_mb": sizes,
            "k": k,
            "personas": len(personas),
            "tuning_splits": plan["tuning_splits"],
            "baseline": baseline,
            "tuning": tuning,
            "best_weight": best_weight,
            "eligible": [_label(*v) for v in eligible],
            "chosen": None
            if chosen is None
            else {
                "encoder": chosen[0],
                "reranker": chosen[1],
                "encoder_weight": best_weight[_label(*chosen)],
            },
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


def to_markdown(results: dict[str, Any], timings: dict[str, Any] | None = None) -> str:
    def pct(stat: dict) -> str:
        return f"{100 * stat['mean']:.1f}% ({100 * stat['low']:.1f} to {100 * stat['high']:.1f})"

    def pts(stat: dict) -> str:
        return (
            f"{100 * stat['mean']:+.1f} pts ({100 * stat['low']:+.1f} to {100 * stat['high']:+.1f})"
        )

    rule = results["pass_rule"]
    lines = [
        "# Round 8: quantized models",
        "",
        f"{results['personas']} personas, k = {results['k']}. Intervals are 95% persona bootstrap.",
        "",
        f"**Outcome: {rule['outcome']}.**",
        "",
        "## Model sizes",
        "",
        "| Model | MB |",
        "| --- | --- |",
    ]
    for key, mb in results["model_mb"].items():
        lines.append(f"| `{results['models'][key]}` | {mb} |")
    lines += [
        "",
        "## Choosing the variant (mean clean hit over "
        + ", ".join(f"`{s}`" for s in results["tuning_splits"])
        + ")",
        "",
        f"Current models: {100 * results['baseline']['mean']:.1f}%.",
        "",
        "| Variant | Encoder weight | Mean clean hit |",
        "| --- | --- | --- |",
    ]
    for label, by_weight in results["tuning"].items():
        for weight, row in by_weight.items():
            mark = " (best)" if float(weight) == results["best_weight"][label] else ""
            lines.append(f"| {label} | {weight}{mark} | {100 * row['mean']:.1f}% |")
    lines += ["", f"Within 0.5 points: {', '.join(results['eligible']) or 'none'}.", ""]
    if not results["splits"]:
        lines += ["The held-out set was not used.", ""]
    titles = {
        "holdout6": "New held-out set (all)",
        "holdout6_direct": "New held-out set: direct",
        "holdout6_indirect": "New held-out set: indirect",
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
            "## Speed (this machine, one CPU thread)",
            "",
            "| Model | Measure | Value |",
            "| --- | --- | --- |",
        ]
        for key, t in timings.items():
            if "texts_per_second" in t:
                lines.append(f"| {key} | texts per second | {t['texts_per_second']} |")
            else:
                lines.append(f"| {key} | ms per question (20 memories) | {t['ms_per_question']} |")
    return "\n".join(lines).rstrip() + "\n"
