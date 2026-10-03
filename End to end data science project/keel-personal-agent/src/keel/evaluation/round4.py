"""Round 4: does a transformer sentence encoder beat WordLlama? (docs/ANALYSIS_PLAN.md)"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from keel.evaluation.benchmark import load_config, paired_difference, score_arm, summarize
from keel.evaluation.scenarios import build_personas
from keel.memory.embeddings import (
    Embedder,
    OnnxSentenceEmbedder,
    WordLlamaEmbedder,
    fetch_model,
)
from keel.memory.retrieval import (
    EMBEDDING_WEIGHT,
    EmbeddingMemory,
    KeelMemory,
    LexicalMemory,
    RecentMemory,
    Retriever,
)

SPLITS = ("holdout2", "holdout2_direct", "holdout2_indirect", "holdout")


def load_encoders(plan: dict[str, Any], cache_dir: Path | None = None) -> dict[str, Embedder]:
    encoders: dict[str, Embedder] = {}
    for key, spec in plan["encoders"].items():
        folder = fetch_model(spec, cache_dir)
        encoders[key] = OnnxSentenceEmbedder(folder, pooling=spec["pooling"], name=spec["name"])
    return encoders


def run_round4(
    config_path: str | Path,
    encoders: Mapping[str, Embedder] | None = None,
    wordllama: Embedder | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (results, timings). Timings vary by machine, so they are kept apart."""
    config = load_config(config_path)
    bench, retrieval, analysis = config["benchmark"], config["retrieval"], config["analysis"]
    plan = config["round4"]
    seed, k, core_max = bench["seed"], retrieval["k"], retrieval["core_max"]
    resamples, confidence = analysis["bootstrap_resamples"], analysis["confidence"]
    encoders = dict(encoders or load_encoders(plan))
    wordllama = wordllama or WordLlamaEmbedder()
    personas = build_personas(bench)

    # Embed every distinct memory and question once per encoder, and time it.
    texts = sorted(
        {m.text for p in personas for m in p.memories}
        | {f"{m.key.replace('_', ' ')}. {m.text}" for p in personas for m in p.memories if m.key}
        | {q.question for p in personas for probes in p.probes.values() for q in probes}
    )
    timings = {}
    for key, encoder in {**encoders, "wordllama": wordllama}.items():
        start = clock()
        encoder.embed(texts)
        elapsed = clock() - start
        timings[key] = {
            "texts": len(texts),
            "seconds": round(elapsed, 2),
            "per_second": round(len(texts) / elapsed, 1) if elapsed else None,
        }

    # 1. Choose the encoder and weight together, on dev only.
    grid: dict[str, dict[str, float]] = {}
    for key, encoder in encoders.items():
        grid[key] = {}
        for weight in plan["weight_grid"]:
            arm = KeelMemory(core_max=core_max, embedder=encoder, embedding_weight=weight)
            grid[key][str(weight)] = float(
                score_arm(arm, personas, "dev", k).all["clean_hit"].mean()
            )
    order = list(encoders)  # config order: MiniLM first, so it wins ties
    best_key, best_weight = max(
        ((key, w) for key in order for w in plan["weight_grid"]),
        key=lambda kw: (grid[kw[0]][str(kw[1])], -kw[1], -order.index(kw[0])),
    )
    chosen = encoders[best_key]

    arms: list[Retriever] = [
        RecentMemory(),
        LexicalMemory(),
        EmbeddingMemory(wordllama),
        EmbeddingMemory(chosen, name="transformer"),
        KeelMemory(core_max=core_max),
        KeelMemory(
            core_max=core_max,
            embedder=wordllama,
            embedding_weight=EMBEDDING_WEIGHT,
            name="keel+embed",
        ),
        KeelMemory(
            core_max=core_max,
            embedder=chosen,
            embedding_weight=best_weight,
            name=plan["candidate"],
        ),
    ]

    # 2. Score every arm on the new held-out set, then on the round 2 set.
    results: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for split in SPLITS:
        scores = [score_arm(a, personas, split, k) for a in arms]
        by_name = {s.name: s for s in scores}
        results[split] = summarize(scores, analysis, seed)
        candidate = by_name[plan["candidate"]].all["clean_hit"]
        comparisons[split] = {
            other: paired_difference(
                candidate, by_name[other].all["clean_hit"], resamples, confidence, seed
            )
            for other in [plan["must_beat"], *plan["also_compare"]]
        }

    primary = comparisons[plan["split"]][plan["must_beat"]]
    return (
        {
            "encoders": {key: e.name for key, e in encoders.items()},
            "k": k,
            "personas": len(personas),
            "dev_grid": grid,
            "chosen_encoder": chosen.name,
            "chosen_weight": best_weight,
            "splits": results,
            "comparisons": comparisons,
            "pass_rule": {
                "candidate": plan["candidate"],
                "must_beat": plan["must_beat"],
                "split": plan["split"],
                "passed": primary["low"] > 0,
            },
            "question_breakdown": _breakdown(personas, arms[5], arms[6], k),
        },
        timings,
    )


def _breakdown(personas: list, before: Retriever, after: Retriever, k: int) -> list[dict]:
    hits: dict[tuple[str, str, str], dict[str, list[bool]]] = defaultdict(
        lambda: {"before": [], "after": []}
    )
    for persona in personas:
        memories = persona.memories
        for style in ("direct", "indirect"):
            for probe in persona.probes[f"holdout2_{style}"]:
                row = hits[(probe.slot, style, probe.question)]
                for label, arm in (("before", before), ("after", after)):
                    ids = {m.id for m in arm.retrieve(memories, probe.question, persona.now, k)}
                    row[label].append(probe.current_id in ids)
    return [
        {
            "detail": slot,
            "style": style,
            "question": question,
            before.name: round(float(np.mean(v["before"])), 3),
            after.name: round(float(np.mean(v["after"])), 3),
        }
        for (slot, style, question), v in sorted(hits.items())
    ]


def to_markdown(results: dict[str, Any], timings: dict[str, Any] | None = None) -> str:
    def pct(stat: dict) -> str:
        return f"{100 * stat['mean']:.1f}% ({100 * stat['low']:.1f} to {100 * stat['high']:.1f})"

    def pts(stat: dict) -> str:
        return (
            f"{100 * stat['mean']:+.1f} pts ({100 * stat['low']:+.1f} to {100 * stat['high']:+.1f})"
        )

    rule = results["pass_rule"]
    lines = [
        "# Round 4: transformer sentence encoders",
        "",
        f"{results['personas']} personas, k = {results['k']}. Intervals are 95% persona bootstrap.",
        "",
        f"**Pass rule (`{rule['candidate']}` beats `{rule['must_beat']}` on "
        f"`{rule['split']}`): {'PASSED' if rule['passed'] else 'FAILED'}**",
        "",
        f"Chosen on dev: `{results['chosen_encoder']}` at weight {results['chosen_weight']}.",
        "",
        "## Encoder and weight, chosen on dev (clean hit)",
        "",
        "| Encoder | " + " | ".join(next(iter(results["dev_grid"].values()))) + " |",
        "| --- |" + " --- |" * len(next(iter(results["dev_grid"].values()))),
    ]
    for key, row in results["dev_grid"].items():
        name = results["encoders"][key]
        lines.append(f"| `{name}` | " + " | ".join(f"{100 * v:.1f}%" for v in row.values()) + " |")
    titles = {
        "holdout2": "Third held-out set (all)",
        "holdout2_direct": "Third held-out set: direct",
        "holdout2_indirect": "Third held-out set: indirect",
        "holdout": "Round 2 held-out set (already seen)",
    }
    for split, title in titles.items():
        lines += [
            "",
            f"## {title}",
            "",
            "| Arm | Clean hit | Stale shown | Allergy shown | Tokens |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name, stats in results["splits"][split].items():
            s = stats["all"]
            lines.append(
                f"| `{name}` | {pct(s['clean_hit'])} | {pct(s['stale'])} | "
                f"{pct(s['allergy_shown'])} | {s['tokens']['mean']:.0f} |"
            )
        lines += ["", f"`{rule['candidate']}` minus:", ""]
        for other, diff in results["comparisons"][split].items():
            lines.append(f"- `{other}`: {pts(diff)}")
    names = [
        k for k in results["question_breakdown"][0] if k not in ("detail", "style", "question")
    ]
    lines += [
        "",
        "## Hit rate per held-out wording",
        "",
        f"| Detail | Style | Question | `{names[0]}` | `{names[1]}` |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in results["question_breakdown"]:
        lines.append(
            f"| {row['detail']} | {row['style']} | {row['question']} | "
            f"{100 * row[names[0]]:.0f}% | {100 * row[names[1]]:.0f}% |"
        )
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
