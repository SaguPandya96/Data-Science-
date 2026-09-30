"""Round 2: does adding embedding similarity to Keel's score help? (docs/ANALYSIS_PLAN.md)"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from keel.evaluation.benchmark import load_config, paired_difference, score_arm, summarize
from keel.evaluation.scenarios import build_personas
from keel.memory.embeddings import Embedder, WordLlamaEmbedder
from keel.memory.retrieval import (
    EmbeddingMemory,
    FullMemory,
    KeelMemory,
    LexicalMemory,
    RecentMemory,
    Retriever,
)

SPLITS = ("holdout", "holdout_direct", "holdout_indirect", "test")


def run_round2(config_path: str | Path, embedder: Embedder | None = None) -> dict:
    config = load_config(config_path)
    bench, retrieval, analysis = config["benchmark"], config["retrieval"], config["analysis"]
    plan = config["round2"]
    seed, k, core_max = bench["seed"], retrieval["k"], retrieval["core_max"]
    resamples, confidence = analysis["bootstrap_resamples"], analysis["confidence"]
    embedder = embedder or WordLlamaEmbedder()
    personas = build_personas(bench)

    # 1. Choose the embedding weight on the dev wordings only.
    grid = {}
    for weight in plan["weight_grid"]:
        arm = KeelMemory(core_max=core_max, embedder=embedder, embedding_weight=weight)
        grid[str(weight)] = float(score_arm(arm, personas, "dev", k).all["clean_hit"].mean())
    best = max(plan["weight_grid"], key=lambda w: (grid[str(w)], -w))

    arms: list[Retriever] = [
        RecentMemory(),
        LexicalMemory(),
        EmbeddingMemory(embedder),
        KeelMemory(core_max=core_max),
        KeelMemory(
            core_max=core_max,
            embedder=embedder,
            embedding_weight=best,
            name=plan["candidate"],
        ),
        FullMemory(),
    ]

    # 2. Score every arm on the held-out wordings, then on the round 1 test wordings.
    results: dict[str, dict] = {}
    comparisons: dict[str, dict] = {}
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
    return {
        "embedder": embedder.name,
        "k": k,
        "personas": len(personas),
        "dev_weight_grid": grid,
        "chosen_weight": best,
        "splits": results,
        "comparisons": comparisons,
        "pass_rule": {
            "candidate": plan["candidate"],
            "must_beat": plan["must_beat"],
            "split": plan["split"],
            "passed": primary["low"] > 0,
        },
        "question_breakdown": _breakdown(personas, arms[3], arms[4], k),
    }


def _breakdown(personas: list, keel: Retriever, candidate: Retriever, k: int) -> list[dict]:
    """Hit rate per held-out wording, before and after embeddings."""
    hits: dict[tuple[str, str, str], dict[str, list[bool]]] = defaultdict(
        lambda: {"keel": [], "candidate": []}
    )
    for persona in personas:
        memories = persona.memories
        for style in ("direct", "indirect"):
            for probe in persona.probes[f"holdout_{style}"]:
                row = hits[(probe.slot, style, probe.question)]
                for label, arm in (("keel", keel), ("candidate", candidate)):
                    ids = {m.id for m in arm.retrieve(memories, probe.question, persona.now, k)}
                    row[label].append(probe.current_id in ids)
    return [
        {
            "detail": slot,
            "style": style,
            "question": question,
            "keel": round(float(np.mean(v["keel"])), 3),
            candidate.name: round(float(np.mean(v["candidate"])), 3),
        }
        for (slot, style, question), v in sorted(hits.items())
    ]


def to_markdown(results: dict) -> str:
    def pct(stat: dict) -> str:
        return f"{100 * stat['mean']:.1f}% ({100 * stat['low']:.1f} to {100 * stat['high']:.1f})"

    def pts(stat: dict) -> str:
        return (
            f"{100 * stat['mean']:+.1f} pts ({100 * stat['low']:+.1f} to {100 * stat['high']:+.1f})"
        )

    rule = results["pass_rule"]
    lines = [
        "# Round 2: embedding search",
        "",
        f"Embedder: `{results['embedder']}`. {results['personas']} personas, k = {results['k']}. "
        "Intervals are 95% persona bootstrap.",
        "",
        f"**Pass rule (`{rule['candidate']}` beats `{rule['must_beat']}` on "
        f"`{rule['split']}`): {'PASSED' if rule['passed'] else 'FAILED'}**",
        "",
        "## Embedding weight, chosen on dev",
        "",
        "| Weight | Dev clean hit |",
        "| --- | --- |",
    ]
    for weight, value in results["dev_weight_grid"].items():
        mark = " (chosen)" if float(weight) == results["chosen_weight"] else ""
        lines.append(f"| {weight}{mark} | {100 * value:.1f}% |")
    titles = {
        "holdout": "New held-out wordings (all)",
        "holdout_direct": "New held-out wordings: direct",
        "holdout_indirect": "New held-out wordings: indirect",
        "test": "Round 1 test wordings (already seen)",
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
    lines += [
        "",
        "## Hit rate per held-out wording",
        "",
        f"| Detail | Style | Question | `keel` | `{rule['candidate']}` |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in results["question_breakdown"]:
        lines.append(
            f"| {row['detail']} | {row['style']} | {row['question']} | "
            f"{100 * row['keel']:.0f}% | {100 * row[rule['candidate']]:.0f}% |"
        )
    return "\n".join(lines) + "\n"
