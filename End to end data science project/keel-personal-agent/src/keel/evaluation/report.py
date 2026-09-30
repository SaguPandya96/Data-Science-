"""Markdown tables from benchmark results. The README copies its numbers from these."""

from __future__ import annotations


def _pct(stat: dict) -> str:
    return f"{100 * stat['mean']:.1f}% ({100 * stat['low']:.1f} to {100 * stat['high']:.1f})"


def _pts(stat: dict) -> str:
    return f"{100 * stat['mean']:+.1f} pts ({100 * stat['low']:+.1f} to {100 * stat['high']:+.1f})"


def _arm_table(arms: dict, subset: str, with_tokens: bool = True) -> list[str]:
    head = "| Arm | Clean hit | Hit | Stale shown | Allergy shown |"
    rule = "| --- | --- | --- | --- | --- |"
    if with_tokens:
        head += " Tokens |"
        rule += " --- |"
    lines = [head, rule]
    for name, stats in arms.items():
        s = stats[subset]
        row = (
            f"| `{name}` | {_pct(s['clean_hit'])} | {_pct(s['hit'])} | {_pct(s['stale'])} "
            f"| {_pct(s['allergy_shown'])} |"
        )
        if with_tokens:
            row += f" {s['tokens']['mean']:.0f} |"
        lines.append(row)
    return lines


def to_markdown(results: dict) -> str:
    rule = results["pass_rule"]
    verdict = "PASSED" if rule["passed"] else "FAILED"
    mem = results["memories_per_persona"]
    lines = [
        f"# Memory benchmark: {results['split']} templates",
        "",
        f"{results['personas']} personas, {results['questions']:,} questions, "
        f"{mem['mean']:.0f} memories per persona on average ({mem['min']} to {mem['max']}), "
        f"{100 * results['updated_share']:.0f}% of questions about a detail that changed. "
        f"k = {results['k']}. Intervals are 95% persona bootstrap.",
        "",
        f"**Pass rule ({rule['metric']} beats {', '.join(rule['baselines'])}): {verdict}**",
        "",
        "## All questions",
        "",
        *_arm_table(results["arms"], "all"),
        "",
        "## Details that changed value",
        "",
        *_arm_table(results["arms"], "updated"),
        "",
        "## Keel minus baseline, clean hit (paired)",
        "",
        "| Baseline | All questions | Changed details |",
        "| --- | --- | --- |",
    ]
    for baseline, diff in results["comparisons"].items():
        lines.append(f"| `{baseline}` | {_pts(diff['all'])} | {_pts(diff['updated'])} |")
    lines += ["", "## Ablation (all questions)", "", *_arm_table(results["ablations"], "all")]
    lines += ["", "## Key noise: share of updates written under a different key", ""]
    lines += ["| Noise | `recent` | `lexical` | `keel` | `keel` on changed details |"]
    lines += ["| --- | --- | --- | --- | --- |"]
    for level, arms in results["key_noise"].items():
        lines.append(
            f"| {100 * float(level):.0f}% | {_pct(arms['recent']['all']['clean_hit'])} | "
            f"{_pct(arms['lexical']['all']['clean_hit'])} | "
            f"{_pct(arms['keel']['all']['clean_hit'])} | "
            f"{_pct(arms['keel']['updated']['clean_hit'])} |"
        )
    lines += ["", "## Memories allowed in the prompt (clean hit, all questions)", ""]
    lines += ["| k | `recent` | `lexical` | `keel` |", "| --- | --- | --- | --- |"]
    for k, arms in results["k_sweep"].items():
        lines.append(
            f"| {k} | {_pct(arms['recent']['all']['clean_hit'])} | "
            f"{_pct(arms['lexical']['all']['clean_hit'])} | "
            f"{_pct(arms['keel']['all']['clean_hit'])} |"
        )
    return "\n".join(lines) + "\n"
