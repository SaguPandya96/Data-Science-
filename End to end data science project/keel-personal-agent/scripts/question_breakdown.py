"""Keel's hit rate for each question wording: where retrieval works and where it doesn't.

Descriptive, run after the test split; it changes nothing in the retriever.
Writes reports/metrics/question_breakdown_<split>.json.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from keel.evaluation.benchmark import load_config
from keel.evaluation.scenarios import build_personas
from keel.memory.retrieval import KeelMemory

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    args = parser.parse_args()

    config = load_config(ROOT / "configs" / "eval.toml")
    arm = KeelMemory(core_max=config["retrieval"]["core_max"])
    k = config["retrieval"]["k"]
    hits: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for persona in build_personas(config["benchmark"]):
        memories = persona.memories
        for probe in persona.probes[args.split]:
            chosen = arm.retrieve(memories, probe.question, persona.now, k)
            hits[(probe.slot, probe.question)].append(probe.current_id in {m.id for m in chosen})

    rows = sorted(
        (
            {"detail": slot, "question": q, "hit_rate": round(sum(v) / len(v), 3), "n": len(v)}
            for (slot, q), v in hits.items()
        ),
        key=lambda r: r["hit_rate"],
    )
    out = ROOT / "reports" / "metrics" / f"question_breakdown_{args.split}.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(f"{100 * row['hit_rate']:5.1f}%  {row['detail']:17s} {row['question']}")
    low = sum(r["hit_rate"] < 0.5 for r in rows)
    print(f"\n{low} of {len(rows)} wordings below 50%")


if __name__ == "__main__":
    main()
