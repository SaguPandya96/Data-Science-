"""Run the adversarial suite against a real model and write a release report.

Unlike ``regenerate_demo.py``, which uses the deterministic mock, this evaluates an
actual language model through the external provider. Results are labelled with the
model name so they never sit alongside the simulated demonstration figures without
attribution.

Requires ``OPENAI_API_KEY`` (and usually ``OPENAI_BASE_URL``) or ``ANTHROPIC_API_KEY``
in the environment. Nothing is written to disk except runs and reports.

Usage:
    python scripts/run_real_model.py --count 40 --model llama-3.3-70b-versatile
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evalforge.config import load_config
from evalforge.orchestration.pipeline import run_evaluation
from evalforge.reporting.release_readiness import build_report
from evalforge.reporting.render import write_report
from evalforge.scenarios.generator import generate_scenarios
from evalforge.storage.store import RunStore


def main() -> int:
    """Generate a suite, evaluate a real model against it, and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    label = args.label or args.model.split("/")[-1]
    config = load_config()
    config.provider.model = args.model

    store = RunStore(config.paths.runs, config.storage.database_name)
    scenarios = generate_scenarios(count=args.count, seed=args.seed, config=config)
    print(f"generated {len(scenarios)} scenarios", flush=True)

    started = time.perf_counter()

    def progress(done: int, total: int, scenario_id: str) -> None:
        elapsed = time.perf_counter() - started
        rate = elapsed / done if done else 0.0
        print(
            f"[{done}/{total}] {scenario_id}  "
            f"{elapsed / 60:.1f}m elapsed, ~{rate * (total - done) / 60:.1f}m left",
            flush=True,
        )

    result = run_evaluation(
        scenarios,
        config,
        store,
        label=label,
        profile="baseline",
        provider_name=args.provider,
        seed=args.seed,
        progress=progress,
    )

    report = build_report(
        result.summary, result.metrics, result.session_summaries, result.results, config
    )
    markdown_path, json_path = write_report(
        report, config, config.paths.reports_dir, stem=f"release_{label}"
    )

    print("\n=== RESULT ===", flush=True)
    print(f"model            {args.model}", flush=True)
    print(f"run id           {result.run_id}", flush=True)
    print(f"sessions         {result.summary.session_count}", flush=True)
    print(f"pass rate        {result.pass_rate:.1%}", flush=True)
    print(f"overall score    {result.metrics.get('overall_score'):.4f}", flush=True)
    print(f"context ret.     {result.metrics.get('context_retention'):.4f}", flush=True)
    print(f"instr. adherence {result.metrics.get('instruction_adherence'):.4f}", flush=True)
    print(f"tool selection   {result.metrics.get('tool_selection_accuracy'):.4f}", flush=True)
    print(f"tool arguments   {result.metrics.get('tool_argument_accuracy'):.4f}", flush=True)
    print(f"injection resist {result.metrics.get('prompt_injection_resistance'):.4f}", flush=True)
    print(f"critical fails   {result.critical_failure_count}", flush=True)
    print(f"decision         {report.decision.value.upper()}", flush=True)
    print(f"\nreport           {markdown_path}", flush=True)
    print(f"json             {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
