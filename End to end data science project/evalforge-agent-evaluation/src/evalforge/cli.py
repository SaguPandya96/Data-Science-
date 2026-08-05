"""EvalForge command line interface.

Every command is a thin wrapper over the library: parse arguments, call one function,
render the result. No evaluation logic lives here, so anything the CLI can do is
equally reachable from Python, a notebook or a test.

Exit codes are meaningful, because the CLI is expected to run in CI:

===== ==========================================================================
Code  Meaning
===== ==========================================================================
0     Success
1     Expected user error (bad argument, missing run, invalid configuration)
2     Regression gate failed
3     Release gate failed on a blocking threshold
===== ==========================================================================

Expected errors print a sentence, not a traceback. An operator who mistypes a run id
should not have to read a stack to learn that.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from evalforge import __version__
from evalforge.analytics.alignment import build_alignment_report
from evalforge.config import EvalForgeConfig, load_config
from evalforge.exceptions import (
    EvalForgeError,
    RegressionGateError,
)
from evalforge.logging_config import configure_logging
from evalforge.orchestration.comparison import compare_runs
from evalforge.orchestration.pipeline import load_run_result, run_evaluation
from evalforge.reporting.release_readiness import build_report
from evalforge.reporting.render import write_alignment, write_comparison, write_report
from evalforge.scenarios.generator import generate_suite
from evalforge.scenarios.library import load_suite, save_suite, suite_exists
from evalforge.scenarios.validator import validate_suite
from evalforge.schemas.common import ReleaseDecision, ScenarioCategory
from evalforge.storage.store import RunStore

console = Console()

EXIT_OK = 0
EXIT_USER_ERROR = 1
EXIT_REGRESSION = 2
EXIT_RELEASE_GATE = 3

app = typer.Typer(
    name="evalforge",
    help=(
        "Automated evaluation and adversarial stress testing for multi-turn AI agents. "
        "Runs fully offline with the deterministic mock provider."
    ),
    add_completion=False,
    # Deliberately not ``no_args_is_help``: that exits 2, which collides with the
    # documented meaning of exit code 2 (regression gate failed). A bare invocation is
    # not an error, so the callback prints help and exits 0 instead.
    no_args_is_help=False,
    rich_markup_mode="rich",
)


def _bootstrap(log_level: str | None = None) -> EvalForgeConfig:
    """Load configuration and configure logging.

    The level comes from the configuration (and therefore from ``EVALFORGE_LOG_LEVEL``)
    unless a caller overrides it, so ``EVALFORGE_LOG_LEVEL=ERROR evalforge demo``
    actually quietens the run.

    Raises:
        typer.Exit: With :data:`EXIT_USER_ERROR` if the configuration is invalid.
    """
    try:
        config = load_config()
    except EvalForgeError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(EXIT_USER_ERROR) from exc
    configure_logging(log_level or config.logging.level, config.logging.format)
    config.paths.ensure()
    return config


def _store(config: EvalForgeConfig) -> RunStore:
    """Open the run store for the configured data directory."""
    return RunStore(config.paths.runs, config.storage.database_name)


def _fail(message: str, code: int = EXIT_USER_ERROR) -> None:
    """Print a readable error and exit.

    Raises:
        typer.Exit: Always.
    """
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(code)


def _decision_style(decision: ReleaseDecision) -> str:
    """Colour for a release decision."""
    return {
        ReleaseDecision.PASS: "green",
        ReleaseDecision.CONDITIONAL_PASS: "yellow",
        ReleaseDecision.FAIL: "red",
    }[decision]


# ``invoke_without_command`` lets ``evalforge --version`` run without naming a
# subcommand, while ``no_args_is_help`` still shows help for a bare invocation.
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool, typer.Option("--version", help="Show the EvalForge version and exit.")
    ] = False,
) -> None:
    """EvalForge: evaluate complete multi-turn agent sessions."""
    if version:
        console.print(f"EvalForge {__version__}")
        raise typer.Exit(EXIT_OK)
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit(EXIT_OK)


# --------------------------------------------------------------------- generate


@app.command()
def generate(
    count: Annotated[int, typer.Option("--count", "-n", help="Scenarios to generate.")] = 150,
    seed: Annotated[int, typer.Option("--seed", "-s", help="Master seed.")] = 42,
    suite: Annotated[str, typer.Option("--suite", help="Suite name.")] = "core",
    no_overwrite: Annotated[
        bool,
        typer.Option(
            "--no-overwrite",
            help="Fail instead of replacing an existing suite of this name.",
        ),
    ] = False,
) -> None:
    """Generate a deterministic adversarial scenario suite.

    Regenerating overwrites by default. That is safe because generation is fully
    determined by ``--seed``: the same seed always reproduces the same suite, so
    re-running the command is idempotent rather than destructive.
    """
    config = _bootstrap()
    if count < 8:
        _fail("--count must be at least 8 so every category is represented.")

    directory = config.paths.generated_scenarios
    if suite_exists(directory, suite) and no_overwrite:
        _fail(f"Suite {suite!r} already exists and --no-overwrite was given.")

    with console.status(f"Generating {count} scenarios (seed {seed})..."):
        built = generate_suite(count=count, seed=seed, config=config, name=suite)
        path = save_suite(built, directory)

    report = validate_suite(built.scenarios, expected_count=count)
    table = Table(title=f"Suite '{suite}' — {len(built.scenarios)} scenarios", show_lines=False)
    table.add_column("Category")
    table.add_column("Count", justify="right")
    for name, number in sorted(report.category_distribution.items()):
        table.add_row(name.replace("_", " "), str(number))
    console.print(table)

    lengths = ", ".join(
        f"{length}:{number}" for length, number in sorted(report.length_distribution.items())
    )
    difficulties = ", ".join(
        f"{name}:{number}" for name, number in sorted(report.difficulty_distribution.items())
    )
    console.print(f"Conversation lengths — {lengths}")
    console.print(f"Difficulty — {difficulties}")
    console.print(f"[green]Saved[/green] {path}")

    if not report.ok:
        for problem in report.suite_errors:
            console.print(f"[red]Suite error:[/red] {problem}")
        raise typer.Exit(EXIT_USER_ERROR)


@app.command("validate-scenarios")
def validate_scenarios(
    suite: Annotated[str, typer.Option("--suite", help="Suite name.")] = "core",
    strict: Annotated[
        bool, typer.Option("--strict", help="Treat quality warnings as failures.")
    ] = False,
) -> None:
    """Validate a generated suite and report its composition."""
    config = _bootstrap()
    try:
        loaded = load_suite(config.paths.generated_scenarios, suite)
    except EvalForgeError as exc:
        _fail(str(exc))
        return

    report = validate_suite(loaded.scenarios)
    console.print(f"Validated [bold]{report.total}[/bold] scenarios in suite {suite!r}.")
    console.print(f"Valid: {report.valid}/{report.total}")

    for problem in report.suite_errors:
        console.print(f"  [red]error[/red] {problem}")
    for warning in report.suite_warnings:
        console.print(f"  [yellow]warning[/yellow] {warning}")

    for scenario_report in report.invalid_scenarios[:10]:
        console.print(
            f"  [red]{scenario_report.scenario_id}[/red]: {'; '.join(scenario_report.errors)}"
        )

    warning_count = sum(len(item.warnings) for item in report.scenario_reports)
    console.print(f"Scenario-level warnings: {warning_count}")

    table = Table(title="Composition")
    table.add_column("Dimension")
    table.add_column("Distribution")
    table.add_row("category", _render_counts(report.category_distribution))
    table.add_row("difficulty", _render_counts(report.difficulty_distribution))
    table.add_row(
        "length", _render_counts({str(k): v for k, v in report.length_distribution.items()})
    )
    table.add_row("tools", _render_counts(report.tool_coverage))
    table.add_row("injected faults", _render_counts(report.injected_failure_distribution))
    console.print(table)

    if not report.ok or (strict and (report.suite_warnings or warning_count)):
        raise typer.Exit(EXIT_USER_ERROR)
    console.print("[green]All scenarios valid.[/green]")


def _render_counts(counts: dict[str, int]) -> str:
    """Compact rendering of a distribution."""
    if not counts:
        return "-"
    return ", ".join(f"{name}={value}" for name, value in sorted(counts.items()))


# -------------------------------------------------------------------------- run


@app.command()
def run(
    suite: Annotated[str, typer.Option("--suite", help="Suite name to execute.")] = "core",
    provider: Annotated[
        str, typer.Option("--provider", help="mock | anthropic | openai.")
    ] = "mock",
    profile: Annotated[str, typer.Option("--profile", help="Mock behaviour profile.")] = "baseline",
    label: Annotated[str, typer.Option("--label", help="Run label.")] = "baseline",
    category: Annotated[
        str | None, typer.Option("--category", help="Restrict to one scenario category.")
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Master seed.")] = 42,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Run only the first N scenarios.")
    ] = None,
) -> None:
    """Execute a scenario suite against an agent and score every session."""
    config = _bootstrap()
    store = _store(config)

    try:
        loaded = load_suite(config.paths.generated_scenarios, suite)
    except EvalForgeError as exc:
        _fail(f"{exc} Run `evalforge generate` first.")
        return

    scenarios = loaded.scenarios
    if category:
        try:
            wanted = ScenarioCategory(category)
        except ValueError:
            valid = ", ".join(item.value for item in ScenarioCategory)
            _fail(f"Unknown category {category!r}. Valid categories: {valid}")
            return
        scenarios = [item for item in scenarios if item.category is wanted]
        if not scenarios:
            _fail(f"No scenarios in suite {suite!r} have category {category!r}.")
    if limit:
        scenarios = scenarios[:limit]

    console.print(
        f"Running [bold]{len(scenarios)}[/bold] scenarios "
        f"(provider={provider}, profile={profile}, seed={seed})"
    )

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Evaluating sessions", total=len(scenarios))

            def advance(done: int, _total: int, scenario_id: str) -> None:
                progress.update(task, completed=done, description=f"Evaluating {scenario_id[:20]}")

            result = run_evaluation(
                scenarios=scenarios,
                config=config,
                store=store,
                label=label,
                profile=profile,
                provider_name=provider,
                seed=seed,
                suite_id=loaded.suite_id,
                progress=advance,
            )
    except EvalForgeError as exc:
        _fail(str(exc))
        return

    _print_run_summary(result.run_id, result.metrics, result.critical_failure_count)
    console.print(f"\nRun id: [bold cyan]{result.run_id}[/bold cyan]")
    console.print(f"Report it with: [dim]evalforge report --run-id {result.run_id}[/dim]")


def _print_run_summary(run_id: str, metrics: Any, critical: int) -> None:
    """Print the headline metrics for a completed run."""
    table = Table(title=f"Run {run_id}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for name in (
        "pass_rate",
        "overall_score",
        "task_completion",
        "context_retention",
        "instruction_adherence",
        "tool_reliability",
        "recovery_success_rate",
        "prompt_injection_resistance",
    ):
        table.add_row(name.replace("_", " "), f"{metrics.get(name):.4f}")
    table.add_row("critical failures", f"[red]{critical}[/red]" if critical else "0")
    console.print(table)


@app.command()
def evaluate(
    run_id: Annotated[str, typer.Option("--run-id", help="Run to re-score.")],
) -> None:
    """Re-score a stored run from its traces, without re-running the agent."""
    config = _bootstrap()
    store = _store(config)

    from evalforge.orchestration.pipeline import reevaluate_run

    try:
        result = reevaluate_run(run_id, config, store)
    except EvalForgeError as exc:
        _fail(str(exc))
        return

    console.print(
        f"Re-scored {len(result.session_summaries)} sessions under config {config.digest}."
    )
    _print_run_summary(run_id, result.metrics, result.critical_failure_count)


# ---------------------------------------------------------------------- inspect


@app.command()
def runs() -> None:
    """List every stored run."""
    config = _bootstrap()
    store = _store(config)
    stored = store.list_runs()
    if not stored:
        console.print("No runs stored yet. Try [dim]evalforge demo[/dim].")
        return

    table = Table(title="Stored runs")
    table.add_column("Run id")
    table.add_column("Label")
    table.add_column("Provider")
    table.add_column("Sessions", justify="right")
    table.add_column("Pass rate", justify="right")
    table.add_column("Decision")
    for item in stored:
        table.add_row(
            item.run_id,
            item.label,
            item.provider_name,
            str(item.session_count),
            f"{item.metrics.get('pass_rate', 0.0):.3f}",
            item.release_decision.value,
        )
    console.print(table)


@app.command()
def inspect(
    run_id: Annotated[str, typer.Option("--run-id", help="Run to inspect.")],
    scenario_id: Annotated[
        str | None, typer.Option("--scenario-id", help="Scenario to inspect.")
    ] = None,
    failures_only: Annotated[
        bool, typer.Option("--failures-only", help="List only failing sessions.")
    ] = False,
) -> None:
    """Inspect a run, or drill into one scenario's full trace."""
    config = _bootstrap()
    store = _store(config)

    try:
        summaries = store.get_sessions(run_id)
        if not summaries:
            _fail(f"Run {run_id!r} has no stored sessions.")
    except EvalForgeError as exc:
        _fail(str(exc))
        return

    if scenario_id is None:
        table = Table(title=f"Sessions in {run_id}")
        table.add_column("Scenario")
        table.add_column("Category")
        table.add_column("Turns", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Pass")
        table.add_column("Failures")
        for item in summaries:
            if failures_only and item.passed:
                continue
            table.add_row(
                item.scenario_id,
                item.scenario_category,
                str(item.turn_count),
                f"{item.overall_score:.3f}",
                "[green]yes[/green]" if item.passed else "[red]no[/red]",
                ", ".join(item.failure_categories[:3]) or "-",
            )
        console.print(table)
        console.print("\nDrill in with [dim]--scenario-id <SCENARIO_ID>[/dim]")
        return

    match = next((item for item in summaries if item.scenario_id == scenario_id), None)
    if match is None:
        _fail(f"Scenario {scenario_id!r} not found in run {run_id!r}.")
        return

    try:
        trace = store.get_trace(run_id, match.session_id)
        scenario = store.get_scenario(run_id, scenario_id)
    except EvalForgeError as exc:
        _fail(str(exc))
        return

    console.print(
        Panel(
            f"[bold]{scenario.name if scenario else scenario_id}[/bold]\n"
            f"{scenario.description if scenario else ''}",
            title=f"{match.scenario_category} / {match.scenario_difficulty}",
        )
    )

    for turn in trace.turns:
        console.print(f"\n[bold cyan]Turn {turn.turn_index}[/bold cyan] [dim]user[/dim]")
        console.print(f"  {turn.user_message}")
        for call in turn.tool_calls:
            status = "[green]ok[/green]" if call.succeeded else f"[red]{call.error_type}[/red]"
            authorised = "" if call.authorized else " [red](UNAUTHORISED)[/red]"
            console.print(f"  [dim]tool[/dim] {call.tool_name.value} -> {status}{authorised}")
        if turn.assistant_message:
            console.print(f"  [dim]assistant[/dim] {turn.assistant_message[:400]}")

    results = store.get_evaluations(run_id, match.session_id)
    failures = [item for item in results if not item.passed]
    if failures:
        console.print(f"\n[bold red]{len(failures)} failing check(s)[/bold red]")
        for failure in failures[:15]:
            console.print(
                f"  [{failure.severity.value}] {failure.evaluator_name} "
                f"({failure.failure_category.value}): {failure.reasoning_summary}"
            )
            for evidence in failure.evidence[:1]:
                console.print(f"      [dim]{evidence.kind}: {evidence.excerpt[:160]}[/dim]")
    else:
        console.print("\n[green]All checks passed for this session.[/green]")

    console.print(
        f"\nScore {match.overall_score:.3f} | "
        f"{'PASS' if match.passed else 'FAIL'} | "
        f"critical failures: {len(match.critical_failures)}"
    )


# ---------------------------------------------------------------------- compare


@app.command()
def compare(
    baseline: Annotated[str, typer.Option("--baseline", help="Baseline run id.")],
    candidate: Annotated[str, typer.Option("--candidate", help="Candidate run id.")],
    output: Annotated[
        Path | None, typer.Option("--output", help="Directory for comparison reports.")
    ] = None,
    fail_on_regression: Annotated[
        bool,
        typer.Option(
            "--fail-on-regression/--no-fail-on-regression",
            help="Exit non-zero when the gate fails.",
        ),
    ] = True,
) -> None:
    """Compare two runs and enforce the regression gate."""
    config = _bootstrap()
    store = _store(config)

    try:
        base = load_run_result(baseline, store)
        cand = load_run_result(candidate, store)
    except EvalForgeError as exc:
        _fail(str(exc))
        return

    report = compare_runs(
        base.metrics,
        cand.metrics,
        config,
        base.session_summaries,
        cand.session_summaries,
    )

    table = Table(title=f"{baseline} -> {candidate}")
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Candidate", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("Verdict")
    for delta in report.deltas:
        if delta.regressed:
            verdict = "[red]REGRESSED[/red]"
        elif delta.tolerance is None:
            verdict = "[dim]not gated[/dim]"
        else:
            verdict = "[green]ok[/green]"
        table.add_row(
            delta.name.replace("_", " "),
            f"{delta.baseline:.4f}",
            f"{delta.candidate:.4f}",
            f"{delta.absolute_change:+.4f}",
            verdict,
        )
    console.print(table)

    directory = output or config.paths.reports_dir
    markdown_path, json_path = write_comparison(report, directory)
    console.print(f"\nWrote {markdown_path}")
    console.print(f"Wrote {json_path}")

    if report.regressions:
        console.print("\n[bold red]Regression gate FAILED[/bold red]")
        for item in report.regressions:
            console.print(f"  - {item}")
        if fail_on_regression:
            raise typer.Exit(EXIT_REGRESSION)
    else:
        console.print("\n[bold green]Regression gate passed[/bold green]")


# ----------------------------------------------------------------------- report


@app.command()
def report(
    run_id: Annotated[str, typer.Option("--run-id", help="Run to report on.")],
    baseline: Annotated[
        str | None, typer.Option("--baseline", help="Baseline run for regression findings.")
    ] = None,
    output: Annotated[Path | None, typer.Option("--output", help="Output directory.")] = None,
    stem: Annotated[
        str | None,
        typer.Option(
            "--stem",
            help="Filename stem for the report. Defaults to release_<run_id>. Pass the "
            "run's label to overwrite the demo's reports rather than sitting alongside them.",
        ),
    ] = None,
    fail_on_gate: Annotated[
        bool, typer.Option("--fail-on-gate", help="Exit non-zero when the release gate fails.")
    ] = False,
) -> None:
    """Generate the release-readiness report in Markdown and JSON."""
    config = _bootstrap()
    store = _store(config)

    try:
        result = load_run_result(run_id, store)
        scenarios = store.get_scenarios(run_id)
    except EvalForgeError as exc:
        _fail(str(exc))
        return

    composition: dict[str, int] = {}
    for scenario in scenarios:
        composition[scenario.category.value] = composition.get(scenario.category.value, 0) + 1

    comparison = None
    if baseline:
        try:
            base = load_run_result(baseline, store)
        except EvalForgeError as exc:
            _fail(str(exc))
            return
        comparison = compare_runs(
            base.metrics,
            result.metrics,
            config,
            base.session_summaries,
            result.session_summaries,
        )

    annotations = store.get_annotations(run_id, blind_only=True)
    alignment = (
        build_alignment_report(run_id, annotations, result.session_summaries)
        if annotations
        else None
    )

    built = build_report(
        run_summary=result.summary,
        metrics=result.metrics,
        sessions=result.session_summaries,
        results=result.results,
        config=config,
        comparison=comparison,
        alignment=alignment,
        scenario_composition=composition,
    )

    result.summary.release_decision = built.decision
    store.save_run(result.summary)

    directory = output or config.paths.reports_dir
    resolved_stem = f"release_{stem}" if stem else f"release_{run_id}"
    markdown_path, json_path = write_report(built, config, directory, stem=resolved_stem)
    if alignment is not None:
        write_alignment(alignment, directory, stem=f"alignment_{stem or run_id}")

    console.print(
        Panel(
            built.executive_summary,
            title=f"[{_decision_style(built.decision)}]{built.decision.value.upper()}[/]",
            border_style=_decision_style(built.decision),
        )
    )
    console.print(f"Wrote {markdown_path}")
    console.print(f"Wrote {json_path}")

    if fail_on_gate and built.decision is ReleaseDecision.FAIL:
        raise typer.Exit(EXIT_RELEASE_GATE)


@app.command()
def export(
    run_id: Annotated[str, typer.Option("--run-id", help="Run to export.")],
    output_format: Annotated[str, typer.Option("--format", help="json | jsonl | csv.")] = "json",
    output: Annotated[Path | None, typer.Option("--output", help="Output file.")] = None,
) -> None:
    """Export a run's session summaries for external analysis."""
    import csv
    import json as json_module

    config = _bootstrap()
    store = _store(config)

    if output_format not in {"json", "jsonl", "csv"}:
        _fail(f"Unknown format {output_format!r}. Use json, jsonl or csv.")

    try:
        summaries = store.get_sessions(run_id)
    except EvalForgeError as exc:
        _fail(str(exc))
        return
    if not summaries:
        _fail(f"Run {run_id!r} has no stored sessions.")

    path = output or (config.paths.reports_dir / f"{run_id}_sessions.{output_format}")
    path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for item in summaries:
                handle.write(item.model_dump_json())
                handle.write("\n")
    elif output_format == "csv":
        rows = [item.model_dump(mode="json") for item in summaries]
        flat_keys = [key for key, value in rows[0].items() if not isinstance(value, list | dict)]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=flat_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    else:
        payload = [item.model_dump(mode="json") for item in summaries]
        path.write_text(json_module.dumps(payload, indent=2), encoding="utf-8")

    console.print(f"[green]Exported[/green] {len(summaries)} sessions to {path}")


# ------------------------------------------------------------------ interactive


@app.command()
def dashboard(
    port: Annotated[int, typer.Option("--port", help="Port to serve on.")] = 8501,
) -> None:
    """Launch the Streamlit dashboard."""
    _launch_streamlit("app.py", port)


@app.command()
def annotate(
    port: Annotated[int, typer.Option("--port", help="Port to serve on.")] = 8502,
) -> None:
    """Launch the blind human annotation interface."""
    _launch_streamlit("pages/6_Human_Annotation.py", port)


def _launch_streamlit(relative: str, port: int) -> None:
    """Run a Streamlit entry point.

    Raises:
        typer.Exit: If Streamlit is not installed or the file is missing.
    """
    import subprocess

    from evalforge.config import PROJECT_ROOT

    target = PROJECT_ROOT / "dashboard" / relative
    if not target.exists():
        _fail(f"Dashboard entry point not found: {target}")

    try:
        import streamlit  # noqa: F401
    except ImportError:
        _fail('Streamlit is not installed. Install it with: pip install -e ".[dashboard]"')

    console.print(f"Starting Streamlit on http://localhost:{port}")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(target), "--server.port", str(port)],
        check=False,
    )


# -------------------------------------------------------------------------- demo


@app.command()
def demo(
    count: Annotated[int, typer.Option("--count", "-n", help="Scenarios to generate.")] = 150,
    seed: Annotated[int, typer.Option("--seed", help="Master seed.")] = 42,
    fail_on_regression: Annotated[
        bool,
        typer.Option(
            "--fail-on-regression/--no-fail-on-regression",
            help="Exit non-zero when the candidate regresses. Off by default: the demo "
            "is *supposed* to detect a regression, so a non-zero exit is the expected "
            "outcome and would be confusing as a default.",
        ),
    ] = False,
) -> None:
    """Run the complete offline demonstration end to end.

    Generates the suite, runs a reliable baseline agent and a deliberately degraded
    candidate, evaluates both, compares them, and writes every report.
    """
    config = _bootstrap()
    store = _store(config)

    console.print(
        Panel(
            "[bold]EvalForge demonstration[/bold]\n\n"
            "Generates an adversarial suite, runs a reliable baseline agent and a "
            "deliberately degraded candidate, scores both, and enforces the regression "
            "and release gates.\n\n"
            "[dim]Everything runs offline with the deterministic mock provider. "
            "Results characterise the evaluation system, not any language model.[/dim]",
            border_style="cyan",
        )
    )

    # 1. Scenarios
    console.print("\n[bold]1/7[/bold] Generating scenarios")
    built = generate_suite(count=count, seed=seed, config=config, name="core")
    save_suite(built, config.paths.generated_scenarios)
    validation = validate_suite(built.scenarios, expected_count=count)
    if not validation.ok:
        for problem in validation.suite_errors:
            console.print(f"[red]{problem}[/red]")
        raise typer.Exit(EXIT_USER_ERROR)
    console.print(
        f"   {len(built.scenarios)} scenarios, "
        f"{len(validation.category_distribution)} categories, all valid"
    )

    # 2-3. Baseline and candidate
    runs_done = {}
    for step, (label, profile) in enumerate(
        (("baseline", "baseline"), ("candidate", "candidate")), start=2
    ):
        console.print(f"\n[bold]{step}/7[/bold] Running {label} agent")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"{label} sessions", total=len(built.scenarios))

            def advance(done: int, _total: int, _scenario: str, _task: Any = task) -> None:
                progress.update(_task, completed=done)

            runs_done[label] = run_evaluation(
                scenarios=built.scenarios,
                config=config,
                store=store,
                label=label,
                profile=profile,
                provider_name="mock",
                seed=seed,
                suite_id=built.suite_id,
                progress=advance,
            )
        result = runs_done[label]
        console.print(
            f"   pass rate {result.metrics.get('pass_rate'):.1%}, "
            f"score {result.metrics.get('overall_score'):.3f}, "
            f"critical failures {result.critical_failure_count}"
        )

    baseline_result = runs_done["baseline"]
    candidate_result = runs_done["candidate"]

    # 4. Comparison
    console.print("\n[bold]4/7[/bold] Comparing candidate against baseline")
    comparison = compare_runs(
        baseline_result.metrics,
        candidate_result.metrics,
        config,
        baseline_result.session_summaries,
        candidate_result.session_summaries,
    )
    console.print(
        f"   {len(comparison.regressions)} metric(s) beyond tolerance; "
        f"gate {'passed' if comparison.gate_passed else 'FAILED'}"
    )
    for item in comparison.regressions[:6]:
        console.print(f"   [red]-[/red] {item}")

    # 5. Alignment (only if annotations exist)
    console.print("\n[bold]5/7[/bold] Checking evaluator alignment")
    composition: dict[str, int] = {}
    for scenario in built.scenarios:
        composition[scenario.category.value] = composition.get(scenario.category.value, 0) + 1

    alignments = {}
    for label, result in runs_done.items():
        annotations = store.get_annotations(result.run_id, blind_only=True)
        alignments[label] = (
            build_alignment_report(result.run_id, annotations, result.session_summaries)
            if annotations
            else None
        )
        console.print(f"   {label}: {len(annotations)} blind annotation(s)")
    if not any(alignments.values()):
        console.print(
            "   [dim]No annotations yet. Collect them with `evalforge annotate` to "
            "populate the alignment section.[/dim]"
        )

    # 6. Reports
    console.print("\n[bold]6/7[/bold] Writing reports")
    written: list[Path] = []
    for label, result in runs_done.items():
        built_report = build_report(
            run_summary=result.summary,
            metrics=result.metrics,
            sessions=result.session_summaries,
            results=result.results,
            config=config,
            comparison=comparison if label == "candidate" else None,
            alignment=alignments[label],
            scenario_composition=composition,
        )
        result.summary.release_decision = built_report.decision
        store.save_run(result.summary)
        markdown_path, json_path = write_report(
            built_report, config, config.paths.reports_dir, stem=f"release_{label}"
        )
        written.extend([markdown_path, json_path])
        if alignments[label] is not None:
            written.append(
                write_alignment(
                    alignments[label], config.paths.reports_dir, stem=f"alignment_{label}"
                )
            )
        console.print(
            f"   {label}: [{_decision_style(built_report.decision)}]"
            f"{built_report.decision.value.upper()}[/]"
        )

    comparison_paths = write_comparison(comparison, config.paths.reports_dir)
    written.extend(comparison_paths)

    # 7. Summary
    console.print("\n[bold]7/7[/bold] Done\n")
    table = Table(title="Demonstration result")
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Candidate", justify="right")
    table.add_column("Change", justify="right")
    for name in (
        "pass_rate",
        "overall_score",
        "context_retention",
        "instruction_adherence",
        "tool_argument_accuracy",
        "prompt_injection_resistance",
        "critical_failure_count",
    ):
        base_value = baseline_result.metrics.get(name)
        cand_value = candidate_result.metrics.get(name)
        table.add_row(
            name.replace("_", " "),
            f"{base_value:.4f}",
            f"{cand_value:.4f}",
            f"{cand_value - base_value:+.4f}",
        )
    console.print(table)

    console.print("\n[bold]Output locations[/bold]")
    console.print(f"  scenarios   {config.paths.generated_scenarios}")
    console.print(f"  runs + traces {config.paths.runs}")
    console.print(f"  database    {store.database_path}")
    console.print(f"  reports     {config.paths.reports_dir}")
    for path in written:
        console.print(f"    [dim]{path.name}[/dim]")

    console.print("\n[bold]Run ids[/bold]")
    console.print(f"  baseline  [cyan]{baseline_result.run_id}[/cyan]")
    console.print(f"  candidate [cyan]{candidate_result.run_id}[/cyan]")

    decision_line = (
        "[bold red]Regression gate FAILED[/bold red] — the candidate degraded beyond "
        "tolerance, which is exactly what this demonstration is built to detect."
        if not comparison.gate_passed
        else "[bold green]Regression gate passed[/bold green]"
    )
    console.print(f"\n{decision_line}")
    console.print(
        f"\nExplore it with: [dim]evalforge dashboard[/dim] or "
        f"[dim]evalforge inspect --run-id {candidate_result.run_id} --failures-only[/dim]"
    )

    if fail_on_regression and not comparison.gate_passed:
        raise typer.Exit(EXIT_REGRESSION)


def entrypoint() -> None:
    """Console-script entry point with top-level error handling."""
    try:
        app()
    except RegressionGateError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(EXIT_REGRESSION)
    except EvalForgeError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(EXIT_USER_ERROR)


if __name__ == "__main__":
    entrypoint()
