"""CLI behaviour, exit codes, and headless dashboard rendering."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalforge.cli import EXIT_REGRESSION, EXIT_USER_ERROR, app

pytestmark = pytest.mark.integration

runner = CliRunner()

DASHBOARD = Path(__file__).resolve().parents[2] / "dashboard"


class TestCliBasics:
    """The commands a user reaches for first."""

    def test_version(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "EvalForge" in result.output

    def test_help_lists_every_documented_command(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for command in (
            "generate",
            "validate-scenarios",
            "run",
            "evaluate",
            "inspect",
            "compare",
            "report",
            "export",
            "dashboard",
            "annotate",
            "demo",
        ):
            assert command in result.output, f"{command} missing from help"

    def test_bare_invocation_shows_help(self) -> None:
        result = runner.invoke(app, [])
        assert result.exit_code == 0


class TestCliErrorHandling:
    """Expected errors must be sentences, not stack traces."""

    def test_unknown_run_id_is_a_clean_error(self) -> None:
        result = runner.invoke(app, ["inspect", "--run-id", "run_does_not_exist"])
        assert result.exit_code == EXIT_USER_ERROR
        assert "Traceback" not in result.output
        assert "Error" in result.output

    def test_unknown_category_lists_the_valid_ones(self) -> None:
        result = runner.invoke(app, ["run", "--category", "not_a_category"])
        assert result.exit_code == EXIT_USER_ERROR
        assert "Traceback" not in result.output

    def test_too_small_a_count_is_rejected(self) -> None:
        result = runner.invoke(app, ["generate", "--count", "2"])
        assert result.exit_code == EXIT_USER_ERROR
        assert "at least 8" in result.output

    def test_unknown_export_format_is_rejected(self) -> None:
        result = runner.invoke(app, ["export", "--run-id", "r", "--format", "parquet"])
        assert result.exit_code == EXIT_USER_ERROR


class TestCliWorkflow:
    """A full generate → run → report → compare cycle through the CLI."""

    @pytest.fixture
    def workspace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Point EvalForge at a temporary data directory."""
        monkeypatch.setenv("EVALFORGE_DATA_DIR", str(tmp_path / "data"))
        from evalforge.config import clear_config_cache

        clear_config_cache()
        return tmp_path

    def test_generate_and_validate(self, workspace: Path) -> None:
        generated = runner.invoke(app, ["generate", "--count", "16", "--seed", "42", "--force"])
        assert generated.exit_code == 0, generated.output
        assert "16 scenarios" in generated.output

        validated = runner.invoke(app, ["validate-scenarios"])
        assert validated.exit_code == 0, validated.output
        assert "All scenarios valid" in validated.output

    def test_run_report_and_compare(self, workspace: Path) -> None:
        assert runner.invoke(app, ["generate", "--count", "16", "--force"]).exit_code == 0

        baseline = runner.invoke(app, ["run", "--label", "baseline", "--profile", "baseline"])
        assert baseline.exit_code == 0, baseline.output

        candidate = runner.invoke(app, ["run", "--label", "candidate", "--profile", "candidate"])
        assert candidate.exit_code == 0, candidate.output

        listed = runner.invoke(app, ["runs"])
        assert listed.exit_code == 0
        assert "baseline" in listed.output and "candidate" in listed.output

        from evalforge.config import load_config
        from evalforge.storage.store import RunStore

        config = load_config()
        store = RunStore(config.paths.runs, config.storage.database_name)
        runs = {item.label: item.run_id for item in store.list_runs()}

        reported = runner.invoke(app, ["report", "--run-id", runs["baseline"]])
        assert reported.exit_code == 0, reported.output

        compared = runner.invoke(
            app,
            ["compare", "--baseline", runs["baseline"], "--candidate", runs["candidate"]],
        )
        # A degraded candidate must trip the gate and exit non-zero.
        assert compared.exit_code == EXIT_REGRESSION, compared.output
        assert "Regression gate FAILED" in compared.output

        tolerated = runner.invoke(
            app,
            [
                "compare",
                "--baseline",
                runs["baseline"],
                "--candidate",
                runs["candidate"],
                "--no-fail-on-regression",
            ],
        )
        assert tolerated.exit_code == 0

        exported = runner.invoke(app, ["export", "--run-id", runs["baseline"], "--format", "csv"])
        assert exported.exit_code == 0
        assert "Exported" in exported.output

    def test_inspect_drills_into_a_session(self, workspace: Path) -> None:
        assert runner.invoke(app, ["generate", "--count", "16", "--force"]).exit_code == 0
        assert runner.invoke(app, ["run", "--label", "baseline"]).exit_code == 0

        from evalforge.config import load_config
        from evalforge.storage.store import RunStore

        config = load_config()
        store = RunStore(config.paths.runs, config.storage.database_name)
        run_id = store.list_runs()[0].run_id
        sessions = store.get_sessions(run_id)

        listing = runner.invoke(app, ["inspect", "--run-id", run_id])
        assert listing.exit_code == 0

        detail = runner.invoke(
            app,
            ["inspect", "--run-id", run_id, "--scenario-id", sessions[0].scenario_id],
        )
        assert detail.exit_code == 0
        assert "Turn 0" in detail.output


@pytest.mark.slow
class TestDashboardRenders:
    """Every page must execute without raising.

    ``AppTest`` runs the page script headlessly and surfaces exceptions, which an HTTP
    check cannot: Streamlit serves its shell before the script runs, so a page that
    crashes still returns 200.
    """

    @pytest.fixture(scope="class", autouse=True)
    def demo_data(self, tmp_path_factory) -> None:
        """A small stored run so the pages have something to show."""
        from evalforge.config import load_config
        from evalforge.orchestration.pipeline import run_evaluation
        from evalforge.scenarios.generator import generate_scenarios
        from evalforge.storage.store import RunStore

        config = load_config()
        if RunStore(config.paths.runs, config.storage.database_name).list_runs():
            return

        store = RunStore(config.paths.runs, config.storage.database_name)
        scenarios = generate_scenarios(count=8, seed=42, config=config)
        run_evaluation(scenarios, config, store, label="baseline", profile="baseline")
        run_evaluation(scenarios, config, store, label="candidate", profile="candidate")

    @pytest.mark.parametrize(
        "page",
        [
            "app.py",
            "pages/1_Failure_Analysis.py",
            "pages/2_Conversation_Length.py",
            "pages/3_Tool_Reliability.py",
            "pages/4_Run_Comparison.py",
            "pages/5_Evaluator_Alignment.py",
            "pages/6_Human_Annotation.py",
            "pages/7_Trace_Explorer.py",
        ],
    )
    def test_page_renders_without_exception(self, page: str) -> None:
        pytest.importorskip("streamlit")
        from streamlit.testing.v1 import AppTest

        target = DASHBOARD / page
        assert target.exists(), f"missing dashboard page: {page}"

        harness = AppTest.from_file(str(target), default_timeout=180)
        harness.run()
        assert not harness.exception, (
            f"{page} raised: {[item.message for item in harness.exception]}"
        )
