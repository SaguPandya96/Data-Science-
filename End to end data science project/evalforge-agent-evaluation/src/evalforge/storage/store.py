"""Persistence: SQLite index plus JSONL traces.

The split is explained in ``docs/ARCHITECTURE.md`` §5. In short: indexed metadata goes
to SQLite because the questions asked of it are relational, and full traces go to JSONL
because they are ragged documents that SQL cannot query usefully but ``grep`` can.

Write ordering matters. The trace is written *before* the index row, so a crash leaves
an orphaned trace file (harmless, and recoverable by re-indexing) rather than an index
row pointing at a trace that does not exist (corrupt, and silently wrong).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evalforge.exceptions import RunNotFoundError, StorageError
from evalforge.logging_config import get_logger
from evalforge.schemas.annotation import HumanAnnotation
from evalforge.schemas.evaluation import EvaluationResult, RunSummary, SessionSummary
from evalforge.schemas.scenario import Scenario
from evalforge.schemas.trace import SessionTrace

logger = get_logger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class RunStore:
    """Read/write access to one evaluation data directory.

    Args:
        root: Directory holding run subdirectories, typically
            ``data/demonstration_runs``.
        database_name: SQLite filename inside ``root``.
    """

    def __init__(self, root: Path, database_name: str = "evalforge.db") -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.database_path = root / database_name
        self._ensure_schema()

    # ------------------------------------------------------------------ plumbing

    def _ensure_schema(self) -> None:
        """Create tables and indexes if they do not exist."""
        try:
            with self.connect() as connection:
                connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        except sqlite3.Error as exc:
            raise StorageError(
                f"Could not initialise database at {self.database_path}: {exc}"
            ) from exc

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection with row access by name, committing on clean exit."""
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except sqlite3.Error as exc:
            connection.rollback()
            raise StorageError(f"Database operation failed: {exc}") from exc
        finally:
            connection.close()

    def run_dir(self, run_id: str) -> Path:
        """Directory holding one run's artifacts."""
        return self.root / run_id

    def traces_dir(self, run_id: str) -> Path:
        """Directory holding one run's session traces."""
        return self.run_dir(run_id) / "traces"

    # -------------------------------------------------------------------- writes

    def save_scenarios(self, run_id: str, scenarios: list[Scenario]) -> Path:
        """Store the exact scenarios a run executed.

        Copied into the run directory rather than referenced, so a run stays
        reproducible even if the generator or its templates change later.
        """
        directory = self.run_dir(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "scenarios.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for scenario in scenarios:
                handle.write(scenario.model_dump_json())
                handle.write("\n")
        return path

    def save_trace(self, trace: SessionTrace) -> Path:
        """Write one session trace as a single-record JSONL file.

        One file per session gives O(1) lookup for the trace explorer, while the JSONL
        shape keeps ``grep`` and ``jq`` working across ``traces/*.jsonl``.
        """
        directory = self.traces_dir(trace.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{trace.session_id}.jsonl"
        path.write_text(trace.model_dump_json() + "\n", encoding="utf-8")
        return path

    def save_run(self, summary: RunSummary) -> None:
        """Insert or replace a run record."""
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, label, suite_id, provider_name, model_name, prompt_version,
                    agent_version, started_at, completed_at, session_count,
                    release_decision, config_digest, metrics_json, metadata_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    summary.run_id,
                    summary.label,
                    summary.suite_id,
                    summary.provider_name,
                    summary.model_name,
                    summary.prompt_version,
                    summary.agent_version,
                    summary.started_at.isoformat(),
                    summary.completed_at.isoformat() if summary.completed_at else None,
                    summary.session_count,
                    summary.release_decision.value,
                    summary.config_digest,
                    json.dumps(summary.metrics),
                    json.dumps(summary.metadata, default=str),
                ),
            )

    def save_session(self, summary: SessionSummary) -> None:
        """Insert or replace one session summary."""
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO sessions (
                    session_id, run_id, scenario_id, scenario_category, scenario_difficulty,
                    turn_count, overall_score, task_completion_score, context_retention_score,
                    instruction_adherence_score, tool_reliability_score, recovery_score,
                    consistency_score, efficiency_score, safety_score, total_latency,
                    total_tokens, estimated_cost, passed, critical_failure_count, summary_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    summary.session_id,
                    summary.run_id,
                    summary.scenario_id,
                    summary.scenario_category,
                    summary.scenario_difficulty,
                    summary.turn_count,
                    summary.overall_score,
                    summary.task_completion_score,
                    summary.context_retention_score,
                    summary.instruction_adherence_score,
                    summary.tool_reliability_score,
                    summary.recovery_score,
                    summary.consistency_score,
                    summary.efficiency_score,
                    summary.safety_score,
                    summary.total_latency,
                    summary.total_tokens,
                    summary.estimated_cost,
                    int(summary.passed),
                    len(summary.critical_failures),
                    summary.model_dump_json(),
                ),
            )

    def save_evaluations(self, results: list[EvaluationResult]) -> None:
        """Bulk-insert evaluation results."""
        if not results:
            return
        rows = [
            (
                result.evaluation_id,
                result.run_id,
                result.session_id,
                result.scenario_id,
                result.turn_index,
                result.evaluator_name,
                result.evaluator_kind,
                result.evaluation_level.value,
                result.dimension.value,
                result.score,
                int(result.passed),
                result.confidence,
                result.failure_category.value,
                result.severity.value,
                result.model_dump_json(),
            )
            for result in results
        ]
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO evaluations (
                    evaluation_id, run_id, session_id, scenario_id, turn_index,
                    evaluator_name, evaluator_kind, evaluation_level, dimension, score,
                    passed, confidence, failure_category, severity, result_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )

    def save_annotation(self, annotation: HumanAnnotation) -> None:
        """Insert or replace one human annotation."""
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO annotations (
                    annotation_id, run_id, session_id, scenario_id, annotator_id,
                    overall_pass, severity, blind, created_at, annotation_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    annotation.annotation_id,
                    annotation.run_id,
                    annotation.session_id,
                    annotation.scenario_id,
                    annotation.annotator_id,
                    int(annotation.overall_pass),
                    annotation.severity.value,
                    int(annotation.blind),
                    annotation.created_at.isoformat(),
                    annotation.model_dump_json(),
                ),
            )

    # --------------------------------------------------------------------- reads

    def list_runs(self) -> list[RunSummary]:
        """Every stored run, newest first."""
        with (
            self.connect() as connection,
            closing(connection.execute("SELECT * FROM runs ORDER BY started_at DESC")) as cursor,
        ):
            return [_row_to_run(row) for row in cursor.fetchall()]

    def get_run(self, run_id: str) -> RunSummary:
        """Load one run.

        Raises:
            RunNotFoundError: If no such run exists.
        """
        with (
            self.connect() as connection,
            closing(connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))) as cursor,
        ):
            row = cursor.fetchone()
        if row is None:
            raise RunNotFoundError(run_id)
        return _row_to_run(row)

    def run_exists(self, run_id: str) -> bool:
        """Whether a run id is present in the index."""
        with (
            self.connect() as connection,
            closing(connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,))) as cursor,
        ):
            return cursor.fetchone() is not None

    def get_sessions(self, run_id: str) -> list[SessionSummary]:
        """Every session summary for a run, worst score first.

        Ordering by score means the failures a reader most wants are at the top of any
        listing without a second sort.
        """
        with (
            self.connect() as connection,
            closing(
                connection.execute(
                    "SELECT summary_json FROM sessions WHERE run_id = ? ORDER BY overall_score ASC",
                    (run_id,),
                )
            ) as cursor,
        ):
            return [SessionSummary.model_validate_json(row["summary_json"]) for row in cursor]

    def get_session(self, session_id: str) -> SessionSummary | None:
        """Load one session summary."""
        with (
            self.connect() as connection,
            closing(
                connection.execute(
                    "SELECT summary_json FROM sessions WHERE session_id = ?", (session_id,)
                )
            ) as cursor,
        ):
            row = cursor.fetchone()
        return SessionSummary.model_validate_json(row["summary_json"]) if row else None

    def get_evaluations(self, run_id: str, session_id: str | None = None) -> list[EvaluationResult]:
        """Evaluation results for a run, optionally narrowed to one session."""
        query = "SELECT result_json FROM evaluations WHERE run_id = ?"
        params: list[Any] = [run_id]
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        with self.connect() as connection, closing(connection.execute(query, params)) as cursor:
            return [EvaluationResult.model_validate_json(row["result_json"]) for row in cursor]

    def get_trace(self, run_id: str, session_id: str) -> SessionTrace:
        """Load one session trace.

        Raises:
            StorageError: If the trace file is missing or unreadable.
        """
        path = self.traces_dir(run_id) / f"{session_id}.jsonl"
        if not path.exists():
            raise StorageError(f"No trace stored for session {session_id!r} in run {run_id!r}.")
        line = path.read_text(encoding="utf-8").strip()
        if not line:
            raise StorageError(f"Trace file for session {session_id!r} is empty.")
        return SessionTrace.model_validate_json(line.splitlines()[0])

    def iter_traces(self, run_id: str) -> Iterator[SessionTrace]:
        """Stream every trace in a run, in filename order."""
        directory = self.traces_dir(run_id)
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.jsonl")):
            content = path.read_text(encoding="utf-8").strip()
            if content:
                yield SessionTrace.model_validate_json(content.splitlines()[0])

    def get_scenarios(self, run_id: str) -> list[Scenario]:
        """The scenarios a run executed.

        Raises:
            StorageError: If the run directory has no scenario file.
        """
        path = self.run_dir(run_id) / "scenarios.jsonl"
        if not path.exists():
            raise StorageError(f"No scenarios stored for run {run_id!r}.")
        scenarios: list[Scenario] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    scenarios.append(Scenario.model_validate_json(line))
        return scenarios

    def get_scenario(self, run_id: str, scenario_id: str) -> Scenario | None:
        """One scenario from a run."""
        for scenario in self.get_scenarios(run_id):
            if scenario.scenario_id == scenario_id:
                return scenario
        return None

    def get_annotations(
        self, run_id: str | None = None, blind_only: bool = True
    ) -> list[HumanAnnotation]:
        """Stored annotations, blind ones only by default.

        The default is deliberate: pooling non-blind annotations into agreement
        statistics would inflate them, since the annotator already saw the automated
        verdict they are being compared against.
        """
        query = "SELECT annotation_json FROM annotations WHERE 1=1"
        params: list[Any] = []
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if blind_only:
            query += " AND blind = 1"
        with self.connect() as connection, closing(connection.execute(query, params)) as cursor:
            return [HumanAnnotation.model_validate_json(row["annotation_json"]) for row in cursor]

    def annotated_session_ids(self, run_id: str, annotator_id: str) -> set[str]:
        """Sessions this annotator has already labelled, so the UI can skip them."""
        with (
            self.connect() as connection,
            closing(
                connection.execute(
                    "SELECT session_id FROM annotations WHERE run_id = ? AND annotator_id = ?",
                    (run_id, annotator_id),
                )
            ) as cursor,
        ):
            return {row["session_id"] for row in cursor}

    def delete_run(self, run_id: str) -> None:
        """Remove a run's index rows. Trace files are left on disk deliberately.

        Deleting a directory of artifacts is not something a storage helper should do
        implicitly; the caller can remove the run directory if that is genuinely wanted.
        """
        with self.connect() as connection:
            connection.execute("DELETE FROM evaluations WHERE run_id = ?", (run_id,))
            connection.execute("DELETE FROM sessions WHERE run_id = ?", (run_id,))
            connection.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        logger.info("run_index_deleted", run_id=run_id)


def _row_to_run(row: sqlite3.Row) -> RunSummary:
    """Rehydrate a run record from its indexed columns and JSON payload."""
    from evalforge.schemas.common import ReleaseDecision

    return RunSummary(
        run_id=row["run_id"],
        label=row["label"],
        suite_id=row["suite_id"],
        provider_name=row["provider_name"],
        model_name=row["model_name"],
        prompt_version=row["prompt_version"],
        agent_version=row["agent_version"],
        started_at=_parse_time(row["started_at"]),
        completed_at=_parse_time(row["completed_at"]) if row["completed_at"] else None,
        session_count=row["session_count"],
        metrics=json.loads(row["metrics_json"]),
        release_decision=ReleaseDecision(row["release_decision"]),
        config_digest=row["config_digest"],
        metadata=json.loads(row["metadata_json"]),
    )


def _parse_time(value: str) -> datetime:
    """Parse a stored ISO timestamp, defaulting to UTC when naive."""
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
