-- EvalForge SQLite index.
--
-- This database holds only *queryable metadata*. Full session traces live alongside it
-- as JSONL (see docs/ARCHITECTURE.md §5): traces are deep, ragged documents that SQL
-- cannot usefully query anyway, while "every failing session in this run with a critical
-- safety failure, ordered by score" is exactly a relational query.
--
-- Every table stores the validated Pydantic payload in a *_json column as well as the
-- indexed columns, so a record can always be rehydrated into its typed model without a
-- lossy round trip through the flattened columns.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    label             TEXT NOT NULL DEFAULT '',
    suite_id          TEXT NOT NULL DEFAULT '',
    provider_name     TEXT NOT NULL DEFAULT 'mock',
    model_name        TEXT NOT NULL DEFAULT 'mock',
    prompt_version    TEXT NOT NULL DEFAULT 'v1',
    agent_version     TEXT NOT NULL DEFAULT 'v1',
    started_at        TEXT NOT NULL,
    completed_at      TEXT,
    session_count     INTEGER NOT NULL DEFAULT 0,
    release_decision  TEXT NOT NULL DEFAULT 'fail',
    -- Ties a stored run to the exact rubric and thresholds that scored it (ADR-005).
    config_digest     TEXT NOT NULL DEFAULT '',
    metrics_json      TEXT NOT NULL DEFAULT '{}',
    metadata_json     TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id                  TEXT PRIMARY KEY,
    run_id                      TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    scenario_id                 TEXT NOT NULL,
    scenario_category           TEXT NOT NULL DEFAULT '',
    scenario_difficulty         TEXT NOT NULL DEFAULT '',
    turn_count                  INTEGER NOT NULL DEFAULT 0,
    overall_score               REAL NOT NULL DEFAULT 0,
    task_completion_score       REAL NOT NULL DEFAULT 0,
    context_retention_score     REAL NOT NULL DEFAULT 0,
    instruction_adherence_score REAL NOT NULL DEFAULT 0,
    tool_reliability_score      REAL NOT NULL DEFAULT 0,
    recovery_score              REAL NOT NULL DEFAULT 0,
    consistency_score           REAL NOT NULL DEFAULT 0,
    efficiency_score            REAL NOT NULL DEFAULT 0,
    safety_score                REAL NOT NULL DEFAULT 0,
    total_latency               REAL NOT NULL DEFAULT 0,
    total_tokens                INTEGER NOT NULL DEFAULT 0,
    estimated_cost              REAL NOT NULL DEFAULT 0,
    passed                      INTEGER NOT NULL DEFAULT 0,
    critical_failure_count      INTEGER NOT NULL DEFAULT 0,
    summary_json                TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sessions_run        ON sessions(run_id);
CREATE INDEX IF NOT EXISTS idx_sessions_category   ON sessions(run_id, scenario_category);
CREATE INDEX IF NOT EXISTS idx_sessions_difficulty ON sessions(run_id, scenario_difficulty);
CREATE INDEX IF NOT EXISTS idx_sessions_length     ON sessions(run_id, turn_count);
CREATE INDEX IF NOT EXISTS idx_sessions_passed     ON sessions(run_id, passed);

CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id    TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    session_id       TEXT NOT NULL,
    scenario_id      TEXT NOT NULL,
    turn_index       INTEGER,
    evaluator_name   TEXT NOT NULL,
    evaluator_kind   TEXT NOT NULL DEFAULT 'deterministic',
    evaluation_level TEXT NOT NULL DEFAULT 'session',
    dimension        TEXT NOT NULL,
    score            REAL NOT NULL DEFAULT 0,
    passed           INTEGER NOT NULL DEFAULT 0,
    confidence       REAL NOT NULL DEFAULT 1,
    failure_category TEXT NOT NULL DEFAULT 'none',
    severity         TEXT NOT NULL DEFAULT 'info',
    result_json      TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_eval_run      ON evaluations(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_session  ON evaluations(session_id);
CREATE INDEX IF NOT EXISTS idx_eval_failure  ON evaluations(run_id, failure_category);
CREATE INDEX IF NOT EXISTS idx_eval_kind     ON evaluations(run_id, evaluator_kind);

CREATE TABLE IF NOT EXISTS annotations (
    annotation_id   TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    scenario_id     TEXT NOT NULL,
    annotator_id    TEXT NOT NULL,
    overall_pass    INTEGER NOT NULL DEFAULT 1,
    severity        TEXT NOT NULL DEFAULT 'info',
    -- Only blind annotations are eligible for agreement statistics; one collected after
    -- the annotator saw automated scores is contaminated and must be excluded, not
    -- silently pooled.
    blind           INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL,
    annotation_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_annotations_run     ON annotations(run_id);
CREATE INDEX IF NOT EXISTS idx_annotations_session ON annotations(session_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_annotations_unique
    ON annotations(session_id, annotator_id);
