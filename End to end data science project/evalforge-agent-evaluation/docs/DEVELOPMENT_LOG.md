# Development Log

A running record of what was built, why, what was rejected, and what remains weak.
Entries are chronological and correspond to commits on
`feature/evalforge-agent-evaluation`.

---

## M1 — Package scaffolding and tooling

**Commit:** `chore: initialize EvalForge package and development tooling`

**Problem.** The project nests inside an existing multi-project repository
(`Data-Science-`) that has no Python packaging, no linting and no CI. EvalForge needed
modern tooling without touching any sibling project.

**Decisions.**
- `src/` layout with Hatchling. The `src/` layout was chosen so tests import the
  *installed* package rather than the working directory, which is what makes the CI
  install step meaningful.
- Ruff for both lint and format, replacing the Black + isort + flake8 stack. One tool,
  one config block, and it is fast enough to run on every commit.
- mypy with `disallow_untyped_defs` on `src/` only. Tests are deliberately excluded;
  requiring full annotations on fixtures adds noise without catching defects.
- Python 3.11 pinned as the floor for `StrEnum` and the `X | Y` union syntax used
  throughout the schemas.

**Alternatives considered.** Poetry (heavier, and its lock file would be the only one in
the repository); flat layout (rejected — it hides packaging errors until publish time).

**Tests added.** None yet; this milestone is configuration only.

**Limitations.** The repository has no root-level Python tooling, so EvalForge's config
is self-contained and does not apply to sibling projects. That is intentional.

---

## M2 — Architecture, failure taxonomy and ADRs

**Commit:** `docs: define evaluation architecture and failure taxonomy`

**Problem.** The scoring rules, the failure vocabulary and the reasons behind both needed
to exist *before* the evaluators, otherwise the taxonomy would end up being whatever the
code happened to emit.

**Decisions.**
- Session-level evaluation as the primary unit (ADR-001), because context retention,
  goal drift and cascading errors are properties of sessions and are invisible to
  per-response scoring.
- Deterministic and judge scoring kept structurally separate (ADR-002). The rule adopted:
  *if a reliable expected value exists, a deterministic evaluator must be used.*
- An offline mock provider as a mandatory first-class component (ADR-003), so CI needs no
  secrets and the regression gate contains no sampling noise.
- Critical failures as categorical release blockers (ADR-004), because a weighted average
  lets a single obeyed prompt injection average away to a 0.97.
- All policy in versioned YAML with rationale strings (ADR-005).
- 30 failure categories defined, each mapped to a dimension and an owning evaluator. Six
  are critical.

**Alternatives considered.** Judging everything with an LLM (rejected: makes exact checks
probabilistic and breaks offline operation); weighting safety at 40% instead of using
overrides (rejected: arithmetic still averages a single incident away).

**Tests added.** None yet — the taxonomy is exercised by evaluator tests in M5.

**Limitations.** The taxonomy is a design artifact; whether the categories are the *right*
carve-up is only testable once real disagreement data exists. The alignment analysis in
M9 is the first check on that.

---

## M3 — Typed schemas and configuration system

**Commit:** `feat: add typed schemas and configuration system`

**Problem.** Traces are written by one component and read by four others, months apart.
Passing dicts would let a field rename break scoring silently.

**Decisions.**
- Pydantic v2 models with `extra="forbid"` at every boundary. A typo'd key is an error,
  not a silently ignored field.
- `Fact` and `Constraint` are frozen. Scenario contracts are read by many evaluators
  concurrently and must not be mutable.
- Constraints carry `turn_added` / `turn_removed` and an `is_active_at` method, so
  "was this constraint live at turn 12" is a model question, not an evaluator's
  re-derivation.
- `Scenario.final_fact_values()` resolves override semantics centrally — later turns win.
  This is the exact logic the "reduce the budget but keep the date" case depends on, and
  it belongs in one place.
- `EvaluationResult.is_critical_failure` requires *both* critical severity and membership
  in the critical category set, so a heuristic cannot gate a release alone.
- Config layered YAML → env → CLI, hashed into a `config_digest` stored on every run.
- Rubric weights validated to sum to 1.0 at load; failure is a `ConfigurationError`
  rather than silent renormalisation.
- `DeterministicIdFactory` alongside random ids, so committed demo artifacts are
  byte-stable across regeneration.

**Alternatives considered.** Dataclasses plus manual validation (rejected: no serialisation
story, no boundary enforcement); a single flat `Config` object (rejected: nested models
give per-section validation and much better error messages).

**Tests added.** Deferred to M13's schema round-trip suite.

**Limitations.** `extra="forbid"` makes old traces unreadable if a field is ever removed.
Acceptable pre-1.0; a real migration story would be needed before that.
