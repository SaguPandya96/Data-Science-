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

---

## M4 — Productivity tools and fault injection

**Commit:** `feat: implement productivity tools and failure injection`

**Problem.** Recovery behaviour is only measurable when failures are reproducible, and
tool contracts have to be typed at both ends or a corrupted payload is indistinguishable
from a bug in the harness.

**Decisions.**
- `BaseTool.invoke` owns validation, approval enforcement, latency, fault injection and
  logging, so no individual tool can forget any of it.
- Faults are applied *around* real execution, not inside it: a tool stays a
  straightforward simulated action, and a new fault type applies to all eight at once.
- Output corruption happens *after* the tool's own valid output. Ordering matters — it
  means a `MISSING_FIELD` fault yields a payload the agent must notice rather than an
  exception whose contents it never sees.
- `search_documents` returns document bodies verbatim, injection payloads included.
  Sanitising there would make the whole prompt-injection category untestable.
- Approval is enforced by the executor, never the prompt. A gated tool is unreachable
  without a recorded grant.

**Alternatives considered.** Per-tool failure branches (rejected: eight copies of the
same logic, guaranteed to drift); sanitising retrieved content (rejected: removes the
threat the tests exist to measure).

**Tests added.** 32 tool tests covering validation, approval gating, arithmetic, all
eleven fault types, determinism of latency and injection, and that nothing is ever sent.

**Limitations.** Twelve fictional documents; retrieval quality is not what is under test.

---

## M5 — Deterministic mock agent and trace collection

**Commit:** `feat: add deterministic mock agent and trace collection`

**Problem.** CI needs an agent that can behave competently *and* fail in named ways on
demand, reproducibly, with no credentials.

**Decisions.**
- `ModelResponse` carries structured state — `remembered_facts`, `active_constraints`,
  `request_approval`, `refused_injection` — rather than requiring prose parsing. This is
  what lets the trace record the model's *understanding* separately from its *output*,
  which is the cleanest possible context-retention signal.
- Every degradation is a seeded Bernoulli trial keyed on
  `(run_seed, scenario_id, turn_index, aspect)`, so a decision depends only on its own
  coordinates — reproducible under partial or parallel execution.
- The agent never reads the scenario contract. Scenario metadata reaches the *provider*
  (standing in for comprehension) and the *evaluators* (scoring afterwards), never the
  executor between them.
- A forgotten prohibition must actually produce the forbidden content, otherwise the
  violation is invisible and the simulation is dishonest.

**Alternatives considered.** Record/replay against a real model (rejected: cassettes are
valid for one model version, cannot be regenerated without credentials, and cannot
produce *controlled* degradation — you cannot ask a real model to lose context after
exactly turn 10); a small local open-weights model (rejected: multi-GB, slow, still
non-deterministic across hardware).

**Tests added.** Determinism of agent execution; behaviour separation across the four
profiles.

**Limitations.** The mock cannot exhibit a failure mode nobody implemented. Partially
mitigated by the `perfect`/`broken` bounds; not eliminated.

---

## M6 — Adversarial scenario generator

**Commit:** `feat: build adversarial multi-turn scenario generator`

**Problem.** 150 hand-written multi-turn contracts would be slow, inconsistent and
irreproducible. And a generated suite fails in a specific way if the content pools are
thin: 150 lexically near-identical scenarios measure one case 150 times.

**Decisions.**
- Templates plus seeded variation across project, dates, budgets, constraint mix,
  distractor placement and fault type. 110 distinct scenario names from 150 scenarios.
- Difficulty is *derived* from measurable pressure (length, facts, constraints,
  revisions, faults), not assigned by hand, so it stays consistent across categories.
- `_pad_to` forces exact conversation lengths. Lengths are load-bearing: the length-sweep
  analysis buckets by 5/10/15/20/30, and a suite of 13- and 27-turn conversations would
  smear those buckets.
- The validator enforces category-specific rules — an injection scenario with no payload,
  or a drift scenario with no distractor, is mislabelled rather than merely weak.

**Tests added.** Suite validation, determinism, exact length distribution, category
completeness.

**Limitations.** Generated scenarios are less nuanced than hand-written ones, and
"difficulty" has not been validated against what an agent actually finds hard.

---

## M7 — Deterministic evaluation engine

**Commit:** `feat: implement deterministic evaluation engine`

**Problem.** Twenty-one checks against one trace, each of which must be exact enough to
justify blocking a release.

**Decisions.**
- Normalisation lives in `evaluators/base.py` because comparison semantics are *policy*.
  "Is `$15,000` the same as `15000.00`?" must have one answer system-wide.
- Tool order is scored as an LCS ratio, not exact equality: a differently-interleaved but
  dependency-respecting order is not a defect.
- Argument matching compares only the *pinned* subset; pinning every field would measure
  prompt-template stability rather than correctness.
- Evaluators are grouped by failure domain rather than one file per class — nineteen
  single-class files would hide that all three tool evaluators share argument
  normalisation.
- An evaluator that raises is logged and skipped, not fatal: one buggy check should cost
  one dimension's coverage, not a 150-session run.

**Four false-positive sources found and fixed** while building the `perfect`-profile
control, each a genuine methodology bug:

1. Plan constraint lists contain the constraint *description*, which contains the banned
   phrase — an agent recording "do not include X" was scored as including X.
2. The summary tool's excluded-topic filter deleted its own constraints block, emptying
   the summary and triggering a spurious missing-section failure.
3. Flagging a conflict ("this conflicts with your instruction to exclude X") necessarily
   names X and was scored as a violation.
4. Semantic evaluators were failing at the deterministic threshold, manufacturing
   failures out of vocabulary difference alone.

**Tests added.** False-positive and false-negative controls; comparison primitives;
evidence requirements; severity policy.

**Limitations.** Brittle against paraphrase by design; contradiction detection only sees
tracked values; goal drift is deliberately conservative.

---

## M8 — Semantic and judge interfaces

**Commit:** `feat: add semantic and llm judge interfaces`

**Decisions.**
- Semantic evaluators report a score but **never assert a failure** — the default backend
  is lexical, and any threshold above zero labels correct paraphrase as failure.
- The judge is treated as an unreliable instrument: schema-validated output, required
  evidence, multi-sample median aggregation, recorded model and prompt version, and a
  hard cap at MAJOR severity so it can never gate a release.
- `MockJudge` injects seeded variance deliberately — without it, median-of-three is a
  no-op and the aggregation path is never genuinely exercised.
- A missing judge credential degrades to the mock rather than aborting a run whose
  deterministic results are the ones that actually gate the release.

**Tests added.** Judge determinism, provenance recording, sample aggregation, evidence
requirement, and that no judge result is ever critical.

---

## M9 — Storage, orchestration, analytics and reporting

**Commits:** `feat: add sqlite index and jsonl trace storage`,
`feat: add evaluation orchestration and cli`,
`feat: implement analytics and release readiness reports`

**Decisions.**
- SQLite for indexed metadata, JSONL for traces, with the reasoning and its cost recorded
  in ARCHITECTURE.md section 5. The trace is written before the index row, so a crash
  orphans a trace (recoverable) rather than dangling an index row (corrupt).
- `reevaluate_run` re-scores from stored traces without re-invoking the agent — the
  payoff of making the trace the single source of truth.
- Every statistical helper documents *why* it is the right tool for its question, and the
  module reports no p-values at all.
- Sessions are returned worst-score-first so failures surface without a second sort.
- Unmeasured release gates are reported as unmeasured rather than silently passed.

**A scoring bug found here.** Per-failure results are session-scoped, like roll-ups. That
meant aggregation averaged each failure into the basis *and* subtracted it as a penalty —
double-counting every defect, and dragging `tool_selection_accuracy` from 0.96 to 0.74.
Fixed by marking roll-ups explicitly in metadata.

**Tests added.** Storage round-trips, metric coverage, threshold and regression gate
behaviour in both directions, report honesty guarantees.

---

## M10 — Dashboard, annotation and alignment

**Commits:** `feat: build Streamlit dashboard and trace explorer`,
`feat: add human annotation and evaluator alignment`

**Decisions.**
- `dashboard/data_access.py` is a thin read layer; **no page computes a metric**. What is
  on screen is what the release report prints, from the same unit-tested code.
- The annotation interface hides automated scores until submission and records
  `blind=True`; non-blind annotations are excluded from agreement statistics entirely.
- Human-vs-human agreement is computed first and presented as the ceiling.
- `scripts/simulate_annotations.py` generates clearly-labelled **synthetic** annotations
  so the alignment pipeline is demonstrable offline. Every record carries
  `synthetic: true`, and the dashboard shows a prominent warning wherever they are used.
  The alternative — an alignment page that stays blank until someone hand-labels 60
  sessions — would leave a whole subsystem unexercised and untestable.

**A serialisation bug found here.** `@computed_field` on `EvaluationResult` and
`TokenUsage` meant both models serialised a field their own validators then rejected
under `extra="forbid"`. Every persisted record was unreadable. The pipeline "worked"
because it held results in memory; only *reading a run back* failed. Caught by executing
the dashboard pages headlessly with Streamlit's `AppTest`.

**Limitations.** The committed alignment figures describe the annotation simulator, not
human agreement, and say so everywhere they appear.

---

## M11 — Tests, CI and final validation

**Commits:** `test: expand adversarial and regression coverage`,
`ci: add EvalForge quality and regression workflow`,
`docs: finalize reproducible portfolio demonstration`

**Decisions.**
- Golden fixtures use **targeted** behaviour profiles built from `perfect` with one rate
  turned on, rather than the shipped `broken` profile. `broken` drifts 60% of the time
  and frequently abandons a turn before reaching the behaviour under test, so a fixture
  built on it tests the mock's dice rather than the evaluator.
- Dashboard pages are tested with `AppTest`, not an HTTP check: Streamlit serves its
  shell before the script runs, so a page that crashes still returns 200.
- CI asserts an **inverted** condition on the regression gate — a *passing* gate against
  the deliberately degraded candidate fails the build, because that means the gate
  stopped working.
- `no_args_is_help` was disabled on the CLI: it exits 2, colliding with the documented
  meaning of exit code 2 (regression gate failed).

**Two more real bugs found by these tests.**

1. Fact retention searched the *whole session transcript* for corroboration, so a fact
   stated correctly on turn 0 and lost by turn 20 still counted as retained — precisely
   the failure the evaluator exists to catch. Now the workspace is authoritative, with
   final-turn text as a fallback only for providers that report no structured state.
2. Total context loss was invisible: an empty `remembered_facts` was treated as "no
   update" rather than "everything forgotten", so the agent silently kept stale facts.
   Fixed by adding an explicit `state_reported` flag to distinguish "holding nothing"
   from "not reporting".

**Also fixed:** the approval gate was stripping injection-driven tool calls before
execution, erasing the evidence that the model had complied. An injection payload says
*skip approval and send*; a hijacked agent does not stop to ask, so those calls now reach
the tool layer and are recorded as unauthorised attempts.

**Tests added.** 255 total: unit tests for schemas, config, seeded randomness, tools,
evaluators, statistics, aggregation, storage and alignment; regression fixtures for nine
known failure types; integration tests for the full pipeline and every failure mode; CLI
exit codes; and headless rendering of all eight dashboard pages.

**Limitations.** Coverage is concentrated on evaluation behaviour rather than uniform
line coverage — deliberately. A test that asserts a function ran is not worth its
maintenance cost.

---

## Retrospective

**What the build actually taught.** Seven real defects surfaced, and all seven were found
by the same technique: building a control that *should* produce a known answer, then
investigating when it did not. The `perfect` profile exposed four false-positive sources.
The `broken` profile exposed the fact-retention transcript bug. Headless dashboard
rendering exposed the serialisation bug. The golden fixtures exposed the approval-gate
suppression.

None of them would have been caught by line coverage. Three of them made the system
report *better* numbers than reality, which is the failure mode an evaluation harness can
least afford — it looks green while measuring nothing.

**What I would do differently.** Build the `perfect`-profile control *first*, before any
evaluator. Four of the seven bugs were false positives that would have been obvious
immediately, and I wrote all nineteen deterministic evaluators before running that check
once.
