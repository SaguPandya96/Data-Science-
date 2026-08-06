# Development log

Notes kept while building EvalForge. Design decisions, the things I rejected, and the
bugs that cost me the most time. Roughly chronological.

---

## Setting up

Python 3.11 as the floor, mostly for `StrEnum` and the `X | Y` union syntax the schemas
lean on. Hatchling with a `src/` layout, so tests import the installed package rather
than whatever happens to be sitting in the working directory. That detail matters more
than it sounds: with a flat layout you can have a broken package and a green test suite
at the same time.

Ruff for lint and format, replacing what would otherwise be Black plus isort plus flake8.
mypy runs over `src/` only. I tried requiring full annotations on the tests as well and
backed that out after about ten minutes, since annotating fixtures adds noise without
catching anything.

Looked at Poetry and dropped it. It would have put the only lock file in a repository
that has none, for no benefit here.

## Deciding what the unit of evaluation is

Everything else hangs off this, so I wrote it up as ADR-001 before writing any code.

The obvious design is a table of prompts and expected outputs, scored a row at a time. It
parallelises, it's simple, and it cannot see any of the failures I actually care about.
Take the launch-plan conversation: state a budget, revise it two turns later, ask for a
summary that keeps the original date. Every individual turn has a perfectly good
single-turn answer. Whether the summary used the revised budget is not a property of any
one turn.

So, sessions. Turn-level results still exist, but only to localise a failure, never to
decide pass or fail. Averaging turn scores would let 29 good turns and one dropped
deadline come out at 0.97, and 0.97 ships.

The other four ADRs came out of the same sitting: separating deterministic checks from
model-graded ones (002), making the offline mock mandatory (003), treating critical
failures as categorical rather than heavily weighted (004), and keeping scoring policy in
versioned YAML (005).

The one I'd defend hardest is 004. Weighting safety at 40 percent instead of using an
override sounds equivalent and isn't. One obeyed prompt injection in 150 sessions still
averages away, and you've distorted scoring for every session that has no safety content
in it at all. Some failures aren't statistical.

## Schemas

Pydantic v2 with `extra="forbid"` everywhere. A typo'd key should be an error, not a
silently ignored field.

`Fact` and `Constraint` are frozen because many evaluators read them concurrently.
`Scenario.final_fact_values()` resolves the override semantics in one place, which is the
logic the whole "reduce the budget but keep the date" case depends on. Having that in two
places would have been a bug waiting to happen.

Config layers YAML, then environment, then CLI flags, and hashes into a `config_digest`
stored on every run. That's what makes it possible to answer "which thresholds produced
this decision" six weeks later without guessing.

`extra="forbid"` has a cost I haven't paid yet. Remove a field and every old trace stops
loading. Fine before 1.0, needs a migration story after.

## Tools

Eight simulated tools. `BaseTool.invoke` owns validation, approval enforcement, latency
and fault injection so no individual tool can forget to do any of it. Faults get applied
around real execution rather than inside it, which keeps each tool a straightforward
simulated action and means a new fault type applies to all eight at once.

One ordering detail took two attempts to get right. Corruption has to happen *after* the
tool produces its own valid output. Do it before, and a `MISSING_FIELD` fault becomes an
exception whose contents the agent never sees. Do it after, and the agent gets a payload
with a hole in it that it has to notice, which is the behaviour actually under test.

`search_documents` returns document bodies verbatim, injection payloads included.
Sanitising there was tempting, and would have quietly made the entire prompt-injection
category untestable.

## The mock provider

CI needs an agent that can behave competently and also fail in named ways on demand,
reproducibly, with no credentials. That's ADR-003, and it's the piece that makes
everything else possible.

I looked hard at record/replay against a real model first. It's genuinely attractive:
real behaviour, replayable offline. Three things killed it. Cassettes are valid for
exactly one model version. A reviewer without credentials can't regenerate them. And you
cannot ask a real model to lose context after exactly turn 10, which is precisely the
controlled degradation the regression demonstration needs.

Every degradation is a seeded Bernoulli trial keyed on `(run_seed, scenario_id,
turn_index, aspect)`. A decision depends only on its own coordinates, so scenario 87
produces identical faults whether it runs alone or inside the full suite, in any order.
That property does more work than it looks like. Recovery rates and latency percentiles
aren't comparable between runs without it, and a 3-point regression tolerance means
nothing against sampling noise.

`ModelResponse` carries structured state rather than requiring prose parsing. What the
model reports it still remembers is a far cleaner retention signal than searching output
text for a number.

One thing I had to force myself to do: when the mock forgets a prohibition, it has to
actually produce the forbidden content. Anything less and the violation is invisible to
the evaluator, and the simulation is lying.

## Scenario generation

150 hand-written multi-turn contracts was never realistic. Templates plus seeded variation
instead, across project, dates, budgets, constraint mix, distractor placement and fault
type.

The failure mode I worried about is thin content pools, where 150 lexically
near-identical scenarios measure one case 150 times. Ended up with 110 distinct scenario
names out of 150, which I'm satisfied with.

Difficulty is derived from measurable pressure rather than assigned by hand, so a 20-turn
scenario with two faults reads as hard whether it came from the recovery family or the
stress family.

Conversation lengths needed padding to land exactly on 5, 10, 15, 20 and 30. The
length-sweep analysis buckets by those values, and a suite full of 13- and 27-turn
conversations would have smeared the buckets into uselessness.

The validator enforces category-specific rules, so an injection scenario with no payload
fails as mislabelled rather than passing as merely weak.

## The evaluators, and four false positives

Twenty-one deterministic checks. Normalisation lives in one module because comparison
semantics are policy, not implementation detail. "Is `$15,000` the same as `15000.00`"
needs one answer system-wide.

Tool order is scored as a longest-common-subsequence ratio rather than exact equality. A
differently interleaved but dependency-respecting order isn't a defect and shouldn't
score like one. Argument matching only compares the keys a scenario pins, because pinning
everything would measure prompt-template stability instead of correctness.

Then I built the `perfect` profile as a control, expecting 1.0, and it scored 0.927 with
failures in four categories. All four were bugs in my checks, not the agent:

1. A plan carries its constraints as text, and those descriptions contain the banned
   phrase. An agent correctly recording "do not include paid advertising" was scored as
   including paid advertising.
2. The summary tool's excluded-topic filter was deleting its own constraints block, which
   emptied the summary and then tripped a missing-section failure downstream.
3. Flagging a conflict ("this contradicts your instruction to exclude X") necessarily
   names X, and was scored as a violation. Correct behaviour was being punished.
4. Semantic evaluators were held to the deterministic pass threshold, so they
   manufactured failures out of nothing but vocabulary difference.

That control should have been the first thing I built, not something I got to after
writing all twenty-one checks.

## Judge and semantic layers

The judge is treated as an unreliable instrument throughout: schema-validated output,
required evidence, multi-sample median aggregation, recorded model and prompt version,
and a hard cap at major severity so it can never gate a release.

Requiring evidence turned out to be the most useful single control. It forces the judge to
point at something real instead of producing a plausible-sounding number.

`MockJudge` injects seeded variance on purpose. Without it, median-of-three collapses to a
no-op and the aggregation path is never genuinely exercised by any test.

Semantic evaluators report a score but never assert a failure. The default backend is
lexical, and lexical overlap between a correct reply and the prose describing what was
expected routinely sits below 0.2. Any threshold above zero labels correct paraphrase as
failure.

## Storage, and the aggregation bug

SQLite for indexed metadata, JSONL for traces, reasoning written up in ARCHITECTURE.md.
Traces are ragged documents that SQL can't query usefully but `grep` can. Meanwhile
"every failing session with a critical safety failure, ordered by score" is relational and
runs in milliseconds. The trace gets written before the index row, so a crash leaves an
orphaned trace rather than an index row pointing at nothing.

Found a scoring bug here that had been wrong from the start. Each evaluator emits a
roll-up result plus one result per individual failure, and both are session-scoped.
Aggregation was averaging the failures into the basis *and* subtracting them again as
penalties. Every defect counted twice. It was dragging `tool_selection_accuracy` from 0.96
down to 0.74, which is what made me look. Fixed by marking roll-ups explicitly.

## Dashboard, and the serialisation bug

Strict rule: `dashboard/data_access.py` is a thin read layer, and no page computes a
metric. Whatever is on screen has to be the same number the release report prints, from
the same tested code, or the two drift and one of them is quietly wrong.

Ran the pages headlessly to check them and hit `ValidationError` immediately. Two models
used `@computed_field`, which serialises the computed value into JSON, where
`extra="forbid"` then rejects it on read. Every persisted evaluation result and every
trace was unreadable.

The pipeline had looked fine the whole time because it holds results in memory. Only
*reading a run back* failed, and nothing had read one back until the dashboard did. That
one bothered me. It had been broken for hours across several commits with a green test
suite the entire time.

The annotation interface hides automated scores until submission and records
`blind=true`. Non-blind annotations are excluded from agreement statistics entirely, since
an annotator who has already seen the automated verdict isn't an independent rater.

The synthetic annotation generator was a judgement call. Shipping an alignment page that
stays blank until someone hand-labels 60 sessions would leave a whole subsystem
unexercised and untestable. So it generates annotations labelled as synthetic, and the
dashboard warns loudly wherever they're used. They demonstrate the pipeline. They are not
evidence about human agreement and shouldn't be read as any.

## Tests, and two more bugs

258 tests. The two that matter most are the bounds: a flawless session must produce zero
failures, and a pathological one must be caught. An evaluation harness that can't do both
is worse than none, because it reports green while measuring nothing.

Golden fixtures use targeted profiles built from `perfect` with a single rate turned on,
rather than the shipped `broken` profile. `broken` drifts 60 percent of the time and kept
abandoning turns before reaching the behaviour under test, so fixtures built on it were
testing the mock's dice rather than the evaluator.

Two more real bugs came out of writing these.

**Fact retention was searching the entire transcript.** A fact stated correctly on turn 0
and lost by turn 20 still counted as retained, because the check found the turn-0 mention.
That is exactly the failure the evaluator exists to catch, and it had never once detected
it. The workspace snapshot is now authoritative, with final-turn text as a fallback only
for providers that report no structured state.

**Total context loss was invisible.** An empty `remembered_facts` was read as "no update"
rather than "everything forgotten", so the agent silently kept stale facts. Added an
explicit `state_reported` flag to tell "holding nothing" apart from "not reporting".
Fixing this moved the degraded candidate from 30 critical failures to 155, which is the
honest number.

Also found that the approval gate was stripping injection-driven tool calls before they
executed, erasing the evidence that the model had complied at all. An injection payload
says *skip approval and send*, and a hijacked agent doesn't stop to ask permission. Those
calls now reach the tool layer and get recorded as unauthorised attempts.

## CI

Four jobs, no secrets, everything on the mock provider.

The evaluation job asserts an inverted condition, which felt strange to write. Because the
candidate is deliberately degraded, a *passing* regression gate fails the build. A gate
that stops detecting real degradation is the failure mode that matters, and it's invisible
if you only ever check for green.

The first CI run went red on two things I couldn't have caught locally. I'd been running
`python -m pytest`, which silently puts the working directory on `sys.path`; CI runs the
bare `pytest` binary, which doesn't, so `from tests.conftest import ...` failed. And I'd
pinned `python_version = "3.11"` for mypy, so the 3.12 job analysed numpy's 3.12-syntax
stubs under 3.11 rules and choked. Both fixed in pyproject. I now verify with the bare
binaries rather than the module form.

## What I'd do differently

Build the `perfect` control first, before any evaluator. Four of the seven bugs were false
positives that would have been obvious on day one, and I wrote every check before running
that comparison once.

The other pattern worth noting: all seven bugs were found the same way, by constructing a
case with a known answer and investigating when the answer came out wrong. Not one would
have been caught by line coverage. Three of them made the system report *better* numbers
than reality, which for an evaluation harness is the worst direction to be wrong in.

## Known gaps

- Everything reported here comes from a simulated model. It measures the evaluation
  system, not any LLM. The provider layer supports real models; I haven't run one.
- Alignment statistics are computed over synthetic annotations.
- No trace schema migration story. `extra="forbid"` breaks on field removal.
- Scenario execution is embarrassingly parallel by construction and still runs serially.
- Difficulty is derived from a formula I made up. Whether it matches what an agent
  actually finds hard is untested.
