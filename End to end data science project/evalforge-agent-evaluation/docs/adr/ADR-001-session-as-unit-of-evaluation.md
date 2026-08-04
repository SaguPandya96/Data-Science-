# ADR-001: Use complete sessions as the primary unit of evaluation

- **Status:** Accepted
- **Date:** 2026-08-04
- **Deciders:** Project author

## Context

The obvious design for an evaluation harness is a table of prompts with expected
outputs, scored one row at a time. It is simple, parallelises trivially, and produces a
clean accuracy number.

It is also blind to the failures that break agent products. Consider the demonstration
conversation:

```
Turn 1: Create a launch plan. Launch date September 15. Budget $20,000. No paid advertising.
Turn 2: Add a two-week QA period.
Turn 3: Reduce the budget to $15,000.
Turn 4: Prepare an executive summary. Keep the original launch date.
Turn 5: Add paid social advertising.
```

Every individual turn has a perfectly good single-turn response. The interesting
questions only exist *across* turns:

- Did turn 4's summary keep September 15, or silently absorb a date shifted by the QA
  period added in turn 2?
- Did it use $15,000 or the superseded $20,000?
- Did turn 5 get flagged as conflicting with the turn 1 constraint, or quietly obeyed?
- Did adding QA change fields nobody asked to change?

None of these are properties of a response. They are properties of a session.

## Decision

The unit of evaluation is a **complete session**: a scripted multi-turn conversation
executed end to end, producing a `SessionTrace` containing every message, tool call,
state transition and workspace snapshot.

Turn-level evaluation still exists, but as *localisation* — it answers "where did it
break", not "did it pass". Release readiness is decided at session level only.

## Consequences

**Positive**

- Context retention, instruction persistence, goal drift, cascading errors and recovery
  become directly measurable. None are expressible in a single-turn harness.
- Failure localisation is preserved: turn-level results point at the exact turn.
- Human annotators and automated evaluators judge the same artifact, which is what makes
  agreement statistics meaningful.

**Negative**

- Sessions are expensive. A 30-turn scenario is ~30× a single-turn case in tokens and
  latency. Mitigated by the deterministic mock provider making CI runs free.
- Failures are entangled. A wrong tool argument on turn 3 can *cause* what looks like a
  context failure on turn 9. This is why `cascading_error` is its own category with a
  propagation-depth metric, rather than being counted as several independent failures.
- Scenario authoring is much heavier than prompt authoring. This is why the generator
  is template-driven and seeded rather than hand-written.

**Neutral**

- Statistical power is now a function of session count, not turn count. 150 sessions is
  a small sample, and the analytics layer reports bootstrap confidence intervals rather
  than bare point estimates because of it.

## Alternatives considered

**Single-turn scoring with conversation context in the prompt.** Cheap and simple, but
it can only score the final response. It cannot observe tool calls, cannot detect that
turn 3's budget update was ignored until it surfaces in turn 9's artifact, and gives no
purchase on recovery behaviour.

**Turn-level scoring with post-hoc session roll-up.** Closer, and partially adopted.
Rejected as the *primary* unit because averaging turn scores systematically understates
session failure: 29 good turns and one dropped deadline averages to 0.97, which is a
number that would ship a broken agent.
