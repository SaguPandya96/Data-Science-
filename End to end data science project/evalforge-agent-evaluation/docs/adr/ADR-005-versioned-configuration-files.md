# ADR-005: Store evaluation configuration and thresholds as versioned files

- **Status:** Accepted
- **Date:** 2026-08-04
- **Deciders:** Project author

## Context

An evaluation result is meaningless without the configuration that produced it. "The
agent scored 0.87" is not a fact; "the agent scored 0.87 under rubric v1.0.0 with
context retention weighted at 0.20 and a session pass threshold of 0.75" is.

Three specific failure modes motivated this:

1. **Silent bar movement.** Someone lowers `context_retention` from 0.90 to 0.85 to
   unblock a release. Six weeks later nobody can tell why the metric improved.
2. **Irreproducible decisions.** A stored run cannot be re-explained if the thresholds
   that judged it live only in whatever the code said that day.
3. **Invisible product judgement.** Dimension weights are a statement about what the
   organisation values. Buried in a Python dict, that statement is invisible to the
   people who should be arguing about it.

## Decision

All evaluation policy lives in versioned YAML under `configs/`:

| File | Owns |
|---|---|
| `default.yaml` | Provider, agent, scenario, evaluation, storage, logging, cost |
| `evaluation_rubrics.yaml` | Dimension weights, pass thresholds, severity penalties, judge rubric |
| `release_thresholds.yaml` | Release gates and regression tolerances |
| `failure_injection.yaml` | Tool fault catalogue, behaviour profiles, latency model |

Additionally:

- Every file carries a `version` field.
- Every threshold carries a `rationale` string, in the file, next to the number.
- The effective merged config is hashed into a `config_digest`, stored on every
  `RunSummary`, and printed in every report.
- Rubric weights are validated to sum to 1.0 at load time; a config that does not is a
  `ConfigurationError`, not a silently renormalised score.
- Layering is YAML → environment → explicit CLI overrides, all Pydantic-validated.

## Consequences

**Positive**

- Changing the bar is a reviewable diff with a required rationale. `git blame` answers
  "who lowered this and why".
- Any stored run can be tied to the exact policy that scored it via its digest.
- Weights are visible to non-engineers, which is where the argument about them belongs.
- Comparing runs scored under different digests is detectable rather than accidental.

**Negative**

- Config and code can drift: a new dimension in code with no weight in YAML. Mitigated
  by validating weights against the `Dimension` enum at load and failing loudly.
- More indirection — reading the scoring logic means reading two files. Accepted.
- YAML permits deep nesting and type surprises. Mitigated by Pydantic models with
  `extra="forbid"`, so a typo'd key is an error rather than a silently ignored setting.

## Alternatives considered

**Constants in Python.** Simplest, type-checked, refactorable. Rejected: it makes policy
a code change, hides it from non-engineers, and offers no way to record a rationale
alongside a number.

**A database table.** Good for a hosted multi-tenant service. Rejected here: it takes
policy *out* of version control, which is precisely the property that makes it
reviewable, and adds infrastructure the project explicitly avoids.

**Command-line flags only.** Rejected: nothing is recorded, every run is a fresh
argument, and reproducing a decision means finding someone's shell history.
