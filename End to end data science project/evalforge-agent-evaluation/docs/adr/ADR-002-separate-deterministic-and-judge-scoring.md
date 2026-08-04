# ADR-002: Separate deterministic checks from LLM judging

- **Status:** Accepted
- **Date:** 2026-08-04
- **Deciders:** Project author

## Context

Some evaluation questions have exact answers. "Does the final plan contain the launch
date 2026-09-15?" is a string comparison. "Is `calculate_budget(15000)` consistent with
the user's turn-3 revision?" is arithmetic.

Other questions do not. "Is this project plan coherent?" and "would a user find this
executive summary useful?" have no expected value to compare against.

The tempting move is to send everything to an LLM judge — it handles both, and the code
is uniform. The problem is that it makes the *reliable* checks as unreliable as the
unreliable ones, and produces a single number in which nobody can tell which part is
trustworthy.

## Decision

Three evaluator families, structurally separated and never averaged into each other:

1. **Deterministic** (`evaluator_kind="deterministic"`) — exact comparison against the
   scenario contract. Confidence is 1.0 by construction. **Used for the release gate.**
2. **Semantic** (`evaluator_kind="semantic"`) — lexical/embedding similarity for
   meaning-preservation questions. Reported, not gated.
3. **Judge** (`evaluator_kind="judge"`) — LLM-graded, for dimensions with no expected
   value. Reported alongside deterministic scores, never blended into them.

The rule for deciding which to use: **if a reliable expected value exists, a
deterministic evaluator must be used.** An LLM judge is a fallback for the residue, not
a default.

Judge results carry `judge_model` and `judge_prompt_version` in metadata, because a
judge score is only interpretable relative to the judge that produced it.

## Consequences

**Positive**

- The release gate rests on checks with no measurement error. If EvalForge says a
  deadline was dropped, the deadline was dropped.
- Judge drift — a new judge model shifting all scores — cannot silently move the release
  bar, because the bar is not made of judge scores.
- The two can be compared against each other and against humans, which is the entire
  point of the alignment analysis. Blending them would destroy that.

**Negative**

- More code: nineteen deterministic evaluators instead of one judge prompt.
- Deterministic evaluators are brittle against paraphrase. Mitigated by alias lists on
  facts and normalised comparison, and accepted deliberately: a false *negative* on
  paraphrase is far cheaper than a false *positive* on a dropped deadline.
- Two score families in the report can confuse readers. Mitigated by labelling every
  result with its `evaluator_kind` in the report and dashboard.

## Alternatives considered

**Judge everything.** Uniform and flexible. Rejected: it makes exact checks probabilistic,
costs money per evaluation, requires network access (breaking the offline requirement),
and is non-deterministic, which would make the regression gate unusable.

**Deterministic only.** Fully reliable and fully offline. Rejected because it cannot
score planning quality, usefulness or nuanced goal alignment at all — those dimensions
would simply be missing, and they are ones product owners actually ask about.

**Weighted blend of both into one score per dimension.** Rejected because the blend
weight is unjustifiable, and because a blended number cannot be audited: you cannot tell
whether a 0.82 means "certainly correct, judged mediocre" or the reverse.
