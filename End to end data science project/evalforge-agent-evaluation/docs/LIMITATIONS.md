# Limitations

What EvalForge does not establish, cannot measure, and gets wrong. This document exists
because an evaluation system that does not state its own limits is asking to be
misread, and the most common way to misuse an eval harness is to quote a number from it
without the sentence that qualifies the number.

---

## 1. The single most important limitation

**Every number in this repository was produced by a simulated model.**

The demonstration runs use `MockModelProvider`, which produces the behaviour its
configuration tells it to produce. It is not a language model and does not approximate
one. Reported figures, pass rates, context-retention scores, injection resistance —
characterise **the evaluation system**, not any model's capability.

Concretely, this means:

- "Context retention 1.000 for the baseline" means *the evaluators correctly detected no
  retention failure in an agent configured to have none*. It says nothing about GPT-4,
  Claude, or any real system.
- The baseline-versus-candidate gap is a property of two config blocks in
  `configs/failure_injection.yaml`, not a measured difference between model versions.
- No commercial model is named in any result, and none should be inferred.

Every generated report carries the banner *"Deterministic demonstration results using
simulated model behavior."* See ADR-003 for why the mock is mandatory anyway: without it
there is no CI, no reproducible regression gate, and no way for a reviewer to run
anything.

**One exception.** A single underpowered run against Llama 3.1 8B exists, reported in
the README under "A first run against a real model". It is explicitly scoped: n = 6
scored sessions, two qualitative claims, and no characterisation of the model. Everything
else in this repository is simulated.

**What would make the numbers mean something about a model:** running
`evalforge run --provider anthropic` (or `openai`) with credentials. The pipeline
supports it; the results in this repository do not use it.

---

## 2. The mock cannot exhibit failures nobody implemented

The eleven simulated degradation modes were chosen from a taxonomy the same author
wrote. That is circular in a specific way: the evaluators are tested against exactly the
failures the simulator knows how to produce.

Partially mitigated by the `perfect` and `broken` profiles, which bound the evaluators
from both sides (no false positives on a flawless session, no false negatives on a
pathological one). Not mitigated at all for **failure modes that were never imagined**.
A real model would fail in ways this taxonomy does not contain, and EvalForge would
score those sessions as clean.

---

## 3. Statistical limitations

- **n is small.** 150 scenarios. Subgroup breakdowns (category × difficulty × length)
  routinely fall below n = 20, where a point estimate is mostly noise. The analytics
  layer reports n alongside every subgroup and suppresses intervals below n = 10, but a
  suppressed interval is a warning, not a fix.
- **No multiple-comparison correction.** Roughly 25 metrics across several subgroups
  means some differences will look notable by chance. Results are therefore framed as
  descriptive diagnostics; **no p-values are reported anywhere**, deliberately.
- **Scenarios are not a sample of production traffic.** They are adversarial by
  construction and deliberately over-sample hard cases. A pass rate here is *not* an
  estimate of a real-world pass rate and must never be quoted as one.
- **Bootstrap intervals assume exchangeability.** Defensible for independently seeded
  scenarios; not defensible for production traffic, which is autocorrelated.
- **Difficulty is derived, not validated.** It is computed from length, fact count,
  constraint count and injected failures. Whether that formula matches what an agent
  actually finds hard is untested.

---

## 4. Evaluator limitations

**Deterministic evaluators are brittle against paraphrase.** They compare against
declared expected values with normalisation and alias lists. An agent that conveys the
right thing in an unanticipated form can be scored wrong. This is a deliberate trade:
a false negative on phrasing is much cheaper than a false positive on a dropped deadline,
so the checks are written conservatively.

**The contradiction evaluator only sees tracked values.** It compares successive
workspace snapshots. Free-text self-contradiction is not detected without a language
model, and the evaluator is honest about only checking what it can check.

**The goal-drift evaluator is conservative.** It fires when a turn with required tool
activity produced none. Subtler drift, the agent doing *related but wrong* work, is
not detected. This is the category where human/automated agreement is expected to be
weakest, and the alignment analysis reports it separately for that reason.

**Semantic scores use a lexical backend by default.** Token overlap is a weak proxy for
meaning. Semantic evaluators therefore report a score but never assert a failure and
never gate a release. Plugging in a real embedder is a one-class change; the shipped
default is not one.

**The `unsupported_claim` check works on numbers and named entities.** An unsupported
qualitative claim ("the team is confident") passes unnoticed.

---

## 5. Judge limitations

- **The default judge is a mock.** `MockJudge` derives verdicts from trace properties
  plus seeded noise. It exercises the full judging path, sampling, median aggregation,
  evidence validation, low-confidence flagging, but it is *not a language model*, and
  its scores are not model judgements.
- **Judge scores never gate a release** (ADR-002/ADR-004), and no judge result may carry
  critical severity. This is enforced and tested.
- **Known judge biases are analysed, not corrected.** Verbosity bias, position bias and
  reliability decay on long sessions are measured and reported; nothing adjusts for them.

---

## 6. Human alignment limitations

**The committed alignment numbers come from synthetic annotations.**
`scripts/simulate_annotations.py` generates them so the analysis pipeline is
demonstrable offline. They are labelled `synthetic: true` in the data, and the dashboard
shows a prominent warning wherever they are used. **Agreement statistics over these
labels describe the simulator, not human agreement**, and must not be cited as evidence
that the evaluators match human raters.

Real numbers require `evalforge annotate` and real annotators. Beyond that:

- Two annotators is the minimum for a kappa and well below what a serious annotation
  study needs.
- Annotators are not calibrated against each other beforehand.
- The rubric has not been validated for inter-rater reliability independently of this
  data.
- Krippendorff's alpha is reported only when enough sessions carry two independent
  annotations; otherwise it is omitted rather than computed on an inadequate sample.

---

## 7. Agent and tool limitations

- **The agent under test is deliberately simple.** A hand-written state machine, not a
  production agent. It is the *subject* of evaluation, not the product.
- **All tools are simulated.** No email is sent, no file outside the run directory is
  written, no network call is made. Real tool integration would introduce latency
  variance, partial failures and auth edge cases that this harness does not model.
- **The document corpus is small**, twelve fictional documents. Retrieval quality is not
  what is under evaluation, and the search implementation is deliberately simple and
  deterministic rather than good.
- **Latency is simulated**, so latency percentiles measure tool-choice cost, not real
  wall-clock behaviour. Cost is reported as zero because the mock provider is free.

---

## 8. Scale

EvalForge has been run at 150 scenarios × 2 runs ≈ 300 sessions. The architecture is
designed to scale further — SQLite is behind a storage interface, scenario execution is
embarrassingly parallel because every fault is seeded independently, but **it has not
been run at larger scale, and no scaling claim is made**. At 10,000+ sessions, SQLite
would likely want replacing with DuckDB and the runner would want actual parallelism.
Both are single-module changes; neither has been done.

---

## 9. What a PASS decision means

A PASS from EvalForge means exactly this:

> This agent revision, on this adversarial suite, under this configuration, exhibited no
> critical failure and cleared every blocking threshold.

It does **not** mean the agent is safe, correct, or ready for production. The suite is
finite and adversarial by construction; it cannot establish the absence of failure modes
nobody wrote a scenario for. Reports state this explicitly and no report language claims
otherwise, there is a test that enforces it
(`test_report_never_claims_general_safety`).

---

## 10. Known open items

| Item | Status |
|---|---|
| Real-model evaluation runs | Supported, not performed |
| Real human annotations | Interface built, not collected |
| Embedding-backed semantic evaluation | Interface built, lexical fallback shipped |
| Parallel scenario execution | Design permits it, not implemented |
| Trace schema migrations | `extra="forbid"` would break on field removal; no migration story |
| Multi-agent / handoff scenarios | Not modelled |
| Non-English conversations | Not modelled |
| Cost modelling for real providers | Price table exists, populated with zeros |
