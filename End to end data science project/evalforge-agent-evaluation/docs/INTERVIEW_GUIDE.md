# Interview Guide

Explanations at four lengths, then the questions this project invites and honest answers
to them.

---

## The 30-second version

> Most LLM evaluation scores one response at a time. That misses the failures that
> actually break agent products, because those failures are temporal — a budget stated on
> turn 1 and needed on turn 14, a constraint issued once that must hold for the rest of
> the session, a wrong value extracted early that quietly corrupts a forecast later.
>
> EvalForge evaluates complete multi-turn sessions. It generates 150 adversarial
> conversations across eight failure categories, runs an agent through them, records a
> full trace, and scores context retention, instruction adherence, tool reliability,
> failure recovery and prompt-injection resistance. Critical failures block release
> outright rather than averaging away.

---

## The 60-second version

Adds the mechanism:

> The unit of evaluation is a session, not a response. A scenario is a *contract*: it
> declares what the user says, which facts must survive, which constraints stay live,
> which tools should fire, and what would count as failure. The agent runs against it and
> produces a trace. Evaluators compare the trace to the contract — nothing about
> correctness is decided at run time.
>
> There are three evaluator families, kept strictly separate. Twenty-one deterministic
> checks do exact comparison and are the only things that gate a release. Semantic checks
> report a similarity diagnostic but never assert a failure. An LLM judge scores the
> dimensions no exact rule can reach — planning quality, usefulness — with multi-sample
> median aggregation and a hard requirement that every verdict cite trace evidence.
>
> Six failure categories are *critical*: obeying a prompt injection, an unauthorised
> send, fabricating a tool result, losing a pinned deadline, a wrong budget, disclosing
> protected content. One occurrence blocks release regardless of the weighted score,
> because a weighted average lets a single obeyed injection average away to a 0.97.
>
> The whole thing runs offline on a deterministic mock provider, which is what makes the
> regression gate trustworthy: baseline and candidate differ only in the agent, never in
> sampling noise.

---

## The two-minute technical version

**The problem.** Single-turn evaluation is structurally blind to session-level failure.
Take the demonstration conversation: create a launch plan with a September 15 date and a
$20,000 budget, no paid advertising; add QA; reduce the budget to $15,000; write a
summary keeping the original date; then add paid social advertising. Every individual
turn has a perfectly good single-turn response. The interesting questions only exist
across turns — did the summary keep September 15 or absorb a date shifted by the QA
phase, did it use $15,000 or the superseded $20,000, was the final request flagged as
conflicting with turn 1 or quietly obeyed.

**The pipeline.** Scenario generator → agent under test → trace collector → evaluation
engine → analytics → dashboard and release report. Data flows one way. The trace is the
single source of truth, which means a stored run can be re-scored months later by newer
evaluators without re-invoking a model — and it means a human annotator and an automated
evaluator are looking at exactly the same evidence, which is what makes agreement
statistics meaningful.

**Determinism.** Every stochastic decision draws from a generator seeded on *what is
being decided* — `(run_seed, scenario_id, turn_index, aspect)` — not on global state. A
decision therefore depends only on its own coordinates, so running scenario 87 alone
produces the same faults as running it inside a 150-scenario suite, in any order, in
parallel. Without that, recovery rate and latency percentiles are not comparable between
runs and a 3-point regression tolerance means nothing.

**Scoring.** Eight weighted dimensions from a versioned YAML rubric. Each failing check
subtracts a severity-scaled penalty, because a plain mean is too forgiving: nine passes
and one dropped deadline averages to 0.90, which is a comfortable number for a broken
session. Critical failures short-circuit the whole thing.

**The regression demonstration.** A reliable baseline profile and a deliberately degraded
candidate that loses long-range context, mis-argues tools, recovers poorly and sometimes
obeys retrieved instructions. The candidate drops from 94.7% to 44.7% pass rate with 30
critical failures, eleven metrics breach tolerance, and `evalforge compare` exits
non-zero. CI asserts the *inverse* polarity: if the gate ever passes against the degraded
candidate, the build fails, because that means the gate stopped working.

**Honesty.** Every number here came from a simulated model. Reports say so on every page.

---

## Architecture questions

**Why sessions rather than turns?**
Because the failures that matter are temporal. Context retention, instruction
persistence, goal drift and cascading errors are properties of a session and are not
expressible per-turn. Turn-level evaluation still exists, but as *localisation* — it
answers "where did it break", not "did it pass". Rolling up turn scores as the primary
metric would systematically understate failure: 29 good turns and one dropped deadline
averages to 0.97. (ADR-001)

**How do you stop the harness grading its own answer key?**
The agent never reads the scenario contract. It sees user turns and its own tool results.
The mock *provider* does consult scenario metadata — it stands in for model comprehension,
so it has to know what a competent agent would understand — but the boundary is the
`SessionTrace`: everything after trace collection sees only what a real deployment would
have logged. If that leaked, the evaluators would be scoring the answer key.

**Why SQLite and JSONL rather than one store?**
Traces are deep, ragged documents; SQL cannot query them usefully but `grep` and `jq`
can. Indexed metadata is the opposite: "every failing session with a critical safety
failure, ordered by score" is a relational query that runs in milliseconds. SQLite over
DuckDB because it ships in the standard library — no dependency, no wheel availability
question in CI. Storage is behind an interface, so swapping in DuckDB at larger scale is
a single-module change. The cost is two sources of truth; the trace is written before the
index row so a crash leaves an orphaned trace (recoverable) rather than a dangling index
row (corrupt).

**Why a hand-written agent instead of LangGraph?**
The agent is the *subject* of evaluation, not the product. A framework graph would add
indirection to the thing being measured. Every decision that affects a score is visible
in one file. LangGraph would be a reasonable choice if the agent were the deliverable.

**What would you change if this ran at 10,000 sessions?**
DuckDB instead of SQLite, and actual parallel execution — which the seeding scheme
already permits, since no fault depends on execution order. Neither has been done, and
the docs say so rather than claiming scale that was never exercised.

---

## Evaluation-methodology questions

**Why separate deterministic checks from the judge?**
Because sending everything to a judge makes the *reliable* checks as unreliable as the
unreliable ones, and produces a single number where nobody can tell which part to trust.
The rule adopted: if a reliable expected value exists, a deterministic evaluator must be
used; the judge is a fallback for the residue, not a default. The practical payoff is
that judge drift — a new judge model shifting all scores — cannot silently move the
release bar, because the bar is not made of judge scores. (ADR-002)

**Why are critical failures categorical instead of heavily weighted?**
Weighting only moves the arithmetic. With safety at 5%, one obeyed injection in 150
sessions moves the aggregate by well under a point; at 40% it still averages away, and it
distorts scoring for sessions with no safety content. The honest framing is that some
failures are not statistical: an agent that obeys an injected instruction 2% of the time
is not 98% safe. The real cost is that a false positive blocks a release, which is why
only exact checks may emit critical severity — no judge or semantic evaluator can. (ADR-004)

**How do you know the evaluators actually work?**
Two controls, both in the test suite. A `perfect` profile proves no false positives — a
flawless session must produce zero failures, and it does. A `broken` profile proves no
false negatives — a pathological session must be caught, and it scores 0.305 with 59
critical failures. Beyond that, nine golden fixtures pin one specific failure mode each to
a specific expected verdict, using targeted profiles that isolate the mode rather than
the shipped `broken` profile, which drifts too often to test anything precisely.

**What's the weakest evaluator?**
Goal drift. It fires only when a turn with required tool activity produced none, so it
catches abandonment but not subtler drift where the agent does related-but-wrong work.
It is deliberately conservative because a false positive there would be expensive, and
the alignment analysis reports agreement on drift scenarios separately for that reason.

**Tell me about a bug you found in your own evaluator.**
Three worth naming. First, `@computed_field` on two models meant every persisted record
serialised a field its own validator then rejected — the pipeline "worked" because it
held results in memory, and only reading a run back failed. Caught by running the
dashboard pages headlessly. Second, per-failure results were also session-scoped, so
aggregation averaged each failure into the basis *and* subtracted it as a penalty —
double-counting every defect. Third, and worst: fact retention checked the *whole
session* transcript, so a fact stated correctly on turn 0 and lost by turn 20 still
counted as retained. That is precisely the failure the evaluator exists to catch, and it
was silently passing.

---

## Statistical questions

**Why Wilson intervals rather than normal approximation?**
The interesting rates sit near the boundaries — injection resistance around 0.98 — where
the normal approximation misbehaves and can produce upper bounds above 1.0. Wilson is
well-behaved at the boundaries and at small n, which is exactly this regime.

**Why bootstrap for means?**
Overall score, latency percentiles and cost-per-success have unknown, skewed sampling
distributions. Bootstrapping assumes nothing about their shape. It does assume sessions
are exchangeable, which is defensible for independently seeded scenarios and is *not* a
claim about production traffic.

**Why Cohen's h for proportion deltas?**
Because a raw percentage-point difference misleads at the edges. 0.98 → 0.95 and
0.50 → 0.47 are the same three points and very different events; the arcsine transform
distinguishes them.

**Why Spearman rather than Pearson for human/automated correlation?**
Human ratings are ordinal — a 1..5 rubric — and the relationship need only be monotonic.
Pearson would assume interval spacing that a rubric scale does not have.

**Why does kappa matter more than raw agreement?**
Chance correction. With an 85% pass rate, two raters who both always say "pass" show 85%
raw agreement and know nothing; kappa is near zero, and kappa is the honest number.

**Why no p-values?**
About 25 metrics across several subgroups with no multiple-comparison correction. Some
differences will look significant by chance. Reporting p-values would lend false
authority, so these are framed as descriptive diagnostics and the limitation is stated
explicitly.

---

## Trade-offs made

| Chose | Over | Because | Cost |
|---|---|---|---|
| Session-level evaluation | Turn-level | Temporal failures are invisible per-turn | Expensive; failures entangle |
| Deterministic mock | Real model | CI without secrets; noiseless regression gate | Results say nothing about real models |
| Deterministic checks gate | Judge gates | No measurement error in the release decision | Brittle against paraphrase |
| Categorical criticals | Heavy weights | Averages hide single incidents | A false positive blocks a release |
| SQLite + JSONL | One store | Right tool per query shape | Two sources of truth |
| Hand-written agent | LangGraph | The measured thing stays small | Less realistic than a framework agent |
| Generated scenarios | Hand-authored | Reproducible; a seed fixes the suite | Less nuanced than human-written cases |
| Config in YAML | Constants in code | Policy changes are reviewable diffs | Indirection; config/code can drift |

---

## Limitations to raise before you are asked

Raising these unprompted is better than being caught by them:

1. **Every number came from a simulated model.** It measures the evaluation system, not
   any LLM.
2. **The committed alignment statistics use synthetic annotations**, clearly labelled as
   such. Real numbers need real annotators.
3. **n = 150** is small; subgroups fall below n = 20 where estimates are noisy.
4. **The mock can only exhibit failures someone implemented** — the taxonomy and the
   simulator share an author, which is circular in a specific way.
5. **Scenarios are adversarial by construction** and are not a sample of production
   traffic, so pass rates here are not real-world pass rates.

Full detail in `docs/LIMITATIONS.md`.

---

## Future improvements, in priority order

1. **Run it against a real model.** The provider layer supports it; the results in this
   repository do not use it. This is the single highest-value next step.
2. **Collect real human annotations** and replace the synthetic ones, which would make
   the alignment analysis mean what it currently only demonstrates.
3. **Plug in a real embedder** for the semantic evaluators, so meaning-preservation stops
   being a lexical proxy.
4. **Mine production traces for failure modes** the taxonomy is missing — the direct fix
   for limitation 4.
5. **Parallel execution and DuckDB** if the suite grows an order of magnitude.
6. **Schema migrations**, before anything depends on traces surviving a field removal.
