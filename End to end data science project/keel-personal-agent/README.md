# Keel: A Personal AI Agent That Remembers, and Knows When to Forget

A personal agent is only personal if it remembers you. It also has to forget: when you
move from Denver to Austin, it should stop suggesting restaurants in Denver. I built Keel,
a personal agent in the spirit of this year's consumer agents, to find out one thing: does
its memory put the right, *current* fact in front of the model more often than the simple
ways of doing it?

Short answer: **yes, by a wide margin, with two limits worth knowing.** On 3,200 held-out
questions from 200 synthetic users, Keel put the current answer in the prompt, with no
outdated version beside it, **62.6%** of the time (95% CI 61.4 to 63.9). Showing the newest
memories managed 4.6%, and standard keyword search 9.8%. For details that had changed, the
gap was widest: 67.8% for Keel against 5.5% for keyword search, which kept surfacing the old
value. The limits: Keel still misses questions that only hint at their topic ("Who should I
book the anniversary dinner with?"), and its forgetting depends on the model labeling
updates consistently.

## What Keel does

You talk to it in a terminal or a browser. It works through tools rather than just
chatting:

- **Remembers** facts, preferences, constraints, goals and events across sessions, and
  **retires** the old value when something changes. You can see and delete anything it
  knows about you.
- **Plans**: turns a goal ("run a half marathon in March") into dated milestones and tasks,
  and checks your calendar and constraints (allergy, budget, gym days) before proposing
  anything.
- **Manages your day**: calendar with conflict checks and free-slot search, tasks, notes,
  email drafts, and web search.
- **Briefs you each morning** with today's events, clashes, overdue tasks, milestones due
  this week, goals you haven't checked in on, and actions waiting for you. The brief needs
  no model call, so it can run from cron for free.
- **Asks before acting on your behalf.** Anything that reaches another person, such as sending
  an email, becomes a pending approval. The agent can't approve its own requests. Every
  tool call is written to an audit log.

It never stores passwords, card numbers or ID numbers, even if the model tries to save them.

```text
                        ┌──────────────── your message ────────────────┐
                        ▼                                              │
  memory ──► Keel retriever ──► <context> time · memories · keys · agenda
                                      │
                                      ▼
                        Claude (tool use, adaptive thinking, web search)
                                      │ tool calls
                                      ▼
            ┌──────────── Toolbox: validates input, logs every call ───────────┐
            │ read: recall, calendar_list, task_list, goal_list, daily_brief …  │
            │ write: remember, forget, calendar_add, task_add, goal_plan …      │
            │ outward: email_send ──► approval queue ──► you approve or reject  │
            └────────────────────────────────────────────────────────────────────┘
```

## How the memory works

Each memory is one sentence with a kind (fact, preference, constraint, goal, episode) and,
for details that can change, a key such as `home_city`. Before each reply, Keel picks up to
eight memories for the prompt:

1. **Only current memories.** Saving a new value under a key the user already has
   retires the old one. The old memory stays in the database for the history view, but it
   is never shown to the model again.
2. **Constraints always go in.** Allergies and hard limits are included whatever the
   question, capped at half the budget. Forgetting a peanut allergy while booking dinner
   is the mistake that matters most.
3. **Hybrid search for the rest.** BM25 over each memory's key and text, with stemming and
   synonym expansion ("live" also matches "based", "home", "city"), plus small boosts for
   recent and important memories. Repeats of the same small talk are skipped.

The model sees the keys already in use, so when you say "I moved to Austin" it can save the
new value under `home_city` and retire Denver.

## Measuring it

### The benchmark

I wrote the [analysis plan](docs/ANALYSIS_PLAN.md) and the pass rule before running the
test questions, and committed them first so the order is visible in git history.

The benchmark generates **200 users**, each with 16 personal details spread over 12 weekly
sessions: city, employer, diet, allergy, partner, pet, and so on. 45% of details change
later (a move, a new job), and a quarter of those change again. Every session adds 4 to 9
small-talk memories, many deliberately sharing words with the questions: asking about
restaurants in *another* city, or a cover letter for a job somewhere else. That gives about
100 memories per user. At the end, each user asks one question per detail, and each method
may put **5 memories** in the prompt.

The main measure is a **clean hit**: the current value is in the prompt and no earlier value
is. A model shown both "lives in Denver" and "moved to Austin" can get it wrong; a clean hit
leaves nothing to get wrong.

I write both the retriever and the questions, which is a real risk of tuning one to the
other. So every detail has two sets of question wordings. I built and debugged against the
*dev* set only and ran the *test* set once, after freezing the code. That split turned out
to matter a lot (see below).

### Results on the held-out questions

| Method | Clean hit | Outdated value shown | Allergy in prompt* | Prompt tokens |
| --- | --- | --- | --- | --- |
| No memory | 0.0% | 0.0% | 0% | 0 |
| Newest 5 memories | 4.6% (3.9 to 5.2) | 0.0% | 0% | 84 |
| Keyword search (BM25) | 9.8% (8.8 to 10.8) | 5.0% | 1% | 29 |
| **Keel** | **62.6% (61.4 to 63.9)** | **0.0%** | **100%** | 75 |
| Every memory | 62.2% (60.6 to 63.8) | 37.8% | 100% | 1,664 |

\*For questions about something other than the allergy. 95% intervals are from a bootstrap
over users. Source: [`reports/metrics/benchmark_test.md`](reports/metrics/benchmark_test.md).

**The pre-registered rule passed.** Keel beat the newest-memories method by **58.1 points**
(95% CI 56.7 to 59.5) and keyword search by **52.8 points** (51.3 to 54.3).

Putting every memory in the prompt ties Keel on clean hits while using **22 times the
tokens**, and for details that changed it *always* shows the outdated value alongside the
current one. A strong model might still pick the newer date, which the end-to-end check below
is built to test. But the cost grows with every conversation, and Keel's doesn't.

### What each part contributes

Removing one piece at a time (clean hit, all questions):

| Keel without | Clean hit | Change |
| --- | --- | --- |
| (nothing removed) | 62.6% | |
| Synonym expansion | 34.7% | −27.9 pts |
| Retiring old values | 43.5% | −19.1 pts, and outdated values appear 19.7% of the time |
| Searching the key | 51.7% | −10.9 pts |
| Recency and importance | 58.6% | −4.0 pts |
| Skipping repeats | 60.8% | −1.8 pts |
| Always-on constraints | 62.3% | −0.3 pts, but the allergy is in the prompt only 15% of the time instead of 100% |

The always-on constraints cost almost nothing in accuracy and are what keep the allergy
visible while you plan a dinner.

## What this does not show

- **Dev and test disagree, and that's the most important number here.** On the dev wordings
  I built against, Keel scored 92.5%; on the held-out wordings, 62.6%. The retriever handles
  questions that name their topic ("Which city am I based in these days?": 100%) and fails
  on ones that only imply it ("Who should I book the anniversary dinner with?": 0%; "What
  should I look for in the parking lot?": 11%). 14 of the 32 test wordings scored below 50%,
  and 18 scored 85% or more
  ([breakdown](reports/metrics/question_breakdown_test.json)). A hand-written synonym list
  can't bridge that gap. The agent has a `recall` tool it can call with its own rephrasing,
  which should help, but only the live check can measure that.
- **Forgetting depends on consistent keys.** If the model saves "moved to Austin" under
  `residence` instead of `home_city`, nothing is retired. When I renamed the key on every
  update, Keel's clean-hit rate on changed details fell from 67.8% to 26.1%, still above
  keyword search (5.5%) but far from the clean case. Showing the model its existing keys is
  meant to prevent this; how often a real model complies is untested.
- **The keyword baseline is plain BM25**, not an embedding model. An embedding retriever
  would handle paraphrase better, for both the baseline and Keel. That comparison is the
  obvious next step.
- **The users are synthetic** and every memory is written perfectly. The numbers describe
  retrieval, not how well a model decides what to remember.
- **The end-to-end check hasn't been run.** `keel eval-live` has the model answer each
  question from each method's memories and grades the answer. It costs about 1,440 API
  calls for 30 users and needs a key, so it isn't in CI. Until it runs, I haven't shown
  that better retrieval gives better answers, only that it gives the model a better chance.
- **Two bugs were fixed after the test run.** The original stemmer treated "lives" and
  "live", or "siblings" and "sibling", as different words. I switched to the standard Snowball
  stemmer and reran. The pre-registered run gave Keel 59.7% and keyword search 8.1%, with the
  same verdict. Both runs are kept; details are in the plan's
  [change log](docs/ANALYSIS_PLAN.md#change-log).

## Running it

```bash
cd "End to end data science project/keel-personal-agent"
python -m pip install -e ".[dev,app]"

keel --db ~/.keel/demo.db demo          # load a made-up user to explore
keel --db ~/.keel/demo.db brief         # their morning brief
keel --db ~/.keel/demo.db memories --all
KEEL_DB=~/.keel/demo.db streamlit run app/app.py

export ANTHROPIC_API_KEY=...             # to talk to it
keel chat                                # /brief, /memories, /approvals, /quit
keel approvals && keel approve 1         # approved emails are written to ~/.keel/outbox

python scripts/run_benchmark.py          # the benchmark; writes reports/metrics/
python scripts/question_breakdown.py     # hit rate per question wording
keel eval-live --personas 30 --yes       # end-to-end check (costs money)
```

The brief, memory view, approvals and the whole test suite work without a key. Chat uses
Claude through the Anthropic SDK with adaptive thinking, prompt caching (the system prompt
and tool list never change, and per-turn context goes in the user message), server-side
refusal fallbacks and web search. Email approval writes an `.eml` file instead of sending
through a mail account. Connecting a real mailbox is left to the user; this project is
about the gate in front of it.

Run the checks with `make check` (Ruff, mypy, pytest). The 50 tests use a scripted model,
so they need no key and cost nothing.

## Project layout

```text
src/keel/
  agent.py            the tool-use loop, context block, end-of-session reflection
  model.py            Claude client, and a scripted model for tests
  memory/             store with supersession, retrievers, text processing
  tools/              tool registry (risk levels, validation, audit) and built-in tools
  approvals.py        the queue outward actions go through
  briefing.py         the daily brief
  evaluation/         synthetic users, benchmark, report tables, live check
  cli.py, demo.py
app/app.py            Streamlit: chat, today, memory, goals, approvals, activity
configs/eval.toml     benchmark settings, fixed before the test run
docs/ANALYSIS_PLAN.md the plan, pass rule and change log
reports/metrics/      every number in this README
```

*Built with:* Python, the Claude API (tool use, adaptive thinking, prompt caching, web
search), SQLite, BM25, NumPy, Streamlit, pytest, Ruff, mypy, GitHub Actions.
