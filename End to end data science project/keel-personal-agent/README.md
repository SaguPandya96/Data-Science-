# Keel: A Personal AI Agent That Remembers, and Knows When to Forget

A personal agent is only personal if it remembers you. It also has to forget: when you
move from Denver to Austin, it should stop suggesting restaurants in Denver. I built Keel,
a personal agent in the spirit of this year's consumer agents, to find out one thing: does
its memory put the right, *current* fact in front of the model more often than the simple
ways of doing it?

Short answer: **yes, by a wide margin, with limits worth knowing.** On held-out questions
from 200 synthetic users, Keel put the current answer in the prompt, with no outdated
version beside it, **74.9%** of the time (95% CI 74.5 to 75.3). Transformer embedding
search managed 44.9%, keyword search 20.4%, and showing the newest memories 4.6%. Tuning the
encoder's weight added another **3.0 points** on a fresh set, and a larger encoder another
**3.8**. The limits:
questions that only hint at their topic ("What should I tell the valet to bring around?")
are still the weak spot, and forgetting depends on the model labeling updates
consistently.

[How the agent works](docs/AGENT.md) · [Analysis plan and change log](docs/ANALYSIS_PLAN.md)

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
- **Asks before acting on your behalf.** Anything that reaches another person, such as
  sending an email, becomes a pending approval. The agent can't approve its own requests.
  Every tool call is written to an audit log.

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
   synonym expansion ("live" also matches "based", "home", "city"), plus similarity from a
   transformer sentence encoder, plus small boosts for recent and important memories.
   Repeats of the same small talk are skipped.

The model sees the keys already in use, so when you say "I moved to Austin" it can save the
new value under `home_city` and retire Denver.

Sentence similarity comes from `all-mpnet-base-v2`, a transformer encoder run on CPU with
ONNX Runtime. It is downloaded once (about 400 MB, checked against a pinned SHA-256) and
cached. Offline, or without the `transformer` extra, Keel falls back to
[WordLlama](https://github.com/dleemiller/WordLlama) embeddings, which ship inside their
Python package, and without those to the BM25 hybrid alone.

## Measuring it

### The benchmark

The benchmark generates **200 users**, each with 16 personal details spread over 12 weekly
sessions: city, employer, diet, allergy, partner, pet, and so on. 45% of details change
later (a move, a new job), and a quarter of those change again. Every session adds 4 to 9
small-talk memories, many deliberately sharing words with the questions: asking about
restaurants in *another* city, or a cover letter for a job somewhere else. That gives about
100 memories per user. At the end, each user asks about each detail, and each method may
put **5 memories** in the prompt.

The main measure is a **clean hit**: the current value is in the prompt and no earlier value
is. A model shown both "lives in Denver" and "moved to Austin" can get it wrong; a clean hit
leaves nothing to get wrong.

I write both the retriever and the questions, which is a real risk of tuning one to the
other. So the question wordings are split: I build against a *dev* set, and each round is
judged on a held-out set that is written, committed and then run once. The plan and pass
rule for each round were committed before its held-out run, so the order is visible in
git history.

### Round 1: memory without embeddings

| Method | Clean hit | Outdated value shown | Allergy in prompt* | Prompt tokens |
| --- | --- | --- | --- | --- |
| No memory | 0.0% | 0.0% | 0% | 0 |
| Newest 5 memories | 4.6% (3.9 to 5.2) | 0.0% | 0% | 84 |
| Keyword search (BM25) | 9.8% (8.8 to 10.8) | 5.0% | 1% | 29 |
| **Keel** | **62.6% (61.4 to 63.9)** | **0.0%** | **100%** | 75 |
| Every memory | 62.2% (60.6 to 63.8) | 37.8% | 100% | 1,664 |

\*For questions about something other than the allergy. 95% intervals are from a bootstrap
over users. Source: [`reports/metrics/benchmark_test.md`](reports/metrics/benchmark_test.md).

**The pre-registered rule passed.** Keel beat the newest memories by **58.1 points** (95% CI
56.7 to 59.5) and keyword search by **52.8 points** (51.3 to 54.3). For details that had
changed, keyword search mostly surfaced the old value: 5.5% clean hits against Keel's 67.8%.

Putting every memory in the prompt ties Keel while using **22 times the tokens**, and for
details that changed it *always* shows the outdated value alongside the current one.

What each part contributes (removing one at a time):

| Keel without | Clean hit | Change |
| --- | --- | --- |
| (nothing removed) | 62.6% | |
| Synonym expansion | 34.7% | −27.9 pts |
| Retiring old values | 43.5% | −19.1 pts, and outdated values appear 19.7% of the time |
| Searching the key | 51.7% | −10.9 pts |
| Recency and importance | 58.6% | −4.0 pts |
| Skipping repeats | 60.8% | −1.8 pts |
| Always-on constraints | 62.3% | −0.3 pts, but the allergy is in the prompt only 15% of the time instead of 100% |

Round 1 also showed where Keel fails. On the dev wordings it scored 92.5%, but only 62.6%
on held-out ones. The misses were questions that imply their topic without naming it: "Who
should I book the anniversary dinner with?" scored 0%.

### Round 2: adding embedding search

To test whether embeddings fix that, I wrote a new held-out set before running anything.
For each detail it has one **direct** wording ("What car do I own?") and one **indirect**
wording ("What model should I tell the mechanic I'm bringing in?"). The embedding weight
was chosen on dev only.

| Method | Clean hit | Direct | Indirect | Outdated shown |
| --- | --- | --- | --- | --- |
| Newest 5 memories | 4.6% | 4.6% | 4.6% | 0.0% |
| Keyword search (BM25) | 18.9% | 30.7% | 7.2% | 11.0% |
| Embedding search | 39.2% | 55.6% | 22.8% | 20.1% |
| Keel, round 1 | 64.4% | 90.9% | 37.8% | 0.0% |
| **Keel with embeddings** | **69.1% (68.6 to 69.6)** | **95.5%** | **42.7%** | **0.0%** |

Source: [`reports/metrics/round2.md`](reports/metrics/round2.md).

**The pre-registered rule passed**: embeddings added **4.7 points** (95% CI 4.3 to 5.1) over
round 1 Keel, on both direct and indirect questions. Embedding search on its own is a much
weaker memory: it has no idea which value is current, so it shows outdated values 20% of
the time. But indirect questions only rose from 37.8% to 42.7%. WordLlama averages word
vectors, so it can't read a question as a whole.

### Round 3: an end-to-end check with Claude

Designed and pre-registered, but not yet run; see below.

### Round 4: a transformer sentence encoder

So I tried two small transformer encoders, `all-MiniLM-L6-v2` and `bge-small-en-v1.5`, with
a third held-out set written before either was run. The encoder and its weight were chosen
together on dev: MiniLM won.

| Method | Clean hit | Direct | Indirect | Outdated shown |
| --- | --- | --- | --- | --- |
| Newest 5 memories | 4.6% | 4.6% | 4.6% | 0.0% |
| Keyword search (BM25) | 20.4% | 34.4% | 6.4% | 9.8% |
| Embedding search (WordLlama) | 35.0% | 54.0% | 16.0% | 17.2% |
| Embedding search (MiniLM) | 44.9% | 61.7% | 28.1% | 25.3% |
| Keel, round 1 | 64.3% | 98.6% | 30.1% | 0.0% |
| Keel with WordLlama (round 2) | 66.9% | 98.8% | 35.0% | 0.0% |
| **Keel with MiniLM** | **74.9% (74.5 to 75.3)** | **100.0%** | **49.8%** | **0.0%** |

Source: [`reports/metrics/round4.md`](reports/metrics/round4.md).

**The pre-registered rule passed**: MiniLM added **8.0 points** (95% CI 7.5 to 8.6) over Keel
with WordLlama, and **14.8 points** on indirect questions. It is now the agent's default.
Some indirect wordings jumped ("Who could be my kids' aunt or uncle on my side?" went from
8% to 100%), but two got worse ("What should I tell the valet to bring around?" fell from
14% to 0%). It is also about 80 times slower than WordLlama, at about 255 texts a second on
one CPU thread. For one person's memory that's fine, because each text is embedded once
and cached.

### Round 5: tuning the encoder's weight

Round 4 chose MiniLM's weight on dev, where scores were still rising at 2.0, the largest
value tried. Dev was also near its ceiling (98.8%), so it couldn't separate weights well.
Round 5 tried weights from 1 to 16, chose one on the three question sets I had already
seen (dev and the round 2 and round 4 held-out sets), and judged it once on a fourth,
fresh set.

| Weight | Clean hit | Direct | Indirect |
| --- | --- | --- | --- |
| 2.0 (round 4) | 84.4% | 100.0% | 68.8% |
| **6.0 (chosen)** | **87.4% (87.0 to 87.8)** | **100.0%** | **74.8%** |

Source: [`reports/metrics/round5.md`](reports/metrics/round5.md).

**The pre-registered rule passed**: 6.0 added **3.0 points** (95% CI 2.6 to 3.4), all of it
on indirect questions, so the agent now uses it. This fourth set is easier than round 4's
(the old default scores 84.4% here against 74.9% there), so compare the rounds by their
paired gains, not their absolute rates. And the gain isn't universal: on the round 4 set,
larger weights were slightly worse.

### Round 6: a larger encoder

MiniLM has 22 million parameters. Round 6 tried two encoders about five times larger,
`bge-base-en-v1.5` and `all-mpnet-base-v2`, alongside MiniLM, each at weights from 2 to 12.
The pair was chosen on the four question sets already seen and judged once on a fifth,
fresh set. Because a larger model means a bigger download and slower embedding, I set the
bar in advance: adopt it only if the gain's 95% interval starts at 1 point or more.

| Encoder | Clean hit | Direct | Indirect |
| --- | --- | --- | --- |
| MiniLM, weight 6 (round 5) | 85.5% | 100.0% | 71.0% |
| **mpnet, weight 12 (chosen)** | **89.3% (89.0 to 89.5)** | **100.0%** | **78.5%** |

Source: [`reports/metrics/round6.md`](reports/metrics/round6.md).

**It cleared the bar**: mpnet added **3.8 points** (95% CI 3.4 to 4.2), and **7.6** on
indirect questions, so it is now the default. Size alone didn't do it: `bge-base-en-v1.5`,
the same size, scored below MiniLM on every weight. And the tuning sets barely separated
mpnet from MiniLM (0.2 points), so most of the evidence comes from one fresh set. The cost
is a 400 MB first download and embedding about seven times slower than MiniLM (about 50
texts a second on one CPU thread), which for one person's memory is a few seconds once.

## What this does not show

- **Indirect questions are still the weak spot.** Direct questions reach 100%, but with the
  current encoder about one indirect question in five still misses (78.5% on the latest
  set). The agent also has a `recall` tool it can call with its own rephrasing, which only
  the live check can measure.
- **The current weight is at the top of the range tried.** Round 6 chose weight 12, the
  largest it tested, so a higher weight might do better still.
- **Forgetting depends on consistent keys.** If the model saves "moved to Austin" under
  `residence` instead of `home_city`, nothing is retired. When I renamed the key on every
  update, Keel's clean-hit rate on changed details fell from 67.8% to 26.1%, still above
  keyword search (5.5%) but far from the clean case. Showing the model its existing keys is
  meant to prevent this; how often a real model complies is untested.
- **The users are synthetic** and every memory is written perfectly. The numbers describe
  retrieval, not how well a model decides what to remember.
- **The end-to-end check hasn't been run.** Its design and pass rule are fixed in the
  [analysis plan](docs/ANALYSIS_PLAN.md#round-3-end-to-end-check), amended before any run
  to test the agent's default retriever: the model answers held-out questions from each method's
  memories (including every memory at once) and the answers are graded automatically. It
  makes up to 5,760 API calls for 30 users and needs a key, so it isn't in CI. Until it
  runs, I haven't shown that better retrieval gives better answers, only that it gives the
  model a better chance.
- **Two bugs were fixed after the round 1 test run.** The original stemmer treated "lives"
  and "live", or "siblings" and "sibling", as different words. I switched to the standard
  Snowball stemmer and reran. The pre-registered run gave Keel 59.7% and keyword search
  8.1%, with the same verdict. Both runs are kept; details are in the plan's
  [change log](docs/ANALYSIS_PLAN.md#change-log).

## Running it

```bash
cd "End to end data science project/keel-personal-agent"
python -m pip install -e ".[dev,app,embed,transformer]"

keel --db ~/.keel/demo.db demo          # load a made-up user to explore
keel --db ~/.keel/demo.db brief         # their morning brief
keel --db ~/.keel/demo.db memories --all
KEEL_DB=~/.keel/demo.db streamlit run app/app.py

export ANTHROPIC_API_KEY=...             # to talk to it
keel chat                                # /brief, /memories, /approvals, /quit
keel approvals && keel approve 1         # approved emails are written to ~/.keel/outbox

python scripts/run_benchmark.py          # round 1; writes reports/metrics/
python scripts/run_round2.py             # round 2 (WordLlama embeddings)
python scripts/run_round4.py             # round 4 (transformer encoders; downloads ~160 MB once)
python scripts/run_round5.py             # round 5 (weight grid for MiniLM)
python scripts/run_round6.py             # round 6 (larger encoders; downloads ~600 MB once)
python scripts/question_breakdown.py     # round 1 hit rate per question wording
keel eval-live --personas 10 --yes       # trial run of the end-to-end check (costs money)
keel eval-live --yes                     # the planned run: 30 users, 6 methods
```

The brief, memory view, approvals and the whole test suite work without a key. Chat uses
Claude through the Anthropic SDK with adaptive thinking, prompt caching, server-side
refusal fallbacks and web search. Email approval writes an `.eml` file instead of sending
through a mail account. Connecting a real mailbox is left to the user; this project is
about the gate in front of it.

Set `KEEL_OFFLINE=1` to forbid model downloads; Keel then uses whatever is already on disk.

Run the checks with `make check` (Ruff, mypy, pytest). The tests use a scripted model and
never download anything, so they need no key and cost nothing. CI also reruns every
benchmark round and fails if any committed number changes.

## Project layout

```text
src/keel/
  agent.py            the tool-use loop, context block, end-of-session reflection
  model.py            Claude client, and a scripted model for tests
  memory/             store with supersession, retrievers, embeddings, text processing
  tools/              tool registry (risk levels, validation, audit) and built-in tools
  approvals.py        the queue outward actions go through
  briefing.py         the daily brief
  evaluation/         synthetic users, benchmark rounds, report tables, live check
  cli.py, demo.py
app/app.py            Streamlit: chat, today, memory, goals, approvals, activity
configs/eval.toml     benchmark settings, fixed before each held-out run
docs/AGENT.md         how the agent is designed
docs/ANALYSIS_PLAN.md the plan, pass rules and change log
reports/metrics/      every number in this README
```

*Built with:* Python, the Claude API (tool use, adaptive thinking, prompt caching, web
search), SQLite, BM25, mpnet and MiniLM via ONNX Runtime, WordLlama, NumPy, Streamlit, pytest, Ruff,
mypy, GitHub Actions.
