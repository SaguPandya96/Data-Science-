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
encoder's weight added another **3.0 points** on a fresh set, a larger encoder another
**3.8**, a cross-encoder reranker another **2.3**, and a search-trained encoder another
**1.3**. The limits:
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
4. **A second look at the short list.** The 20 best candidates are rescored by a
   cross-encoder, which reads the question and each memory together, and its score is
   blended with the hybrid score.

The model sees the keys already in use, so when you say "I moved to Austin" it can save the
new value under `home_city` and retire Denver.

Sentence similarity comes from an int8 version of `e5-large-v2`, a transformer encoder
trained for search and run on CPU with ONNX Runtime; questions get a `query: ` prefix and memories `passage: `, as
the model expects. The second look comes from an int8 version of `ms-marco-MiniLM-L-6-v2`. Both
are downloaded once (about 340 MB and 23 MB, each checked against a pinned SHA-256) and
cached. If only the
encoder is available, Keel skips the second look. Offline, or without the `transformer`
extra, Keel falls back to
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

### Round 7: a second look with a cross-encoder

The encoder compares a question and a memory through two separate vectors. A cross-encoder
reads them together, which is slower but better at implied links. On question sets already
seen, the right memory was in the top 20 candidates far more often than in the 5 shown
(86% to 93% against 49% to 79% for indirect wordings), so round 7 let a cross-encoder
reorder that short list. Two ms-marco MiniLM rerankers were tried, on short lists of 10,
20 or 30, blended with Keel's score at five weights or used alone. The setting was chosen
on five question sets already seen and judged once on a sixth, fresh set, with the same
1-point bar as round 6.

| Retriever | Clean hit | Direct | Indirect |
| --- | --- | --- | --- |
| mpnet, weight 12 (round 6) | 85.8% | 100.0% | 71.6% |
| **+ `ms-marco-MiniLM-L-6-v2` on the top 20, weight 2 (chosen)** | **88.1% (87.7 to 88.5)** | **99.9%** | **76.2%** |

Source: [`reports/metrics/round7.md`](reports/metrics/round7.md).

**It cleared the bar**: the reranker added **2.3 points** (95% CI 1.9 to 2.6), and **4.6**
on indirect questions, so it is now the default. Two things worth knowing. Blending
mattered: on the tuning sets, ordering the short list by the cross-encoder alone was
*worse* than not reranking at all, at every list size (84.2% to 86.5% against 87.3%). It
was trained on web search rather than personal notes, which may be why it does better as
a correction than as the judge. And it adds time to every question: about 130 ms on one
CPU thread for 20 candidates (round 8 halved that). The 12-layer reranker was slower and no better.

### Round 8: smaller models

Both models have standard int8 versions about four times smaller. Round 8 asked a
different question from the earlier rounds: not whether they are better, but whether they
are **no worse**. A variant had to stay within 0.5 points on the six question sets already
seen, and then, on a seventh fresh set, rule out a loss of more than 1 point.

| Model | Size | Result |
| --- | --- | --- |
| int8 reranker | 91 → 23 MB | **adopted**: 91.7% against 91.5% on the fresh set (+0.1 points, 95% CI −0.0 to +0.3); reranking time halved, 133 → 65 ms per question |
| int8 encoder | 436 → 110 MB | rejected: about 2.3 points lower on the already-seen sets at every weight tried (8, 12, 16) |

Source: [`reports/metrics/round8.md`](reports/metrics/round8.md).

So the reranker is now the int8 one, and the encoder stays at full precision. The encoder
is where the meaning of a memory is captured once and reused for every question, and
squeezing it to int8 visibly blurred its vectors (cosine 0.82 to 0.92 against the full
model on neutral test sentences); the reranker only reorders a short list, and int8 cost
it nothing measurable.

### Round 9: a smaller encoder

After round 8, almost all of the first download is the encoder (mpnet, 436 MB). Round 9
asked whether a full-precision encoder a third of the size or less, with the reranker on
top, is no worse than mpnet. Four were tried, each at weights from 4 to 48 chosen on the
seven question sets already seen. Any within 0.5 points of mpnet would have gone on to a
fresh set.

| Encoder | Size | Best weight | Already-seen sets | vs mpnet (88.9%) |
| --- | --- | --- | --- | --- |
| `all-MiniLM-L12-v2` | 133 MB | 4 | 87.9% | −1.0 |
| `all-MiniLM-L6-v2` | 90 MB | 4 | 87.8% | −1.1 |
| `gte-small` | 133 MB | 24 | 87.6% | −1.3 |
| `bge-small-en-v1.5` | 133 MB | 4 | 85.8% | −3.1 |

Source: [`reports/metrics/round9.md`](reports/metrics/round9.md).

**None qualified, so mpnet stays** and the fresh set was not used. The reranker did close
most of the gap: before it existed, MiniLM trailed mpnet by 3.8 points (round 6); with it,
by about 1.1, still more than the 0.5-point bar set in advance.

Three of the four did best at the lowest weight tried (4), so **round 10** tried the two
MiniLM models at weights 0.5 to 4 under the same rules. Both got worse below 4 (MiniLM-L6
87.2% to 87.8%, MiniLM-L12 87.4% to 87.9%), so weight 4 was the real peak and the gap to
mpnet stands at about a point. Source: [`reports/metrics/round10.md`](reports/metrics/round10.md).

### Round 11: a better encoder

If smaller does not work, does *different*? Round 11 tried six encoders about mpnet's size
under the reranker, several trained specifically for question-to-passage search, with the
question and memory prefixes each expects. Each chose its weight on the seven question
sets already seen; the best was then tested once on a fresh set.

| Encoder | Already-seen sets (best weight) |
| --- | --- |
| **`e5-base-v2`** | **90.4% (48)** |
| `multi-qa-mpnet-base-dot-v1` | 90.1% (12) |
| `nomic-embed-text-v1.5` | 89.8% (24) |
| `all-mpnet-base-v2` (shipped) | 88.9% (12) |
| `gte-base` | 88.6% (32) |
| `bge-base-en-v1.5` | 87.1% (4) |
| `snowflake-arctic-embed-m-v1.5` | 86.4% (4) |

On the fresh set, e5 scored **92.4% against 91.1%** for mpnet: **+1.3 points** (95% CI 0.7
to 1.8), all of it on indirect questions (+2.6; 84.9% against 82.3%). It is the same size
and speed as mpnet, so it is now the default. The weight it chose, 48, is the top of the
range tried, though 32 scored almost the same. Source:
[`reports/metrics/round11.md`](reports/metrics/round11.md).

### Round 12: a larger encoder

Is a bigger model worth it? Round 12 tried three encoders about three times e5-base's size,
with weights up to 96, on the eight question sets already seen. Because they cost more to
download and run, the best had to beat e5-base on a fresh set by at least 1 point (lower
end of the 95% interval).

| Encoder | Already-seen sets (weight) |
| --- | --- |
| **`e5-large-v2`** | **91.2% (64)** |
| `e5-base-v2` (shipped) | 90.6% (48) |
| `mxbai-embed-large-v1` | 87.9% (8) |
| `bge-large-en-v1.5` | 86.9% (4) |

On the fresh set, e5-large scored **93.9% against 90.1%** for e5-base: **+3.8 points**
(95% CI 3.4 to 4.2), all of it on indirect questions (+7.6; 87.8% against 80.2%). That
clears the 1-point bar, so it is now the default. The cost: a 1.3 GB first download
instead of 440 MB, and embedding about a quarter as fast on one CPU thread. Source:
[`reports/metrics/round12.md`](reports/metrics/round12.md).

### Round 13: another encoder of the same size

Round 13 tried three more encoders of e5-large-v2's size under the same rules as round 11,
on the nine question sets already seen:

| Encoder | Already-seen sets (weight) |
| --- | --- |
| **`e5-large-v2`** (shipped) | **91.5% (64)** |
| `gte-large` | 89.5% (32) |
| `snowflake-arctic-embed-l` | 88.5% (24) |
| `e5-large` (first version) | 86.8% (4) |

**None came close, so e5-large-v2 stays**, and the fresh question set was not used. Source:
[`reports/metrics/round13.md`](reports/metrics/round13.md).

### Round 14: a still larger encoder

Round 14 tried three encoders of about 2.2 GB, built on a model with a much larger vocabulary
but the same depth as e5-large-v2, so about as fast. A winner had to clear the 1-point bar
used for round 12, because of the larger download.

| Encoder | Already-seen sets (weight) |
| --- | --- |
| **`e5-large-v2`** (shipped) | **91.5% (64)** |
| `multilingual-e5-large` | 91.3% (64) |
| `multilingual-e5-large-instruct` | 90.3% (64) |
| `snowflake-arctic-embed-l-v2.0` | 86.6% (4) |

**None beat it, so e5-large-v2 stays**, and the fresh question set is still unused. Source:
[`reports/metrics/round14.md`](reports/metrics/round14.md).

### Round 15: a newer design

Every encoder so far was built on the original BERT or XLM-RoBERTa designs. Round 15 tried
four built on newer ones (ModernBERT, and Alibaba's gte-v1.5), two of them base size and so
about two and a half times as fast as e5-large-v2.

| Encoder | Already-seen sets (weight) |
| --- | --- |
| **`e5-large-v2`** (shipped) | **91.5% (64)** |
| `gte-modernbert-base` | 90.2% (24) |
| `modernbert-embed-base` | 88.9% (8) |
| `modernbert-embed-large` | 88.4% (8) |
| `gte-large-en-v1.5` | 88.3% (12) |

**None beat it, so e5-large-v2 stays**, and the fresh question set is still unused.
gte-modernbert-base is the fastest option within 1.3 points. Source:
[`reports/metrics/round15.md`](reports/metrics/round15.md).

### Round 16: a smaller copy of the same encoder

e5-large-v2 is a 1.3 GB download and the slowest encoder Keel has shipped. Round 16 tried
its int8 copy: a quarter of the size and five to six times as fast on one CPU thread. Since
it costs less, it only had to be no more than 1 point worse on the fresh question set (the
rule round 8 used for the int8 reranker).

| Encoder | Already-seen sets (weight) | Fresh set |
| --- | --- | --- |
| `e5-large-v2` | 91.5% (64) | 93.4% |
| **`e5-large-v2` (int8)** (shipped) | **91.3% (64)** | **93.0%** |

The int8 copy scored **0.4 points lower** on the fresh set (95% CI −0.7 to −0.0), well
inside the 1-point margin, so **it is now the default**. The first download falls from
about 1.3 GB to 340 MB. Source: [`reports/metrics/round16.md`](reports/metrics/round16.md).

## What this does not show

- **Indirect questions are still the weak spot.** Direct questions are at 99.5% or above,
  but about one indirect question in eight still misses (87.8% on the latest set; how hard a
  set is varies, so compare methods within a set, not across sets). The agent also has a
  `recall` tool it can call with its own rephrasing, which only the live check can
  measure.
- **Round 12's weight rule was amended after the run.** The first run chose weight 96,
  but 64 was only about two questions behind out of 25,600, and on CI's processor the two
  swapped places. Weights that close now count as tied and the smaller one wins, so 64 is
  chosen everywhere. The fresh set had already been used once at 96 (93.7%, +3.5 points);
  both runs adopt e5-large, but the 93.9% is a second look, not a first. Details in the
  [analysis plan](docs/ANALYSIS_PLAN.md#amendment-after-the-first-run-ties-between-weights).
- **The tuning gain was smaller than the fresh-set gain.** e5-large led e5-base by 0.6
  points on the sets already seen but by 3.8 on the fresh one. The fresh set's indirect
  wordings seem to be harder for e5-base than earlier sets were, so the size of the gain
  depends on the questions; its direction held on both.
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
python scripts/run_round7.py             # round 7 (rerankers; downloads ~225 MB once, ~30 min)
python scripts/run_round8.py             # round 8 (int8 models; downloads ~135 MB once, ~20 min)
python scripts/run_round9.py             # round 9 (smaller encoders; downloads ~490 MB once, ~20 min)
python scripts/run_round10.py            # round 10 (MiniLM below weight 4; ~20 min)
python scripts/run_round11.py            # round 11 (six other encoders; downloads ~2.7 GB once, ~35 min)
python scripts/run_round12.py            # round 12 (larger encoders; downloads ~4 GB once, ~50 min)
python scripts/run_round13.py            # round 13 (other large encoders; downloads ~4 GB once, ~40 min)
python scripts/run_round14.py            # round 14 (2.2 GB encoders; downloads ~7 GB once, ~45 min)
python scripts/run_round15.py            # round 15 (newer designs; downloads ~4.5 GB once, ~45 min)
python scripts/run_round16.py            # round 16 (int8 e5-large-v2; downloads ~1.7 GB once, ~30 min)
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
benchmark round and fails if any committed decision changes or any number moves by more
than 0.5 points (rounds 1 and 2 must match exactly; rounds 4 to 15 run transformer models,
whose last bits vary with the CPU).

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
search), SQLite, BM25, e5, mpnet, MiniLM and an ms-marco cross-encoder via ONNX Runtime,
WordLlama, NumPy, Streamlit, pytest, Ruff, mypy, GitHub Actions.
