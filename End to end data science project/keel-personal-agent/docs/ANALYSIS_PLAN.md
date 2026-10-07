# Analysis plan

Written, together with `configs/eval.toml`, before the first run on the test split. Choices
made after seeing results are listed in the change log at the bottom and labeled that way.

## The question

A personal agent only feels personal if it remembers you. It also has to forget: when you
move to a new city, it should stop planning around the old one. Does Keel's memory put the
right, *current* fact in front of the model more often than two simple alternatives?

## What is measured

The benchmark isolates retrieval. Each synthetic user has a history of memories written over
12 weekly sessions. At the end the user asks one question per personal detail ("Which city
am I based in these days?"). Each arm chooses at most `k = 5` memories to put in the prompt.

For each question:

- **hit**: the memory holding the current value is in the prompt.
- **stale**: a memory holding an *earlier* value of the same detail is in the prompt.
- **clean hit** (primary): hit and not stale. A model shown both "lives in Denver" and
  "lives in Austin" can answer either; a clean hit leaves nothing to get wrong.
- **context tokens**: estimated prompt size of the chosen memories (characters / 4).
- **allergy shown**: for questions about anything other than the allergy, whether the
  user's allergy is in the prompt. Added after the first dev run; see the change log.

The benchmark gives every arm perfect extraction: memories are written exactly as the
generator states them. So the numbers describe retrieval, not how well a model writes
memories. The end-to-end check with a real model is separate (`keel eval-live`) and is not
part of the pass rule.

## Arms

| Arm | What it puts in the prompt |
| --- | --- |
| `none` | Nothing. Floor. |
| `recent` | The `k` newest memories. The simple alternative most chat apps start with. |
| `lexical` | BM25 top `k` over every memory ever written. Standard retrieval-augmented memory. |
| `keel` | Up to 2 always-on constraints (allergies, hard limits), then hybrid top-`k` over *active* memories: BM25 over each memory's key and text with light stemming and synonym expansion, plus recency and importance, skipping repeats. Memories replaced by a newer value under the same key are excluded. |
| `full` | Every memory. Not a real option at scale; shown for its token cost. |

## Data

`keel.evaluation.scenarios` generates 200 personas with seed `20260930`. Each has 16
personal details (city, employer, diet, allergy, partner, pet and so on). 45% of details
change value later in the history, and 25% of those change again. Every session also adds
4 to 9 small-talk memories, many deliberately sharing words with the questions (asking
about restaurants in another city, for example).

**Template split.** I write both the retriever and the question templates, so the retriever
could end up tuned to my phrasing. To guard against that, every detail has separate *dev*
and *test* question templates. I build and debug against dev only. The test templates are
run once, after this plan and the synonym list are committed.

## Statistics

Rates are averaged per persona, and 95% intervals come from a bootstrap over personas
(2,000 resamples), since questions from one persona are not independent. Differences
between arms are paired: the same personas, questions and memories.

## Pass rule

Keel passes if, on the **test** templates at `k = 5`, its clean-hit rate beats both `recent`
and `lexical`, with the 95% interval of each paired difference entirely above zero.

If it fails, that is the result, and the README says so.

## Reported but not part of the pass rule

- The same metrics on the subset of details that changed value (where staleness can happen).
- An ablation: Keel without supersession, without recency and importance, without synonym
  expansion, without key indexing, without duplicate suppression, and without the
  always-on constraints.
- **Key noise.** Supersession relies on the model reusing the same key ("home_city") when a
  detail changes. In real use it may not. For 0%, 25%, 50% and 100% of updates the benchmark
  writes the new value under a different key ("residence"), which Keel cannot match, and
  reports how much of the advantage survives.
- A `k` sweep over 1, 3, 5 and 10.

## Change log

All made after the first two runs on the **dev** templates and before any run on the test
templates. The pass rule, arms, data generator, `k` and the baselines did not change.

1. **Key indexing.** On dev, Keel missed memories whose text never names the topic
   ("Adopted a parrot named Kiwi" for "What pet do I have?"). Keel now searches the
   memory's key along with its text. `lexical` still searches text only, as a standard
   retrieval memory would.
2. **Duplicate suppression.** Repeated small talk ("Asked how to clean a car's interior"
   three times) filled whole prompts. Keel now skips a memory whose words match one it
   already chose.
3. **Constraint cap.** At `k = 1` the always-on allergy took the only slot. Always-on
   constraints now take at most half of `k` (none at `k = 1`, one at 3, two at 5).
4. **Allergy shown** metric added, so the ablation shows what the always-on constraints
   buy as well as what they cost.
5. Both new mechanisms were added to the ablation.

Dev clean-hit rate for `keel` at `k = 5` was 71.4% before changes 1 and 2 and 92.5% after.
The test split had not been run at that point.

### After the test run

6. **Stemmer bugs.** The hand-written stemmer was inconsistent: "lives" became "liv" while
   "live" stayed "live" (found by a unit test), and "siblings" became "sibling" while
   "sibling" became "sibl" (found in the per-question breakdown, where "Tell me about my
   siblings" scored 0%). These are bugs, not tuning: the same word in two forms never
   matched, in every arm that searches text. I replaced the stemmer with the standard
   Snowball (Porter2) English stemmer rather than patching it further. The synonym list,
   weights, templates and everything else stayed as frozen.

   The run on the frozen code is kept unchanged in
   `reports/metrics/benchmark_test_preregistered.{json,md}`. The verdict is the same in
   both runs. Clean-hit rate on test at `k = 5`:

   | Code | `recent` | `lexical` | `keel` | `full` |
   | --- | --- | --- | --- | --- |
   | Frozen (pre-registered) | 4.6% | 8.1% | 59.7% | 62.2% |
   | Snowball stemmer (reported in the README) | 4.6% | 9.8% | 62.6% | 62.2% |

## Round 2: embedding search

Written after round 1 was finished and before any embedding code was run.

### The question

Round 1 showed Keel failing questions that only imply their topic ("Who should I book the
anniversary dinner with?"). Word matching can't connect "anniversary dinner" to "partner".
Does adding a sentence-embedding similarity to Keel's score fix that without giving up
what round 1 gained?

### Model

WordLlama (`l2_supercat`, 256 dimensions, MIT licence): static token embeddings taken from
a large language model's embedding table and trained for sentence similarity. It is small
(16 MB), runs on NumPy in milliseconds, and ships its weights inside the Python package, so
the benchmark runs offline and in CI. The obvious alternative, a transformer sentence
encoder such as MiniLM, needs a download from Hugging Face, which this build environment
blocks. The retriever takes any embedder, so a stronger model can be swapped in later.

### Arms

| Arm | What changes |
| --- | --- |
| `embedding` | Cosine similarity top `k` over every memory ever written. The standard dense retrieval memory, and the stronger baseline round 1 was missing. |
| `keel+embed` | Keel exactly as in round 1, plus `w × cosine similarity` (between the question and the memory's key and text) added to the hybrid score. |

`w` is chosen from {0.25, 0.5, 1.0, 2.0} by clean-hit rate on the **dev** wordings only.

### New held-out questions

The round 1 test wordings have been seen, and I know which ones fail, so they can't
judge a fix fairly. I wrote a new set before running anything: for each of the 16 details,
one **direct** wording that names the topic ("What car do I own?") and one **indirect**
wording that only implies it ("What model should I tell the mechanic I'm bringing in?").
They are in `HOLDOUT` in `keel.evaluation.scenarios`, committed before the first embedding
run. They are run once, after `w` is fixed.

### Pass rule

`keel+embed` passes if, on the new held-out wordings at `k = 5`, its clean-hit rate beats
round 1 `keel`, with the 95% persona-bootstrap interval of the paired difference entirely
above zero.

If it passes, it becomes the agent's default retriever. If not, the agent keeps round 1
Keel and the README reports the failure.

### Reported, not part of the rule

- Direct and indirect wordings separately.
- `keel+embed` against `embedding`, `lexical` and `recent`.
- All arms on the round 1 test wordings, labeled as already seen.


### Outcome

Run once, as planned. `w = 0.5` was chosen on dev (94.6%, tied with 1.0; the grid rule
takes the smaller weight). On the new held-out wordings at `k = 5`:

| Arm | Clean hit | Direct | Indirect |
| --- | --- | --- | --- |
| `embedding` | 39.2% | 55.6% | 22.8% |
| `keel` (round 1) | 64.4% | 90.9% | 37.8% |
| `keel+embed` | 69.1% | 95.5% | 42.7% |

`keel+embed` − `keel`: **+4.7 points** (95% CI +4.3 to +5.1). **Passed.** As planned, it is
now the agent's default retriever when WordLlama is installed. Full tables:
`reports/metrics/round2.md`.

The gain is real but small, and indirect questions remain the weak spot: four of the
sixteen indirect wordings still score 6% or less. A small static embedding model is not
enough to connect "Who else grew up in the same house as me?" with "Has a sister named
Mira." Embeddings on their own lose to Keel by 29.9 points: they have no notion of which
value is current, so they show outdated values 20.1% of the time.

## Round 3: end-to-end check

Written before the first live run.

### The question

Rounds 1 and 2 measured whether the right memory reaches the prompt. Does that turn into
right answers? A capable model might pick the newest of two conflicting memories on its
own, or might answer correctly from memories that miss the exact one the benchmark looks
for.

### Design

- **Users and questions:** the first 30 benchmark personas, with the round 2 held-out
  wordings (16 direct and 16 indirect per persona), so 960 questions per arm.
- **Arms:** `recent`, `lexical`, `embedding`, `keel` (round 1), `keel+embed` (the agent's
  default) and `full` (every memory). `full` is included because it tests the one thing
  retrieval benchmarks can't: whether the model resolves outdated values by itself.
- **Model:** `claude-opus-5-5` at low effort. The system prompt tells it to answer from
  the given memories only and to say it doesn't know otherwise. Each memory line shows its
  date.
- **Grading:** automatic. An answer is **correct** when it states the current value and no
  earlier one (names and numbers must all appear; other values must appear as a phrase).
  It is **stale** when it states an earlier value and not the current one.
- **Statistics:** per-persona accuracy, 95% persona-bootstrap intervals, paired
  differences as in the earlier rounds.

### Pass rule

`keel+embed` passes if its accuracy beats `embedding`, the standard alternative, with the
95% interval of the paired difference entirely above zero.

### Reported, not part of the rule

- Every arm's accuracy and stale-answer rate, for direct and indirect questions.
- `keel+embed` against `keel`, `lexical`, `recent` and `full`, with token cost.
- A sample of graded answers, so the automatic grading can be checked by eye.

## Round 4: transformer sentence encoders

Written after round 2 was finished and before any transformer model was run on benchmark
text.

### The question

Round 2's WordLlama embeddings lifted indirect questions only from 37.8% to 42.7%. WordLlama
averages static word vectors, so it can't read a question as a whole. Does a small
transformer sentence encoder, which does, close more of that gap?

### Encoders

Two widely used small encoders, both run on CPU with ONNX Runtime:

| Encoder | Size | Pooling | Why |
| --- | --- | --- | --- |
| `all-MiniLM-L6-v2` | 22M parameters, 384 dimensions | mean | The standard small sentence encoder |
| `bge-small-en-v1.5` | 33M parameters, 384 dimensions | CLS | Trained for retrieval; stronger on retrieval benchmarks |

Hugging Face is blocked in my build environment, so the ONNX exports are downloaded from
Qdrant's public model mirror (the source `fastembed` uses). Each archive is pinned by
SHA-256 in `configs/eval.toml`. No query instruction prefix is used for BGE (v1.5 is
designed to work without one), and every text is embedded the same way.

### Arms

| Arm | What it is |
| --- | --- |
| `transformer` | Cosine similarity top `k` over every memory ever written, with the chosen encoder |
| `keel+transformer` | Keel with the chosen encoder in place of WordLlama |

Plus, unchanged: `recent`, `lexical`, `embedding` (WordLlama), `keel` (round 1) and
`keel+embed` (the current default).

The encoder and the weight `w` from {0.25, 0.5, 1.0, 2.0} are chosen together by clean-hit
rate on the **dev** wordings only. Ties go to the smaller weight, then to MiniLM.

### New held-out questions

The round 2 held-out set has been run and I have seen its per-question results. A third
set, `HOLDOUT2` in `keel.evaluation.scenarios`, was written and committed before any
transformer run: one direct and one indirect wording per detail, none repeating an earlier
wording. It is run once, after the encoder and `w` are fixed.

### Pass rule

`keel+transformer` passes if, on the `HOLDOUT2` wordings at `k = 5`, its clean-hit rate
beats `keel+embed` with the 95% persona-bootstrap interval of the paired difference
entirely above zero.

If it passes, it becomes the agent's default retriever when the encoder is available. If
not, WordLlama stays the default and the README reports the result.

### Reported, not part of the rule

- Direct and indirect wordings separately.
- `keel+transformer` against every other arm.
- Both encoders at every weight on dev.
- All arms on the round 2 held-out set, labeled as already seen.
- Time to embed, since a transformer is much slower than WordLlama.

### Outcome

Run once, as planned. On dev, `all-MiniLM-L6-v2` at `w = 2.0` scored highest (98.8%, against
98.1% for `bge-small-en-v1.5` at the same weight). On the third held-out set at `k = 5`:

| Arm | Clean hit | Direct | Indirect |
| --- | --- | --- | --- |
| `embedding` (WordLlama) | 35.0% | 54.0% | 16.0% |
| `transformer` (MiniLM) | 44.9% | 61.7% | 28.1% |
| `keel` (round 1) | 64.3% | 98.6% | 30.1% |
| `keel+embed` (round 2 default) | 66.9% | 98.8% | 35.0% |
| `keel+transformer` | 74.9% | 100.0% | 49.8% |

`keel+transformer` − `keel+embed`: **+8.0 points** (95% CI +7.5 to +8.6). **Passed.** As
planned, it is now the agent's default when ONNX Runtime is installed and the model is on
disk or can be downloaded. Full tables: `reports/metrics/round4.md`.

Things to know about this result:

- **The weight is at the edge of the grid.** Dev clean hits were still rising at `w = 2.0`,
  the largest weight tried, so a larger weight might do better. I did not extend the grid
  after seeing this; a future round can.
- **Gains are uneven.** Some indirect wordings jumped (the aunt-or-uncle question went from
  8% to 100%), and two fell (telling the valet which car went from 14% to 0%, and the race
  bib pickup from 22% to 10%). Half of indirect questions still miss.
- **It costs speed.** On one CPU thread MiniLM embedded 255 texts a second, against about
  20,000 for WordLlama. Each text is embedded once and cached, so for one person's memory
  this is fine.

## Round 3 amendment

Made after round 4 and **before any live run**. Round 3 was meant to test the agent as it
ships, and round 4 changed what ships. The arms and pass rule are updated to match, at the
same number of API calls:

- **Arms:** `recent`, `lexical`, `transformer`, `keel+embed`, `keel+transformer`, `full`.
  `embedding` (WordLlama) and round 1 `keel` are dropped; the retrieval benchmark already
  covers them.
- **Pass rule:** `keel+transformer` must beat `transformer` (the same encoder used as plain
  dense retrieval) on answer accuracy, with the 95% paired interval entirely above zero.

The original design is kept above for the record.

## Round 5: a wider weight grid for MiniLM

Written after round 4 was merged and before any round 5 run.

### The question

In round 4, dev clean hits were still rising at `w = 2.0`, the largest weight tried, so the
shipped weight may be too small. Does a larger weight on MiniLM's similarity do better?

### Tuning set

Dev alone can't settle this: at `w = 2.0` it is already at 98.8%, too close to the
ceiling to separate weights. So `w` is chosen by the **mean clean-hit rate over three
question sets that have all been seen already**: dev, the round 2 held-out set and the
round 4 held-out set. None of them is used to judge the result. The two held-out sets
contain indirect wordings, which is where the weight matters. Ties go to the smaller
weight.

### Grid

`w` ∈ {1, 2, 3, 4, 6, 8, 12, 16}. The lexical term is scaled to at most 1, so at 16 the
ranking is close to pure embedding similarity among current memories.

### New held-out questions

A fourth set, `HOLDOUT3` in `keel.evaluation.scenarios`, was written and committed before
any round 5 run: one direct and one indirect wording per detail, none repeating an earlier
wording. It is run once, after `w` is chosen.

### Pass rule

If the tuning set chooses `w = 2.0`, the current weight is confirmed and nothing changes.
Otherwise the tuned weight passes if, on `HOLDOUT3` at `k = 5`, its clean-hit rate beats
the current `w = 2.0`, with the 95% persona-bootstrap interval of the paired difference
entirely above zero. If it passes, it becomes the agent's weight; if not, 2.0 stays.

### Reported, not part of the rule

- Every weight on every tuning set, and on direct and indirect wordings separately.
- The tuned weight against plain MiniLM search and Keel with WordLlama.

The live check (round 3) uses whatever weight the agent ships with, so it would test the
round 5 result if one is adopted. It has still not been run.

### Outcome

Run once, as planned. The tuning set chose `w = 6.0` (mean clean hit 86.3%, against 85.2%
at 2.0). The curve is flat near the top: every weight from 4 to 12 is within 0.2 points.
On the fourth held-out set at `k = 5`:

| Arm | Clean hit | Direct | Indirect |
| --- | --- | --- | --- |
| `transformer` (plain MiniLM search) | 49.4% | 60.1% | 38.6% |
| `keel+embed` (WordLlama) | 67.6% | 92.0% | 43.2% |
| `keel+transformer`, `w = 2.0` (round 4 default) | 84.4% | 100.0% | 68.8% |
| `keel+transformer`, `w = 6.0` | 87.4% | 100.0% | 74.8% |

`w = 6.0` − `w = 2.0`: **+3.0 points** (95% CI +2.6 to +3.4), all of it on indirect wordings
(+6.0). **Passed**, so the agent now uses `w = 6.0`. Full tables:
`reports/metrics/round5.md`.

Two things to keep in mind:

- **This set is easier than round 4's.** The round 4 default scores 84.4% here against
  74.9% on the round 4 set. Absolute rates can't be compared across rounds, only the
  paired differences within one.
- **Not every set agrees.** On the round 4 held-out set, which was part of tuning, larger
  weights were slightly *worse* (74.9% at 2.0, 74.2% at 6.0). The gain comes from the other
  two tuning sets and holds on the fresh one, but it is not universal.

## Round 6: larger transformer encoders

Written after round 5 was merged and before any larger encoder was run on benchmark text.

### The question

With MiniLM at `w = 6.0`, direct questions are at 100% but indirect ones still miss often
(68.8% to 74.8% across the last two held-out sets). MiniLM has 22M parameters. Does an
encoder about five times larger read implied topics better?

### Encoders

Both are on the same public mirror as MiniLM, pinned by SHA-256, and run on CPU with ONNX
Runtime:

| Encoder | Parameters | Download | Pooling |
| --- | --- | --- | --- |
| `all-MiniLM-L6-v2` (current) | 22M | 83 MB | mean |
| `bge-base-en-v1.5` | 110M | 204 MB | CLS |
| `all-mpnet-base-v2` | 110M | 403 MB | mean |

A 560M-parameter multilingual E5 model is also on the mirror, but at 1.3 GB it is too large
to download by default for a personal agent, so it is not tried.

### Choosing the encoder and weight

All three encoders, each at `w` ∈ {2, 4, 6, 8, 12}, are scored on the **mean clean-hit
rate over four question sets that have all been seen**: dev and the round 2, 4 and 5
held-out sets. The best (encoder, weight) pair is chosen. Ties go to the smaller model,
then the smaller weight. MiniLM is in the grid, so "keep MiniLM" is a possible outcome.

### New held-out questions

A fifth set, `HOLDOUT4` in `keel.evaluation.scenarios`, was written and committed before
any larger encoder was run: one direct and one indirect wording per detail, none repeating
an earlier wording. It is run once, after the choice is made.

### Pass rule

If MiniLM at `w = 6.0` is chosen, nothing changes. Otherwise the chosen pair is compared
with it on `HOLDOUT4` at `k = 5`. A larger encoder costs a bigger first download and
slower embedding, so it is **adopted only if the 95% persona-bootstrap interval of the
paired clean-hit difference has a lower bound of at least 1 point**. If the interval is
above zero but below that bar, the gain is reported and MiniLM stays.

### Reported, not part of the rule

- Every encoder and weight on every tuning set.
- Direct and indirect wordings separately.
- Embedding speed for each encoder on one CPU thread.

### Outcome

Run once, as planned. On the four already-seen sets, `all-mpnet-base-v2` at `w = 12.0` had
the highest mean clean-hit rate (86.8%, against 86.6% for MiniLM at 6.0 and 83.8% at best
for `bge-base-en-v1.5`). On the fifth held-out set at `k = 5`:

| Arm | Clean hit | Direct | Indirect |
| --- | --- | --- | --- |
| `transformer` (plain MiniLM search) | 48.9% | 61.4% | 36.3% |
| `larger` (plain mpnet search) | 52.7% | 61.9% | 43.5% |
| `keel+transformer` (MiniLM, `w = 6.0`) | 85.5% | 100.0% | 71.0% |
| `keel+larger` (mpnet, `w = 12.0`) | 89.3% | 100.0% | 78.5% |

`keel+larger` − `keel+transformer`: **+3.8 points** (95% CI +3.4 to +4.2), all of it on
indirect wordings (+7.6). The lower bound clears the 1-point bar, so **mpnet at `w = 12.0`
is adopted** as the agent's default encoder. Full tables: `reports/metrics/round6.md`.

Things to keep in mind:

- **The tuning sets barely told the models apart.** On them, mpnet led MiniLM by 0.2
  points, and weights 6 to 12 for mpnet were within rounding of each other. The larger
  gain appeared only on the fresh set. That is the pattern a real but modest improvement
  would show on near-ceiling tuning data, but it is one held-out set.
- **The chosen weight is again at the top of the grid.** A larger weight was not tried.
- **Bigger was not automatically better.** `bge-base-en-v1.5`, the same size as mpnet,
  scored below MiniLM on every tuning weight.
- **Cost.** mpnet is a 403 MB first download (MiniLM was 83 MB), and on one CPU thread it
  embedded about 50 texts a second against about 340 for MiniLM. Each memory is embedded
  once and cached, so for one person's memory this is a one-off cost of seconds, but it
  would matter at scale.

## Round 7: cross-encoder reranking

Written after round 6 was merged and before any cross-encoder was run on benchmark text.

### The question

Direct questions are at 100%, so what remains is indirect ones: 78.5% clean hit on the
round 6 held-out set. Keel compares a question and a memory through two separate vectors.
A cross-encoder reads them together, which should be better at implied links such as
"Whose local sports team should I cheer for?" and "Moved to Lisbon". It is too slow to run
on every memory, so it only reorders a short list chosen by Keel's existing score.

Before writing this plan, one thing was measured on question sets that have already been
seen: how often the current memory is anywhere in that short list. For indirect wordings,
it is in the 5 that get shown 49% to 79% of the time, but in the top 20 candidates 86% to
93% of the time. So a reranker has room to help. It cannot fix a miss that the short
list leaves out.

### Rerankers

Both are from the ms-marco MiniLM family, trained on web search queries and passages, and
run on CPU with ONNX Runtime. The ONNX exports are fetched as separate files, and each
file's SHA-256 is written into `configs/eval.toml` on first download, before any benchmark
run:

| Reranker | Layers | Download |
| --- | --- | --- |
| `ms-marco-MiniLM-L-6-v2` | 6 | about 90 MB |
| `ms-marco-MiniLM-L-12-v2` | 12 | about 130 MB |

Larger rerankers such as `bge-reranker-base` (about 1.1 GB) are not tried, for the same
download-size reason that E5-large was left out of round 6.

Memories are short first-person notes, not web passages. Whether a search-trained
reranker transfers to them is part of what this round tests.

### The arm

The first stage is the shipped retriever, unchanged: Keel with `all-mpnet-base-v2` at
`w = 12.0`. The always-on constraints are chosen as before. The `n` best other memories by
Keel's hybrid score are then rescored, either as

- the hybrid score plus `w_r` × the cross-encoder's logit, or
- the cross-encoder's logit alone.

Memories outside the short list are never shown.

### Choosing the setting

Every combination of reranker, `n` ∈ {10, 20, 30} and `w_r` ∈ {0.25, 0.5, 1, 2, 4,
logit alone} is scored on the **mean clean-hit rate over five question sets that have all
been seen**: dev and the round 2, 4, 5 and 6 held-out sets. The best combination is
chosen. Ties go to the smaller model, then the shorter list, then blending before logit
alone, then the smaller weight. If no combination beats the current retriever's mean on
those sets, nothing changes and the held-out set is not used.

### New held-out questions

A sixth set, `HOLDOUT5` in `keel.evaluation.scenarios`, was written and committed before
any cross-encoder was run: one direct and one indirect wording per detail, none repeating
an earlier wording, and none written with any reranker's output in view. It is run once,
after the choice is made.

### Pass rule

The chosen setting is compared with the current retriever on `HOLDOUT5` at `k = 5`. A
reranker adds a download and a model call on every question, so it is **adopted only if
the 95% persona-bootstrap interval of the paired clean-hit difference has a lower bound of
at least 1 point**. If the interval is above zero but below that bar, the gain is reported
and nothing changes.

### Reported, not part of the rule

- Every setting on every tuning set.
- Direct and indirect wordings separately.
- Reranking speed for each model on one CPU thread.

### Outcome

Run once, as planned, after both rerankers' revisions and checksums were committed. On the
five already-seen sets, `ms-marco-MiniLM-L-6-v2` on the top 20 at `w_r = 2` had the highest
mean clean-hit rate (88.6%, against 87.3% without reranking and 88.0% at best for the
12-layer model). On the sixth held-out set at `k = 5`:

| Arm | Clean hit | Direct | Indirect |
| --- | --- | --- | --- |
| `keel+larger` (mpnet, `w = 12.0`, no reranking) | 85.8% | 100.0% | 71.6% |
| `keel+rerank` (+ L-6 reranker, top 20, `w_r = 2`) | 88.1% | 99.9% | 76.2% |

`keel+rerank` − `keel+larger`: **+2.3 points** (95% CI +1.9 to +2.6). Indirect wordings
gained 4.6 points (+3.9 to +5.3); direct wordings lost 0.1 (−0.2 to 0.0). The lower bound
clears the 1-point bar, so **the L-6 reranker on the top 20 at `w_r = 2` is adopted** as
the agent's default second stage. Full tables: `reports/metrics/round7.md`.

Things to keep in mind:

- **The logit alone is worse than no reranking.** On the tuning sets, every reranker-only
  setting scored below the current retriever (84.2% to 86.5% against 87.3%), and longer
  lists made it worse. The gain comes from blending.
- **The chosen setting is inside the grid.** Neither the list size nor the weight is at an
  edge; 20 and 30 candidates were within 0.1 points of each other.
- **Direct questions moved slightly the wrong way.** −0.1 points, with an interval
  touching zero: a handful of direct questions that were right are now wrong.
- **Cost.** A 91 MB download, and about 130 ms per question on one CPU thread for 20
  candidates (the 12-layer model takes twice as long). Unlike embeddings, this cost is
  paid on every question, not once per memory.
- **CI reproduction.** The run took 30 minutes on one CPU thread.

## Round 3 amendment 2

Made after round 7 and **before any live run**, for the same reason as the first
amendment: round 7 changed what ships. The number of arms and API calls is unchanged:

- **Arms:** `recent`, `lexical`, `transformer`, `keel+transformer`, `keel+rerank`, `full`.
  `keel+embed` (the WordLlama fallback) is dropped to make room for `keel+rerank`.
- **Pass rule:** `keel+rerank` must beat `transformer` (plain dense retrieval with the
  same encoder) on answer accuracy, with the 95% paired interval entirely above zero.
  `keel+transformer` stays as an arm, so the reranker's own effect on answers is reported.


## Round 8: quantized models

Written after round 7 was merged and before any quantized model was run on benchmark text.

### The question

The shipped retriever downloads about 500 MB (mpnet 436 MB as ONNX, the reranker 91 MB)
and spends about 130 ms per question on reranking. Both models have standard int8
versions about four times smaller. Can they replace the full-precision models without
losing retrieval quality?

This round asks whether the smaller models are **no worse**, not whether they are better.

### Models

The int8 versions are dynamic-quantized ONNX exports from the same publisher as the
round 7 rerankers, pinned by revision and SHA-256 in `configs/eval.toml`:

| Model | Full precision | int8 |
| --- | --- | --- |
| `all-mpnet-base-v2` | 436 MB | 110 MB |
| `ms-marco-MiniLM-L-6-v2` | 91 MB | 23 MB |

Before writing this plan, both int8 models were loaded and run on four neutral sentences
(no benchmark text). Both were deterministic. The reranker's scores were within 0.3 of
full precision. The encoder kept the sentences in the same order of similarity, but its
vectors agreed with full precision only at cosine 0.82 to 0.92, so it may need a
different weight in Keel's score.

### Variants

The three combinations other than the current one: int8 encoder with the full reranker,
full encoder with the int8 reranker, and both int8. The reranker keeps its shipped
setting (top 20, weight 2.0). The full-precision encoder keeps weight 12. The int8
encoder's weight is chosen from {8, 12, 16} on the tuning sets; ties go to 12, then to
the smaller weight.

### Choosing the variant

Every variant is scored on the **mean clean-hit rate over six question sets that have all
been seen**: dev and the round 2, 4, 5, 6 and 7 held-out sets. A variant more than 0.5
points below the current retriever there is not taken further. Of the rest, the one with
the **smallest combined model size** is chosen (ties: higher tuning mean). If none is
within 0.5 points, nothing changes and the held-out set is not used.

### New held-out questions

A seventh set, `HOLDOUT6` in `keel.evaluation.scenarios`, was written and committed before
any quantized model was run on benchmark text. It has one direct and one indirect wording
per detail, and repeats no earlier wording.

### Pass rule

The chosen variant is compared with the current retriever on `HOLDOUT6` at `k = 5`. It is
**adopted if the 95% persona-bootstrap interval of the paired clean-hit difference has a
lower bound of at least −1 point**: a non-inferiority test with a 1-point margin. A
variant that is better also passes.

### Reported, not part of the rule

- Every variant and int8 encoder weight on every tuning set.
- Direct and indirect wordings separately.
- Model sizes, embedding speed and reranking time per question on one CPU thread.

### Outcome

Run once, as planned. On the six already-seen sets the current models scored 88.6%:

| Variant | Best encoder weight | Mean clean hit |
| --- | --- | --- |
| int8 encoder, full reranker | 8 | 86.3% |
| full encoder, int8 reranker | 12 (fixed) | 88.4% |
| both int8 | 8 | 86.1% |

Only the int8 reranker came within 0.5 points, so it was the variant tested. On the
seventh held-out set at `k = 5`:

| Arm | Clean hit | Direct | Indirect |
| --- | --- | --- | --- |
| `keel+rerank` (current models) | 91.5% | 99.6% | 83.5% |
| `keel+rerank(int8)` (int8 reranker) | 91.7% | 99.5% | 83.8% |

`keel+rerank(int8)` − `keel+rerank`: **+0.1 points** (95% CI −0.0 to +0.3). The lower
bound is far above −1 point, so **the int8 reranker is adopted**. The encoder stays at full
precision. Full tables: `reports/metrics/round8.md`.

Things to keep in mind:

- **The int8 encoder lost at every weight.** Re-choosing its weight did not recover the
  loss: 86.3% at best, against 88.6%. The drift seen on neutral sentences before the run
  (cosine 0.82 to 0.92 against full precision) showed up as lost retrieval.
- **The reranker saving is real but modest in absolute terms.** The download falls from
  91 to 23 MB, while the encoder's 436 MB is unchanged. Reranking time per question
  halved on this machine (133 to 65 ms for 20 memories).
- **The held-out set was easier than round 7's.** The current models scored 91.5% here
  against 88.1% on round 7's set. That is a property of the questions, which is why each
  comparison is made within one set.


## Round 9: smaller encoders

Written after round 8 was merged and before any of the smaller encoders was run on
benchmark text in this round.

### The question

After round 8, almost all of Keel's first download is the encoder: `all-mpnet-base-v2`,
436 MB as ONNX. An int8 copy of it lost about 2.3 points. Since round 7, a cross-encoder
reranker now sits on top of the encoder and reorders its top 20. Can a full-precision
encoder a third of the size or less, with that reranker on top, do as well as mpnet?

Like round 8, this asks whether a smaller encoder is **no worse**, not whether it is
better.

### Encoders

All four are full precision, need no query prefix, and are pinned by SHA-256 in
`configs/eval.toml`. MiniLM and bge-small were run in rounds 4 to 6, before the reranker
existed.

| Encoder | Parameters | ONNX size | Pooling |
| --- | --- | --- | --- |
| `all-mpnet-base-v2` (current) | 110M | 436 MB | mean |
| `all-MiniLM-L6-v2` | 22M | 90 MB | mean |
| `bge-small-en-v1.5` | 33M | 133 MB | CLS |
| `all-MiniLM-L12-v2` | 33M | 133 MB | mean |
| `gte-small` | 33M | 133 MB | mean |

Before writing this plan, the two new ones were run on four neutral sentences (no
benchmark text). Both were deterministic. `gte-small` puts unrelated sentences at 0.69
to 0.77 similarity, where the others put them near 0, so it may need a much larger weight
in Keel's score. The weight grid is wide for that reason.

### The arm

Every encoder is paired with the shipped second stage: the int8 reranker on the top 20,
blend weight 2.0. Only the encoder and its weight change.

### Choosing the encoder

Each smaller encoder's weight is chosen from {4, 8, 12, 16, 24, 32, 48} on the **mean
clean-hit rate over seven question sets that have all been seen**: dev and the round 2,
4, 5, 6, 7 and 8 held-out sets. Ties go to the smaller weight. An encoder more than 0.5
points below the current retriever there is not taken further. Of the rest, the one with
the **smallest model** is chosen (ties: higher tuning mean). If none is within 0.5
points, nothing changes and the held-out set is not used.

### New held-out questions

An eighth set, `HOLDOUT7` in `keel.evaluation.scenarios`, was written and committed before
any of the smaller encoders was run in this round. It has one direct and one indirect
wording per detail, and repeats no earlier wording.

### Pass rule

The chosen encoder is compared with the current retriever on `HOLDOUT7` at `k = 5`. It is
**adopted if the 95% persona-bootstrap interval of the paired clean-hit difference has a
lower bound of at least −1 point**.

### Reported, not part of the rule

- Every encoder at every weight on every tuning set.
- Direct and indirect wordings separately.
- Model sizes and embedding speed on one CPU thread.

### Outcome

Run once, as planned. On the seven already-seen sets, mpnet at `w = 12` with the int8
reranker scored 88.9%:

| Encoder | Best weight | Mean clean hit | vs mpnet |
| --- | --- | --- | --- |
| `all-MiniLM-L12-v2` | 4 | 87.9% | −1.0 |
| `all-MiniLM-L6-v2` | 4 | 87.8% | −1.1 |
| `gte-small` | 24 | 87.6% | −1.3 |
| `bge-small-en-v1.5` | 4 | 85.8% | −3.1 |

**No encoder came within 0.5 points, so mpnet is kept** and `HOLDOUT7` was not used; it
stays unseen for a later round. Full tables: `reports/metrics/round9.md`.

Things to keep in mind:

- **The reranker narrows the gap but does not close it.** MiniLM trailed mpnet by 3.8
  points in round 6, without reranking; here, with it, by 1.1.
- **Three encoders peaked at the edge of the grid.** MiniLM-L6, MiniLM-L12 and bge-small
  all did best at `w = 4`, the smallest weight tried, and declined steadily above it. A
  smaller weight might do somewhat better; this round cannot say. `gte-small` peaked in
  the middle of the grid, at 24, as expected from its compressed similarity range.
- **Speed.** The small encoders embedded 130 to 280 texts a second on one CPU thread,
  against about 40 for mpnet. Each memory is embedded once, so for one person this is
  seconds either way.


## Round 10: MiniLM below weight 4

Written after round 9 was merged and before any weight below 4 was run in this round.

### The question

In round 9, both MiniLM encoders did best at `w = 4`, the smallest weight tried, and got
steadily worse above it: MiniLM-L6 scored 87.8% and MiniLM-L12 87.9%, against 88.9% for
mpnet. With a weight below 4, does either come within 0.5 points of mpnet?

This round was prompted by round 9's own tuning results, which come from the same
already-seen sets used here. Picking a weight on them is fine, but the gain they show
will be optimistic. That is why the held-out test below matters.

### Encoders and weights

`all-MiniLM-L6-v2` (90 MB) and `all-MiniLM-L12-v2` (133 MB), as pinned in round 9, each at
`w` ∈ {0.5, 1, 2, 3, 4}. Weight 4 repeats round 9's best as an anchor. Both run under the
shipped int8 reranker on the top 20, weight 2.0, exactly as in round 9.

### Choosing and testing

Exactly as in round 9:

1. Each encoder's weight is chosen on the mean clean-hit rate over the same seven
   already-seen sets. Ties go to the smaller weight.
2. An encoder more than 0.5 points below mpnet there is not taken further.
3. Of the rest, the smaller model is chosen (MiniLM-L6 before MiniLM-L12).

The test set is `HOLDOUT7`. It was committed for round 9, which never reached it, so it
has still not been run by any method.

### Pass rule

**Adopted if the 95% persona-bootstrap interval of the paired clean-hit difference against
mpnet on `HOLDOUT7` has a lower bound of at least −1 point.** If no encoder qualifies on
the tuning sets, `HOLDOUT7` stays unseen.

### Outcome

Run once, as planned. Mean clean hit over the seven already-seen sets (mpnet: 88.9%):

| Encoder | `w = 0.5` | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| `all-MiniLM-L6-v2` | 87.2% | 87.3% | 87.4% | 87.6% | 87.8% |
| `all-MiniLM-L12-v2` | 87.4% | 87.4% | 87.5% | 87.9% | 87.9% |

**Weight 4 was the peak after all.** Both encoders get worse below it, so round 9's
best weights stand, and neither comes within 0.5 points of mpnet. **mpnet is kept** and
`HOLDOUT7` is still unseen. Full tables: `reports/metrics/round10.md`.

The weight-4 anchor came out 0.01 to 0.02 points below round 9's figure (87.815% against
87.828% for MiniLM-L6, 87.938% against 87.949% for MiniLM-L12). The code and models are the
same, but a different weight grid sends texts through the encoder in different batches.
That can change the last bits of a few vectors and flip one or two near-ties among 22,400
questions: the same effect that led CI to compare transformer rounds within 0.1 points.


## Round 11: other encoders of mpnet's size

Written after round 10 was merged and before any of these encoders was run on benchmark
text.

### The question

Rounds 8 to 10 tried to make the encoder smaller and lost at least a point every time.
This round asks the other question: among encoders about the size of `all-mpnet-base-v2`,
is there one that retrieves **better**? The remaining misses are almost all indirect
questions, so an encoder trained for question-to-passage search might help.

### Encoders

All are pinned by revision and SHA-256 in `configs/eval.toml`. Several were trained with
fixed prefixes on questions and on the text being searched. Keel now adds those prefixes
(`query_prefix`, `doc_prefix`); with none, an encoder sees exactly the text it saw before,
so earlier rounds are unaffected.

| Encoder | ONNX size | Pooling | Question prefix | Memory prefix |
| --- | --- | --- | --- | --- |
| `all-mpnet-base-v2` (current) | 436 MB | mean | none | none |
| `gte-base` | 436 MB | mean | none | none |
| `multi-qa-mpnet-base-dot-v1` | 436 MB | CLS | none | none |
| `e5-base-v2` | 436 MB | mean | `query: ` | `passage: ` |
| `nomic-embed-text-v1.5` | 547 MB | mean | `search_query: ` | `search_document: ` |
| `snowflake-arctic-embed-m-v1.5` | 436 MB | CLS | search instruction | none |
| `bge-base-en-v1.5` | 436 MB | CLS | search instruction | none |

Round 6 tried `bge-base-en-v1.5` without its search instruction and found it worse than
MiniLM, so this is a fairer test of it.

Before writing this plan, each was run on one neutral question and three passages (no
benchmark text). All six were deterministic and ranked the right passage first.

### The arm and choosing

Every encoder runs under the shipped second stage: the int8 reranker on the top 20,
weight 2.0. Each encoder's weight is chosen from {4, 8, 12, 16, 24, 32, 48} on the mean
clean-hit rate over the seven already-seen sets (ties: smaller weight). The single best
(encoder, weight) pair is the candidate. If it does not beat mpnet's tuning mean, nothing
changes and `HOLDOUT7` stays unseen.

### Pass rule

The candidate is compared with mpnet on `HOLDOUT7`, which no method has run on yet. These
encoders cost about the same to download and run as mpnet (nomic is 25% larger), so a
gain only has to be real: **the candidate is adopted if the 95% persona-bootstrap interval
of the paired clean-hit difference is entirely above zero.**

### Outcome

Run once, as planned. Mean clean hit over the seven already-seen sets at each encoder's
best weight (mpnet at `w = 12`: 88.9%):

| Encoder | Best `w` | Mean clean hit |
| --- | --- | --- |
| `e5-base-v2` | 48 | 90.4% |
| `multi-qa-mpnet-base-dot-v1` | 12 | 90.1% |
| `nomic-embed-text-v1.5` | 24 | 89.8% |
| `gte-base` | 32 | 88.6% |
| `bge-base-en-v1.5` (search instruction) | 4 | 87.1% |
| `snowflake-arctic-embed-m-v1.5` | 4 | 86.4% |

e5 at `w = 48` was the candidate. On `HOLDOUT7` at `k = 5`:

| Arm | Clean hit | Direct | Indirect |
| --- | --- | --- | --- |
| `keel+rerank` (mpnet, `w = 12`) | 91.1% | 99.9% | 82.3% |
| `keel+other` (e5, `w = 48`) | 92.4% | 99.9% | 84.9% |

`keel+other` − `keel+rerank`: **+1.3 points** (95% CI +0.7 to +1.8), all on indirect
wordings (+2.6, +1.4 to +3.7). The interval is above zero, so **e5-base-v2 at `w = 48` is
adopted** as the agent's encoder. Full tables: `reports/metrics/round11.md`.

Things to keep in mind:

- **The chosen weight is at the top of the grid.** e5 scored 90.4% at both 32 and 48 (48
  slightly higher before rounding). A larger weight was not tried.
- **Three of six beat mpnet on tuning.** e5, multi-qa-mpnet and nomic all did; the search
  instruction did not rescue bge-base, and arctic-embed was the worst of the six.
- **The gain is modest but consistent with the tuning sets**: 1.5 points there, 1.3 on the
  fresh set.
- **No cost change.** e5 is the same size and speed as mpnet (nomic would have been 25%
  larger and half as fast).


## Round 12: larger encoders

Written after round 11 was merged and before any of these encoders was run on benchmark
text.

### The question

Round 11 adopted `e5-base-v2`, and the misses left are still mostly indirect questions.
This round asks whether a **larger** encoder, of about three times the size, retrieves
enough better to be worth its cost.

### Encoders

All are pinned by revision and SHA-256 in `configs/eval.toml`, and each gets the prefixes
it was trained with.

| Encoder | ONNX size | Pooling | Question prefix | Memory prefix |
| --- | --- | --- | --- | --- |
| `e5-base-v2` (current) | 436 MB | mean | `query: ` | `passage: ` |
| `e5-large-v2` | 1,337 MB | mean | `query: ` | `passage: ` |
| `bge-large-en-v1.5` | 1,337 MB | CLS | search instruction | none |
| `mxbai-embed-large-v1` | 1,337 MB | CLS | search instruction | none |

Before writing this plan, each was run on one neutral question and three passages (no
benchmark text). All three were deterministic and ranked the right passage first. On one
CPU thread they embedded about 9 texts a second against e5-base's 33.

### The arm and choosing

Every encoder runs under the shipped second stage: the int8 reranker on the top 20,
weight 2.0. Each encoder's weight is chosen from {4, 8, 12, 16, 24, 32, 48, 64, 96} on the
mean clean-hit rate over the eight already-seen sets, which now include `HOLDOUT7` (ties:
smaller weight). The grid goes past 48 because e5-base's chosen weight was at the top of
round 11's grid. The single best (encoder, weight) pair is the candidate. If it does not
beat e5-base's tuning mean at its shipped `w = 48`, nothing changes and `HOLDOUT8` stays
unseen.

### New held-out questions

`HOLDOUT8` in `src/keel/evaluation/scenarios.py`: one direct and one indirect wording per
detail, written for this round and checked against every earlier wording for duplicates.

### Pass rule

The candidate is compared with e5-base at `w = 48` on `HOLDOUT8`. A larger encoder triples
the first download and runs at about a quarter of the speed, so, as in rounds 6 and 7,
**the candidate is adopted only if the 95% persona-bootstrap interval of the paired
clean-hit difference has a lower bound of at least 1 point**. A gain above zero but below
that bar is reported and not adopted.

### Outcome

Mean clean hit over the eight already-seen sets (e5-base at its shipped `w = 48`: 90.6%):

| Encoder | Best `w` | Mean clean hit |
| --- | --- | --- |
| `e5-large-v2` | 64 | 91.2% |
| `mxbai-embed-large-v1` | 8 | 87.9% |
| `bge-large-en-v1.5` (search instruction) | 4 | 86.9% |

e5-large at `w = 64` was the candidate. On `HOLDOUT8` at `k = 5`:

| Arm | Clean hit | Direct | Indirect |
| --- | --- | --- | --- |
| `keel+e5` (e5-base, `w = 48`) | 90.1% | 100.0% | 80.2% |
| `keel+larger` (e5-large, `w = 64`) | 93.9% | 100.0% | 87.8% |

`keel+larger` − `keel+e5`: **+3.8 points** (95% CI +3.4 to +4.2), all on indirect
wordings (+7.6, +6.7 to +8.4). The lower bound clears the 1-point bar, so **e5-large-v2 at
`w = 64` is adopted** as the agent's encoder. Full tables: `reports/metrics/round12.md`.

### Amendment after the first run: ties between weights

**This rule was added after the round had been run once, and after `HOLDOUT8` had been
used.** The first run, as planned, chose `w = 96` for e5-large: 91.1777% on the tuning
sets against 91.1719% at `w = 64`, a gap of about two questions out of 25,600. On the CI
runner's processor the order reversed and it chose 64, so the committed decision did not
reproduce. ONNX Runtime's last bits differ between processors, and a gap that small is
inside that noise.

The weight choice now treats tuning means within **0.05 points** of the best as tied, and a
tie goes to the smaller weight, extending the planned "ties: smaller weight" rule. For
e5-large that gives 64 on any processor (48 is 0.09 points below the best, outside the
margin); for bge-large it moves the choice from 12 to 4, which changes nothing else. Round
11 is not affected. The comparison above is from the rerun at `w = 64`. In the first run,
at `w = 96`, e5-large scored 93.7% on `HOLDOUT8` (+3.5 points, 95% CI +3.1 to +4.0), so
the decision to adopt e5-large is the same either way; only the weight changed.

Things to keep in mind:

- **`HOLDOUT8` has now been used twice**, once for each weight. Both runs adopt e5-large,
  and the two weights differ by 0.2 points there, so the second look did not decide
  anything, but the 93.9% is not a first-look number.
- **The weight curve is flat at the top.** e5-large scored 91.1% at 48 and 91.2% at both
  64 and 96. A larger weight would probably not change much.
- **The gain on the fresh set is much larger than on tuning**: 0.6 points on the
  already-seen sets, 3.8 on `HOLDOUT8`. e5-base did worse on `HOLDOUT8`'s indirect
  wordings (80.2%) than on `HOLDOUT7`'s (84.9%), so this set happens to be harder for it.
  The direction is the same on both; the size depends on the questions.
- **Only the e5 family helped.** mxbai and bge-large, both strong on public benchmarks,
  were 2.7 and 3.7 points below e5-base here and got worse as their weight rose.
- **The cost is real.** The first download grows from 440 MB to 1.3 GB, and embedding runs
  at about a quarter of the speed. Each memory is embedded once, so for one person's memory
  this is seconds, not minutes.

## Round 13: other encoders of e5-large's size

Written after round 12 was merged and before any of these encoders was run on benchmark
text.

### The question

Round 12 adopted `e5-large-v2`. Round 11 found that the right encoder of a given size mattered
more than the size alone, so this round asks whether another encoder of e5-large's size
retrieves **better** at the same cost.

### Encoders

All are pinned by revision and SHA-256 in `configs/eval.toml`, and each gets the prefixes it
was trained with.

| Encoder | ONNX size | Pooling | Question prefix | Memory prefix |
| --- | --- | --- | --- | --- |
| `e5-large-v2` (current) | 1,337 MB | mean | `query: ` | `passage: ` |
| `e5-large` (the first version) | 1,337 MB | mean | `query: ` | `passage: ` |
| `snowflake-arctic-embed-l` | 1,337 MB | CLS | search instruction | none |
| `gte-large` | 1,337 MB | mean | none | none |

Before writing this plan, each was run on one neutral question and three passages (no
benchmark text). All three were deterministic and ranked the right passage first, at about
the same speed as e5-large-v2.

Encoders larger again (about 2.2 GB each: `multilingual-e5-large`, `bge-m3`,
`snowflake-arctic-embed-l-v2.0`) are left out, to keep this a same-cost comparison and to
limit what the CI job that reruns every round has to download and store.

### The arm and choosing

Every encoder runs under the shipped second stage: the int8 reranker on the top 20, weight
2.0. Each encoder's weight is chosen from {4, 8, 12, 16, 24, 32, 48, 64, 96} on the mean
clean-hit rate over the nine already-seen sets, which now include `HOLDOUT8`. Weights whose
means are within 0.05 points of the best count as tied, and a tie goes to the smaller
weight (round 12's amended rule, set here in advance). The single best (encoder, weight)
pair is the candidate. If it does not beat e5-large-v2's tuning mean at its shipped
`w = 64`, nothing changes and `HOLDOUT9` stays unseen.

### New held-out questions

`HOLDOUT9` in `src/keel/evaluation/scenarios.py`: one direct and one indirect wording per
detail, written for this round and checked against every earlier wording for duplicates.

### Pass rule

The candidate is compared with e5-large-v2 at `w = 64` on `HOLDOUT9`. These encoders cost
the same to download and run, so, as in round 11, **the candidate is adopted if the 95%
persona-bootstrap interval of the paired clean-hit difference is entirely above zero.**

### Outcome

Run once, as planned. Mean clean hit over the nine already-seen sets (e5-large-v2 at its
shipped `w = 64`: 91.5%):

| Encoder | Best `w` | Mean clean hit |
| --- | --- | --- |
| `gte-large` | 32 | 89.5% |
| `snowflake-arctic-embed-l` | 24 | 88.5% |
| `e5-large` (first version) | 4 | 86.8% |

None beat e5-large-v2 on the tuning sets, so **e5-large-v2 stays** and `HOLDOUT9` was not
used; it remains unseen for a later round. Full tables: `reports/metrics/round13.md`.

Things to keep in mind:

- **The gap is not close.** The best of the three was 2.0 points behind, a much larger gap
  than processor-level differences could close.
- **The second version of e5 matters.** The first e5-large, with the same size and prefixes,
  was the weakest of the three, 4.7 points behind e5-large-v2.
- **Two chosen weights sit near the tie margin.** For arctic-embed-l, weight 32 was 0.045
  points behind 24, and for e5-large, weight 12 was 0.04 behind 4. A processor that moved
  those by a few questions could change which weight is recorded, though not the outcome.

## Round 14: encoders of about 2.2 GB

Written after round 13 was merged and before any of these encoders was run on benchmark
text.

### The question

Round 13 found no other encoder of e5-large-v2's size that did better. The step up from
there is encoders built on the 24-layer XLM-RoBERTa: about 2.2 GB each, because of a much
larger vocabulary, but the same depth and width as e5-large-v2 and so about the same speed.
This round asks whether one of them retrieves enough better to be worth the larger download.
Two are from the e5 family, which won rounds 11 and 12.

### Encoders

All are pinned by revision and SHA-256 in `configs/eval.toml`, and each gets the prefixes it
was trained with. Models this size keep their weights in a separate `model.onnx_data` file,
which is pinned too.

| Encoder | ONNX size | Pooling | Question prefix | Memory prefix |
| --- | --- | --- | --- | --- |
| `e5-large-v2` (current) | 1,337 MB | mean | `query: ` | `passage: ` |
| `multilingual-e5-large` | 2,236 MB | mean | `query: ` | `passage: ` |
| `multilingual-e5-large-instruct` | 2,236 MB | mean | web-search instruction | none |
| `snowflake-arctic-embed-l-v2.0` | 2,268 MB | CLS | `query: ` | none |

The instruct model's question prefix is the retrieval instruction from its model card:
`Instruct: Given a web search query, retrieve relevant passages that answer the query`,
then a new line and `Query: `.

Before writing this plan, each was run on one neutral question and three passages (no
benchmark text). All three were deterministic and ranked the right passage first, at about
the same speed as e5-large-v2 (10.6 to 11.4 texts a second against 11.8 on one CPU thread).
`bge-m3`, the same size, is left out to limit what CI has to download and store.

### The arm and choosing

As in round 13: the shipped int8 reranker on the top 20 at weight 2.0; each encoder's weight
chosen from {4, 8, 12, 16, 24, 32, 48, 64, 96} on the mean clean-hit rate over the nine
already-seen sets, with weights within 0.05 points tied and the smaller winning. The single
best (encoder, weight) pair is the candidate. If it does not beat e5-large-v2's tuning mean
at its shipped `w = 64`, nothing changes and `HOLDOUT9` stays unseen.

### Held-out questions

`HOLDOUT9`, written for round 13. No round 13 encoder qualified to run on it, so no method
has seen it yet.

### Pass rule

The candidate is compared with e5-large-v2 at `w = 64` on `HOLDOUT9`. It runs at about the
same speed but is a 70% larger download, so, as in round 12, **the candidate is adopted
only if the 95% persona-bootstrap interval of the paired clean-hit difference has a lower
bound of at least 1 point**. A gain above zero but below that bar is reported and not
adopted.

### Outcome

Run once, as planned. Mean clean hit over the nine already-seen sets (e5-large-v2 at its
shipped `w = 64`: 91.5%):

| Encoder | Best `w` | Mean clean hit |
| --- | --- | --- |
| `multilingual-e5-large` | 64 | 91.3% |
| `multilingual-e5-large-instruct` | 64 | 90.3% |
| `snowflake-arctic-embed-l-v2.0` | 4 | 86.6% |

None beat e5-large-v2 on the tuning sets, so **e5-large-v2 stays** and `HOLDOUT9` is still
unseen. Full tables: `reports/metrics/round14.md`.

Things to keep in mind:

- **The closest was 0.26 points behind.** multilingual-e5-large came nearest, but still
  about 65 questions short over 25,600, well beyond processor-level differences.
- **The search instruction did not help e5.** The instruct version scored a point below the
  plain multilingual model, and its long question prefix made it embed a third slower.
- **arctic-embed-l-v2.0 did best at the lowest weight tried and worse at every step up**,
  like arctic-embed-l in round 13 and bge and mxbai in round 12.
- **One chosen weight sits near the tie margin.** For the instruct model, weight 64 was
  0.045 points behind 96 and weight 48 was 0.061 behind. A processor that moved those by a
  question or two could change which weight is recorded, though not the outcome.

## Note on model downloads (October 2026)

Rounds 4 to 11 first downloaded four encoders (`all-MiniLM-L6-v2`, `bge-small-en-v1.5`,
`bge-base-en-v1.5` and `all-mpnet-base-v2`) as archives from the `qdrant-fastembed` storage
bucket. That bucket stopped serving public downloads in October 2026. The same models are
now pinned, by revision and SHA-256, to ONNX exports on Hugging Face: `all-MiniLM-L6-v2`
from Qdrant's own upload, and the other three from Xenova's full-precision exports.

They are not byte-for-byte the files the rounds were first run with, so rounds 4 to 11 were
rerun with them before the switch and compared with the committed results by
`scripts/check_metrics.py`, the same check CI applies. Every decision matched exactly and
every number was within the check's tolerance. Nothing in rounds 4 to 11 was rerun for its
result, and no committed number or decision changed.

## Round 15: newer encoder architectures

Written after round 14 was merged and before any of these encoders was run on benchmark
text.

### The question

Every encoder tried so far is built on the original BERT or XLM-RoBERTa designs. This round
asks whether encoders built on newer designs (ModernBERT, and Alibaba's gte-v1.5 model with
rotary position embeddings) retrieve better than e5-large-v2 at a similar cost. Two are base
size: if either matched e5-large-v2, it would also be three times faster.

### Encoders

All are pinned by revision and SHA-256 in `configs/eval.toml`, and each gets the prefixes and
pooling its model card gives.

| Encoder | Design | ONNX size | Pooling | Question prefix | Memory prefix |
| --- | --- | --- | --- | --- | --- |
| `e5-large-v2` (current) | BERT | 1,337 MB | mean | `query: ` | `passage: ` |
| `gte-modernbert-base` | ModernBERT | 596 MB | CLS | none | none |
| `modernbert-embed-base` | ModernBERT | 597 MB | mean | `search_query: ` | `search_document: ` |
| `modernbert-embed-large` | ModernBERT | 1,580 MB | mean | `search_query: ` | `search_document: ` |
| `gte-large-en-v1.5` | gte-v1.5 | 1,746 MB | CLS | none | none |

Before writing this plan, each was run on one neutral question and three passages (no
benchmark text). All four were deterministic and ranked the right passage first. On one CPU
thread the two base models embedded about 35 texts a second and the two large ones about
10, against 12 for e5-large-v2.

### The arm and choosing

As in rounds 13 and 14: the shipped int8 reranker on the top 20 at weight 2.0; each encoder's
weight chosen from {4, 8, 12, 16, 24, 32, 48, 64, 96} on the mean clean-hit rate over the
nine already-seen sets, with weights within 0.05 points tied and the smaller winning. The
single best (encoder, weight) pair is the candidate. If it does not beat e5-large-v2's
tuning mean at its shipped `w = 64`, nothing changes and `HOLDOUT9` stays unseen.

### Held-out questions

`HOLDOUT9`, written for round 13. Neither round 13 nor round 14 qualified an encoder to run
on it, so no method has seen it yet.

### Pass rule

The candidate is compared with e5-large-v2 at `w = 64` on `HOLDOUT9`. The two large
encoders are 18% and 30% larger downloads and run at about the same speed; the base ones are
smaller and faster. As with nomic in round 11 (25% larger), that is close enough in cost
that a gain only has to be real: **the candidate is adopted if the 95% persona-bootstrap
interval of the paired clean-hit difference is entirely above zero.**

### Outcome

Run once, as planned. Mean clean hit over the nine already-seen sets (e5-large-v2 at its
shipped `w = 64`: 91.5%):

| Encoder | Best `w` | Mean clean hit |
| --- | --- | --- |
| `gte-modernbert-base` | 24 | 90.2% |
| `modernbert-embed-base` | 8 | 88.9% |
| `modernbert-embed-large` | 8 | 88.4% |
| `gte-large-en-v1.5` | 12 | 88.3% |

None beat e5-large-v2 on the tuning sets, so **e5-large-v2 stays** and `HOLDOUT9` is still
unseen. Full tables: `reports/metrics/round15.md`.

Things to keep in mind:

- **The newer designs did not help here.** The best, gte-modernbert-base, was 1.3 points
  behind; the two large ones were the weakest of the four.
- **The best base model is a speed option, not a replacement.** gte-modernbert-base embeds
  about two and a half times as fast as e5-large-v2 at under half the size, but costs 1.3
  points on these sets. It sits close to e5-base-v2 (round 11), not above it.
- **The chosen weights are clear of the tie margin.** The nearest runner-up outside it is
  gte-large-en-v1.5's weight 16, 0.076 points behind 12.

## Round 16: an int8 e5-large-v2

Written after round 15 was merged and before the int8 model was run on benchmark text.

### The question

e5-large-v2 is a 1.3 GB first download and the slowest encoder Keel has shipped. Round 8
found that an int8 copy of the reranker lost nothing, while an int8 mpnet lost at every
weight. This round asks the same question of the current encoder: does an int8 copy of
e5-large-v2 retrieve about as well, at a quarter of the download?

### Models

Both come from the same pinned revision of `Xenova/e5-large-v2` and share its tokenizer,
prefixes (`query: `, `passage: `) and mean pooling. Both are pinned by SHA-256 in
`configs/eval.toml`.

| Encoder | ONNX size |
| --- | --- |
| `e5-large-v2` (current) | 1,337 MB |
| `e5-large-v2 (int8)` | 337 MB |

Before writing this plan, both were run on one neutral question and three passages (no
benchmark text). Both were deterministic and ranked the right passage first, with cosine
scores within 0.01 of each other. On one CPU thread the int8 model embedded about 76 texts
a second, against 15 for the full-precision one.

### The arm and choosing

As in rounds 13 to 15: the shipped int8 reranker on the top 20 at weight 2.0. The int8
model's weight is re-chosen from {4, 8, 12, 16, 24, 32, 48, 64, 96} on the mean clean-hit
rate over the nine already-seen sets, with weights within 0.05 points tied and the smaller
winning. As in round 8, it goes on to the held-out set only if its tuning mean is no more
than 0.5 points below e5-large-v2's at the shipped `w = 64`. Otherwise nothing changes and
`HOLDOUT9` stays unseen.

### Held-out questions

`HOLDOUT9`, written for round 13. No round has qualified a candidate to run on it, so no
method has seen it yet.

### Pass rule

The int8 model is compared with e5-large-v2 at `w = 64` on `HOLDOUT9`. It is a quarter of
the download and about five times as fast, so, as with the int8 reranker in round 8, it
does not have to be better: **it is adopted if the 95% persona-bootstrap interval of the
paired clean-hit difference has a lower bound of −1 point or more.**

### Outcome

Run once, as planned. Mean clean hit over the nine already-seen sets: e5-large-v2 at its
shipped `w = 64`, 91.5%; the int8 copy at its best weight, `w = 64`, 91.3%. That is 0.2
points lower, inside the 0.5-point gate, so it went on to `HOLDOUT9`.

`keel+e5large(int8)` − `keel+e5large`: **−0.4 points** (95% CI −0.7 to −0.0); direct
wordings −0.2 (−0.6 to +0.1), indirect −0.5 (−1.1 to +0.1). The lower bound is above −1
point, so **the int8 e5-large-v2 at `w = 64` is adopted** as the agent's encoder. Full
tables: `reports/metrics/round16.md`.

Things to keep in mind:

- **It is a small, real loss.** The interval sits just below zero, so the int8 copy is
  slightly worse, not equal. The trade is a quarter of the download (337 MB against
  1,337 MB) and five to six times the embedding speed.
- **Indirect wordings lose the most.** Their lower bound, −1.1 points, is just past the
  margin; the rule is on all wordings together, as planned.
- **The weight sits inside the tie margin.** `w = 96` scored 0.041 points above `w = 64`
  on tuning, less than the 0.05-point margin, so the smaller weight won under the tie rule
  and the shipped weight is unchanged.
- **Every held-out set written so far has now been used.** A further round needs a new one.

## A new held-out set: `HOLDOUT10`

Written after round 16 was merged, when every earlier held-out set had been used, and before
any round was planned around it. Like the others, it has one direct and one indirect wording
for each of the 16 details, in `src/keel/evaluation/scenarios.py`.

- **No method has been run on it.** Adding it builds the questions but scores nothing; rounds
  1 and 2, which must reproduce exactly, came out unchanged.
- **The wordings are new.** None repeats a wording from the dev, test or earlier held-out
  sets (a test checks this for every set). The closest any of them comes to an earlier
  wording is 56% of words in common, against 71% to 100% for each earlier set against the
  ones before it, so it is the least familiar set so far.
- **It is reserved for the next round's final comparison.** As before, a round chooses its
  candidate on already-seen sets and uses `HOLDOUT10` once.

## Round 17: the reranker's setting for the current encoder

Written after `HOLDOUT10` was added and before any run of this round.

### The question

Round 7 chose how many candidates the reranker reorders (20) and how much its logit counts
against the hybrid score (weight 2.0) when the encoder was mpnet at weight 12. The encoder
is now an int8 e5-large-v2 at weight 64, so the hybrid score the logit is added to sits on
a different scale. Is the round 7 setting still the best one?

### The arm

The shipped first stage (int8 e5-large-v2 at `w = 64`) and the shipped int8
`ms-marco-MiniLM-L-6-v2`, with every combination of:

| Candidates | Weights |
| --- | --- |
| 10, 20, 30, 40 | 0.5, 1, 2, 4, 8, 16 |

### Choosing

Each of the 24 settings is scored by mean clean hit over the ten already-seen sets (`dev`
and `HOLDOUT` to `HOLDOUT9`). Settings within 0.05 points of the best tie. If the current
setting (20 at 2.0) is among them, nothing changes and `HOLDOUT10` stays unseen. Otherwise
the cheapest tied setting is the candidate: the shortest list, then the smallest weight.

### Held-out questions

`HOLDOUT10`, written after round 16. No method has been run on it.

### Pass rule

The candidate is compared with the current setting on `HOLDOUT10`.

- **No more candidates than now** costs the same or less per question: **adopted if the
  95% persona-bootstrap interval of the paired clean-hit difference is entirely above
  zero.**
- **More candidates** makes every question slower (the reranker scores each one): **adopted
  only if the lower bound is at least 1 point**, the bar used for costlier changes since
  round 6.
