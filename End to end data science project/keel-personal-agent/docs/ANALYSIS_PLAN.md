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
