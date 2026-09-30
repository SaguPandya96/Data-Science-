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
