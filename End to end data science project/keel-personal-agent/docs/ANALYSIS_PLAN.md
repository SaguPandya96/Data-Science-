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
