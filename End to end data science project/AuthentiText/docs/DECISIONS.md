# Technical decisions

## 2026-08-08 — Separate development, external, and robustness datasets

**Context:** The first research cycle needs enough generator and domain variety
to study generalization, an external corpus relevant to harmful human false
positives, and realistic robustness conditions. The workstation has no CUDA
tooling and cannot store RAID's complete 16.7 GB release on C:.

**Options considered:** Pool all candidate datasets and randomly split them;
develop only on HC3; train on the complete RAID release; or assign distinct
datasets to distinct scientific roles.

**Decision:** Use English MAGE as the development corpus, excluding its
WritingPrompts source from all fitting and threshold work. Seal the complete
Ghostbuster main corpus for one final external evaluation. Treat RAID as a
conditional, separately reported robustness audit after a bounded source-group
sampling design is validated. Defer multilingual M4, SemEval, and DetectRL-X
work to a later research cycle.

**Reason:** MAGE offers useful domain and generator diversity at a manageable
download size. Ghostbuster directly tests student-essay false positives and has
an explicit data license. Keeping it sealed avoids external-test tuning.
Excluding MAGE WritingPrompts reduces known upstream-source overlap with
Ghostbuster. RAID covers attacks and decoding variation but is too large for an
unbounded local workflow.

**Tradeoffs:** The first cycle supports English only. Ghostbuster generators are
not current, some source families may still overlap semantically, and MAGE's
published random splits may need replacement after leakage analysis. RAID and
multilingual claims remain unavailable until their own acquisition and
evaluation protocols are completed.

## 2026-08-08 — Preserve text and canonicalize MAGE metadata

**Context:** The pinned MAGE files use raw label `0` for machine and `1` for
human, have no stable row identifier, include the WritingPrompts domain shared
with the sealed external corpus, and contain exact text shared between
validation and test.

**Options considered:** Mutate text with broad normalization; retain all source
families; use CSV row numbers as IDs; or create a minimally transformed,
source-aware canonical layer.

**Decision:** Preserve decoded text exactly. Exclude every source beginning
with `wp_` and record the reason as shared upstream domain. Map raw MAGE `0` to
canonical target `1` (machine positive) and raw `1` to target `0`. Derive a
content ID from exact UTF-8 text and a separate record ID from the pinned
dataset, revision, partition, source, label, and content ID. Write deterministic
gzip JSON Lines and keep them outside Git.

**Reason:** This produces model-ready labels and stable grouping keys without
destroying stylistic evidence or obscuring loss accounting. Content IDs expose
cross-partition equality while record IDs retain partition-specific identity.

**Tradeoffs:** Exact content hashes do not identify Unicode-normalized or near
duplicates. The source exclusion removes 43,609 rows and changes class balance.
Processed data duplicates text bytes on local storage, although deterministic
gzip reduces the footprint to 204,866,560 bytes.

## 2026-08-08 — Sanitize published partitions without moving records

**Context:** The canonical MAGE development layer contains 141 duplicated
normalized values, including 65 cross-partition groups, one conflicting-target
group, and 12 additional high-overlap edges confirmed by the bounded audit.

**Options considered:** Keep the published splits unchanged; move complete
overlap groups to newly assigned partitions; drop every member of all overlap
groups; or keep one representative under an explicit precedence rule.

**Decision:** Connect normalized-equal records and confirmed high-overlap edges.
Drop every member of a conflicting-target component. For every other component,
retain the record in the highest-precedence existing partition—train, then
test, then validation—and use the lowest record ID for ties. Never move a
record between published partitions.

**Reason:** Training precedence preserves the fitting population while removing
its known copies from evaluation. Test precedence over validation retains final
internal evaluation cases where the two holdouts overlap. Removing both sides
of a label conflict avoids arbitrarily choosing a target. Not moving records
keeps provenance and upstream partition roles intact.

**Tradeoffs:** The transformation removes 154 of 389,073 records, including two
conflicting-target records. The 12 lexical edges come from a bounded sample, so
the resulting in-distribution split is sanitized against known overlap rather
than certified free of every semantic or template relationship. Source-group
holdouts remain necessary.

## 2026-08-08 — Define holdouts from exact upstream source identifiers

**Context:** All 288 sanitized MAGE source values encode a domain and writing
kind. Machine source values additionally encode a generation strategy and an
exact generator identifier. The acquired schema has no separate stable prompt
or model-family field.

**Options considered:** Invent architecture families; randomly sample sources;
materialize a complete text copy for every fold; or define selectors directly
from the observed identifiers.

**Decision:** Define nine leave-one-domain-out regimes and 27
leave-one-exact-generator-out regimes. Domain folds remove the held domain from
published train and validation and test only on its published test rows.
Generator folds remove held-generator machine rows from train and validation,
then test on that generator's published-test machine rows plus every
published-test human row. Preserve generator strings exactly, including
upstream spelling. Store selectors and measured counts, not copied text.

**Reason:** These designs test an unseen domain or exact generator without
asserting an unsupported model-family taxonomy. Keeping all test humans in a
generator fold preserves the full human false-positive audit. Selector-only
regimes save substantial local storage and keep records linked to the one
sanitized source of truth.

**Tradeoffs:** Generator-fold tests are deliberately imbalanced: 25,634 human
records versus 610 to 1,964 machine records. Evaluation must therefore report
class-aware ranking and error metrics rather than accuracy alone. Exact
generator holdout does not imply holdout of related architectures, and absence
of prompt identifiers prevents prompt-group guarantees.

## 2026-08-08 — Start with bounded, interpretable CPU baselines

**Context:** Development EDA shows a class-correlated length tail, while the
workstation has 15.82 GiB RAM, no CUDA tooling, and limited local disk. The
first model run must quantify simple shortcuts before adding complexity.

**Options considered:** Train only a lexical model; start with a transformer;
use unrestricted sparse features; or fit a small diagnostic ladder.

**Decision:** Train three fixed baselines on sanitized train only: the training
majority/prevalence, balanced logistic regression over log character and
whitespace-token lengths, and balanced logistic regression over lowercased
word unigram/bigram TF-IDF. Bound the vocabulary at 100,000 features, retain
terms occurring in at least five documents, use float32 TF-IDF, and fix the
stochastic seed at 1729. Do not use record or source metadata as features.

**Reason:** Majority establishes the class-imbalance floor. The length-only
diagnostic measures the EDA artifact directly. Sparse lexical logistic
regression is a strong, inspectable CPU baseline whose cost can be measured
before any encoder experiment.

**Tradeoffs:** Word features can encode dataset- and domain-specific vocabulary
and do not capture semantics reliably. The vocabulary cap discards some terms.
The length model is explicitly diagnostic, not a product candidate. Validation
and test behavior remain unknown until separate evaluation commits.

## 2026-08-08 — Prefer abstention over a forced binary threshold

**Context:** Word TF-IDF has useful validation ranking but a 26.6089% human
false-positive rate at threshold 0.5 and large domain variation. Raw scores are
not calibrated authorship probabilities.

**Options considered:** Keep threshold 0.5; choose one binary threshold; fit a
calibrator on all validation data; or reserve distinct validation roles for
calibration, method selection, threshold selection, and internal audit.

**Decision:** Hash-assign validation records to 40% calibration fit, 30%
method/policy selection, and 30% calibration audit. Fit sigmoid and isotonic
calibrators and compare them with the raw score using selection Brier score,
ECE, and deterministic tie-breaking. Use the selected isotonic map. Choose a
likely-machine threshold targeting at most 5% human false-machine decisions and
a likely-human threshold targeting at most 5% machine false-human decisions on
policy selection. Label the middle interval uncertain.

**Reason:** The three-way design retains an internal audit not used to fit or
select the calibrator or thresholds. Separate error constraints center the
costly human false-positive risk while preventing a superficially safe policy
that labels machine text human. Abstention communicates what the validation
evidence actually supports.

**Tradeoffs:** The audit decisive coverage is only 40.0464%; 59.9536% is
uncertain. Aggregate audit error targets are not domain guarantees: human
false-machine rates reach 12.2632% on HellaSwag, while machine false-human rates
reach 20.2484% on Yelp. Validation-derived thresholds still require untouched
test evaluation and cannot prove correctness for individual text.

## 2026-08-08 — Freeze the policy after the first published-test run

**Context:** The calibrated policy was fully specified before published-test
scoring. On test, its human false-machine rate is 5.2391% and machine
false-human rate is 5.9880%, both above their 5% validation-selection point
targets.

**Options considered:** Adjust thresholds to hit 5% on test; conceal the misses;
or preserve the frozen policy and report the deviations with uncertainty.

**Decision:** Do not change the model, calibration map, or thresholds in
response to the test. Version text-free test predictions and the recomputable
report as the immutable first-cycle in-distribution result. Any later model or
policy is a new candidate requiring a new, separately identified evaluation
cycle—not a revision of this test.

**Reason:** Test-driven threshold changes would turn the held-out partition into
development data and bias the claimed result. The observed misses are precisely
the uncertainty that a held-out check is intended to reveal.

**Tradeoffs:** The retained policy does not meet either 5% point target on test
and abstains on 57.0708% of cases. This limits usefulness but preserves the
scientific meaning of the evaluation. The test is still MAGE in-distribution,
not an external generalization benchmark.

## 2026-08-09 — Retain the lexical model as a research baseline and defer a transformer

**Context:** The lexical baseline is fully calibrated and evaluated on
in-distribution and MAGE development OOD data. It has useful in-distribution
ranking and low batch cost, but misses both decisive-error targets, has severe
domain and short-text failures, and degrades to ROC AUC 0.697370 with 12.3360%
human false-machine decisions on the deduplicated OOD stress set. The workstation
has no CUDA tooling, and the locked environment contains no deep-learning stack
or pretrained encoder artifacts.

**Options considered:** Promote the lexical model as production-ready; replace
it with a control; install and train a transformer without a resource pilot;
run a small non-comparable transformer subsample; or retain the measured model
as a constrained research baseline while defining gates for a future candidate.

**Decision:** Retain word TF-IDF logistic regression, isotonic calibration, and
the frozen abstention policy as version 1 of the local research tool. Reject the
majority and length controls. Do not approve the baseline for high-stakes or
production use. Defer transformer modeling at this checkpoint and make no claim
about unmeasured transformer performance.

**Reason:** The selected model is the only implemented candidate materially
above chance, is small and reproducible on the audited CPU, and already supports
the verified inference contract. Its measured failures require the existing
warnings and uncertain state. A CPU-only transformer experiment over a
substantially different sample would not be a fair replacement test, while a
full experiment lacks a measured feasibility case and would expand dependency,
weight-provenance, and artifact requirements.

**Tradeoffs:** The product remains a limited lexical research baseline with poor
OOD generalization and no semantic representation. Deferral delays evaluation
of a potentially stronger architecture. A future candidate must begin a new,
predefined train/calibrate/evaluate cycle and demonstrate meaningful safety and
generalization gains before replacement.

## 2026-08-09 — Require manual evidence-gated model replacement

**Context:** Version 1 now exposes aggregate drift signals, but it has no labeled
production feedback, registry, deployment target, or independent evidence that
a detected shift would be repaired by retraining. Its published-test result is
already used and cannot become a recurring selection set.

**Options considered:** Retrain automatically on drift; silently reuse request
text and model predictions as labels; replace artifacts when one aggregate
metric improves; or require a new, provenance-checked and manually approved
candidate cycle.

**Decision:** Drift triggers investigation only. Every candidate must receive a
new immutable cycle identity and pass provenance, validation, leakage, split,
calibration, development-OOD, one-time external, reproducibility,
responsible-AI, and operational gates defined before results are viewed.
Promotion and rollback remain manual and bind complete model, calibrator,
threshold, report, environment, and package identities. Failed and
inconclusive evidence is retained.

**Reason:** Unlabeled traffic cannot establish detector accuracy, self-labeling
would amplify the current model's errors, and test-driven iteration would erase
the meaning of held-out results. Manual, hash-linked gates make the decision
auditable without pretending that infrastructure or independent approval
already exists.

**Tradeoffs:** Replacement is slower and requires a newly valid final-evaluation
plan, explicit reviewers, and artifact retention. There is no current command
that performs promotion or rollback, and this policy does not make the model
production-ready.

## 2026-08-09 — Refit and recalibrate every domain-holdout fold

**Context:** The selector report defines nine leave-one-domain-out regimes.
Applying the already fitted in-distribution model to individual domains would
measure subgroup variation, not whether a model transfers when that domain is
absent from development.

**Options considered:** Reuse the frozen in-distribution model; refit only the
base model while reusing global thresholds; fit one model per domain and tune
on held-domain test results; or give every fold an independent train,
validation, calibration, threshold, and held-domain test cycle.

**Decision:** For every domain, exclude that domain from published train and
validation, fit all three controls, independently select a calibrator and
abstention thresholds on disjoint validation roles, and only then materialize
and score the held domain's published-test rows. Never retune a fold from its
test result. Remove selected text after each fold and retain hash-linked models
and text-free predictions locally.

**Reason:** Independent calibration keeps the comparison faithful to how a new
candidate would be built without the held domain. Deferring test materialization
until the policy is frozen makes the no-test-tuning boundary executable rather
than aspirational.

**Tradeoffs:** Nine cycles cost materially more CPU than subgroup scoring and
produce different thresholds, so they do not describe one deployable policy.
The resulting median ROC AUC is 0.702483, median uncertainty is 67.9364%, and
median human false-machine rate is 13.4251%. These weak results are retained;
they demonstrate within-MAGE domain dependence but do not replace a sealed
cross-dataset evaluation.

## 2026-08-10 — Preserve all human negatives in exact-generator tests

**Context:** Each exact generator contributes only 610 to 1,964 machine rows
to the sanitized published test, while 25,634 human rows are available. A
balanced subsample would simplify aggregate comparison but weaken measurement
of the high-cost human false-machine error.

**Options considered:** Balance every fold by discarding human rows; reuse the
frozen in-distribution model; pool generator identifiers into inferred
families; tune from held-generator results; or retain every human negative and
fit, calibrate, and freeze an independent policy before each exact-generator
test.

**Decision:** Retain all human test rows and only held-generator machine rows.
Exclude the exact generator's machine rows from train and validation, fit and
calibrate each fold independently, checkpoint after verification, and never
retune from its test result. Report prevalence-sensitive metrics with explicit
class imbalance and make no generator-family claim.

**Reason:** The full human population gives the strongest available audit of
false accusations while preserving the exact metadata-supported holdout
question. Checkpointing makes the 27-fold run recoverable without treating a
partial report as complete.

**Tradeoffs:** Machine prevalence is only 2.3243% to 7.1165%, so AP and Brier
are not comparable with balanced evaluations. Median ROC AUC is 0.803724,
median ECE is 0.311882, and median uncertainty is 65.0098%. All five Flan-T5
folds exceed a 10% machine false-human rate. These results reveal exact-source
dependence but do not establish family-level or cross-dataset transfer.

## 2026-08-14 â€” Prespecify paired prefix-truncation stress conditions

**Context:** The first robustness cycle measured natural length bands and MAGE
paraphrase conditions, but it did not isolate the effect of shortening the same
document. The sanitized MAGE test and frozen artifacts are locally available.

**Options considered:** Infer truncation effects from natural length bands;
choose one cutoff after inspecting outcomes; transform the test and then tune
thresholds; wait for a separate attack corpus; or prespecify multiple paired
prefix conditions under the unchanged frozen policy.

**Decision:** Evaluate 50-, 100-, and 200-token prefixes only for records
strictly longer than each budget. Preserve bytes through the final character
of the budget's last non-whitespace token, pair by stable record ID, retain all
conditions, and prohibit model, calibration, or threshold changes from the
result.

**Reason:** Pairing separates the shortening intervention from population
length differences. Fixing budgets before the real run prevents outcome-driven
condition selection while the three sizes cover the existing short-text
warning boundary and two larger prefixes.

**Tradeoffs:** Conditions overlap and are not independent. Prefix deletion is
only one edit and does not represent summaries, middle deletion, adaptive
attacks, or naturally short prose. The measured degradation is a limitation
of the frozen baseline, not a basis for post-test repair.

## 2026-08-15 — Gate a full-train BERT-Tiny candidate with a preflight

**Context:** Transformer evaluation is the first unfinished model dependency,
but the workstation has no GPU, no installed deep-learning framework, and
limited free disk. Python 3.11 is available separately from the application
environment, and the full sanitized training partition is present.

**Options considered:** Keep the phase deferred without an executable next
step; train a reduced sample and compare it with the full-data lexical model;
attempt a larger encoder without a resource measurement; or pin a small public
checkpoint and make resource, dependency, weight, and data gates executable.

**Decision:** Select Google's two-layer BERT-Tiny checkpoint at immutable
revision `30b0a37ccaaa32f332884b96992754e246e48c5f`. Require the complete 287,843-row
sanitized training partition, seed 1729, maximum sequence length 128, isolated
Python 3.11 dependencies, and the existing model-selection gate. Treat any
train-only throughput probe as operational evidence, not model performance.

**Reason:** BERT-Tiny is a real pretrained transformer released for constrained
research, but its 4.39 million parameters make a CPU experiment more plausible
than the previously considered MiniLM candidate. An executable preflight stops
missing packages or weights from being confused with a completed experiment.

**Tradeoffs:** The candidate may underperform larger encoders and the lexical
baseline. The current preflight is `not_ready` because PyTorch, Transformers,
Tokenizers, Accelerate, and the pinned weights are unavailable in this
workspace. No transformer metric or completion claim is made.
