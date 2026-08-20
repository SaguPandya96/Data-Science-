# Model card: AuthentiText local research baseline v1

## Summary

AuthentiText v1 is a lowercased word unigram/bigram TF-IDF model with balanced
logistic regression, an isotonic calibration map, and two validation-selected
thresholds that create `likely_human`, `uncertain`, and `likely_machine`
outcomes. It is the retained **local research baseline**, not a production or
high-stakes authorship detector.

This system provides a statistical estimate and should not be treated as proof
of authorship. A category describes how the input scores under the frozen MAGE
development setup; it does not identify a writer or generator.

## Model and data identity

| Item | Frozen value |
| --- | --- |
| Dataset | `yaful/MAGE` at `342663f0a2b775455c023f5d36a1341ff0ec5402` |
| Sanitized training rows | 287,843 (86,954 human; 200,889 machine) |
| Base artifact | 1,578,216 bytes; SHA-256 `474481ed4548f89c3d0308bb8389f114f1e1d55a4436c300e3e9fb08d4eb45dd` |
| Calibrator artifact | 2,080 bytes; SHA-256 `ae50d3947e9a541163346e0fd38ad16cbe7a80a98b9b7ffa0036ce80cfea50d3` |
| Feature space | Lowercased word 1–2 grams; 100,000 maximum features; `min_df=5`; `max_df=0.995` |
| Classifier | Balanced logistic regression; `C=1.0`; SAGA; seed 1729 |
| Calibration | Isotonic regression |
| Likely-human maximum | `0.231884057971` |
| Likely-machine minimum | `0.717391304348` |

The machine-generated class is canonical target 1. Source identifiers, record
IDs, and other metadata are retained for auditing but are not model features.
The model was trained only on the sanitized MAGE train partition. MAGE
WritingPrompts rows were excluded because that source overlaps the planned
Ghostbuster external corpus, and known normalized/lexical overlap components
were handled before fitting. See the [data-lineage record](data/DATA_LINEAGE.md)
and [model-selection decision](MODEL_SELECTION.md).

The base and calibration artifacts are ignored by Git. Their sizes and hashes
are committed so a local copy can be verified before loading. Redistribution is
not approved by this card: MAGE is released as Apache-2.0, but terms for its
aggregated upstream human sources still require case-by-case review.

## Calibration and threshold roles

The 50,509-row sanitized validation partition was assigned by record hash to
three disjoint roles. Test data was not used for calibration or threshold
selection.

| Role | Rows |
| --- | ---: |
| Fit isotonic or sigmoid calibration | 20,135 |
| Select calibration method and thresholds | 15,304 |
| Audit the selected policy | 15,070 |

Isotonic regression was selected over raw and sigmoid scores by validation
Brier score, then expected calibration error (ECE), with a deterministic
tie-break. The two thresholds targeted 5% cross-class decisive errors on the
selection role. They were frozen before the published-test evaluation and were
not adjusted when that test missed both point targets.

## Evaluation

The reported values below are read from the committed training, validation,
calibration, frozen-test, and OOD reports. Average precision (AP) depends on
class prevalence. Validation uses the raw model for Brier/ECE; test and OOD use
the selected isotonic calibration.

| Evaluation | Rows | ROC AUC | AP | Brier | ECE | Uncertain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sanitized MAGE validation, raw model | 50,509 | 0.814070 | 0.828215 | 0.176634 | 0.043285 | n/a |
| Frozen sanitized MAGE test | 50,567 | 0.806435 | 0.821832 | 0.177779 | 0.009833 | 57.0708% |
| Deduplicated MAGE development OOD | 3,162 | 0.697370 | 0.867506 | 0.225546 | 0.218292 | 66.1607% |
| Frozen Ghostbuster external | 20,991 | 0.828929 | 0.962187 | 0.130866 | 0.152084 | 42.9660% |

| Frozen policy evaluation | Coverage | Human → machine | Machine → human | Decisive accuracy |
| --- | ---: | ---: | ---: | ---: |
| Sanitized MAGE test | 42.9292% | 5.2391% | 5.9880% | 86.9357% |
| Deduplicated MAGE development OOD | 33.8393% | 12.3360% | 3.7500% | 82.8037% |
| Frozen Ghostbuster external | 57.0340% | 12.5334% | 1.3001% | 94.9131% |

The OOD set is a MAGE development stress test containing GPT-4, GPT-4
paraphrase, and machine-paraphrased-human conditions. Its higher AP reflects a
75.9% machine-positive prevalence and is not evidence of improved
generalization. It is not the sealed cross-dataset external test.

Nine additional models were independently fitted with one MAGE domain excluded
from both training and validation. Each fold selected its own isotonic
calibrator and abstention thresholds before scoring the held domain.

| Leave-one-domain-out metric | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| ROC AUC | 0.625869 | 0.702483 | 0.790932 |
| Average precision | 0.650331 | 0.721720 | 0.813803 |
| Brier score | 0.188402 | 0.220005 | 0.254679 |
| Expected calibration error | 0.044945 | 0.096889 | 0.136629 |
| Coverage | 24.1379% | 32.0636% | 42.8159% |
| Uncertain rate | 57.1841% | 67.9364% | 75.8621% |
| Human false-machine rate | 3.5833% | 13.4251% | 23.0015% |
| Machine false-human rate | 0.6552% | 2.0301% | 4.1487% |

All held-domain ROC AUC values are below the frozen in-distribution aggregate.
Five folds exceed a 10% human false-machine rate. The full per-domain results
and protocol are in the
[leave-one-domain-out evaluation](evaluation/mage_domain_holdouts.md).

Twenty-seven additional models were independently fitted with one exact MAGE
generator excluded from machine train and validation rows. Each test retains
all human rows plus only held-generator machine rows.

| Leave-one-exact-generator-out metric | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| ROC AUC | 0.636999 | 0.803724 | 0.906497 |
| Average precision | 0.055240 | 0.260767 | 0.576487 |
| Brier score | 0.150343 | 0.154304 | 0.155801 |
| Expected calibration error | 0.275423 | 0.311882 | 0.322583 |
| Coverage | 30.0990% | 34.9902% | 45.5653% |
| Uncertain rate | 54.4347% | 65.0098% | 69.9010% |
| Human false-machine rate | 3.8894% | 5.1611% | 7.3613% |
| Machine false-human rate | 1.3004% | 4.6599% | 18.7726% |

Machine prevalence ranges from 2.3243% to 7.1165%, so AP and Brier are not
directly comparable with the balanced in-distribution test. Fourteen folds are
below its ROC AUC, and all five Flan-T5 folds exceed a 10% machine false-human
rate. The [exact-generator evaluation](evaluation/mage_generator_holdouts.md)
contains the protocol and per-generator evidence.

The sealed Ghostbuster evaluation uses 20,991 overlap-gated records: 2,992
human and 17,999 machine. Its 85.7463% machine prevalence inflates AP and
accuracy relative to balanced tests. The frozen external policy has 42.9660%
uncertainty, 12.5334% human false-machine, and 1.3001% machine false-human. ECE
rises to 0.152084. Domain behavior remains sharply uneven: human false-machine
is 1.6032% for Reuters news, 11.3000% for creative writing, and 24.7485% for
student essays. Only 21.5000% of Claude records reach likely machine, while
72.3333% are uncertain. See the
[external evaluation](evaluation/ghostbuster_external.md) for the frozen
protocol, overlap limits, Wilson intervals, and prompt-strategy outcomes.

A deterministic qualitative review examined 9 score-extreme human
false-machine, 6 score-extreme machine false-human, and 6 uncertain-boundary
Ghostbuster records. Human false-machine cases included three narratives,
three newswire reports, and three academic essays rather than one visible
genre. All six machine false-human cases used formal or formulaic prose; four
were academic-style and two were first-person narratives. This single-reviewer
excerpt analysis is descriptive, not a prevalence estimate or causal feature
attribution. Its text-free evidence is in the
[error-review report](evaluation/ghostbuster_error_review.md).

### Material subgroup failures

| Frozen-test slice | Rows | Uncertain | Human → machine | Machine → human | Decisive accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Under 50 whitespace tokens | 13,208 | 74.0612% | 9.6887% | 11.9406% | 58.8441% |
| HellaSwag domain | 6,346 | 65.1592% | 13.7303% | 1.0478% | 78.1095% |
| Yelp domain | 5,199 | 45.2972% | 0.5279% | 25.4417% | 76.7229% |
| Ghostbuster student essays | 6,994 | 46.1395% | 24.7485% | 2.5000% | 89.4877% |

These slices show why aggregate ranking and calibration do not justify an
individual decision. Wilson intervals and the complete domain, generator,
strategy, and length results are preserved in
[`mage_frozen_test_report.json`](../data/metadata/mage_frozen_test_report.json).
The [MAGE OOD report](evaluation/mage_ood.md) documents further domain and
paraphrase failures.

The prespecified prefix-truncation stress test compares each eligible complete
MAGE test record with a deterministic prefix under the unchanged frozen
policy. Populations differ across budgets, but each row is paired within its
condition.

| Prefix budget | Paired rows | Original ROC AUC | Prefix ROC AUC | Original uncertain | Prefix uncertain | Category changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50-token prefix | 36,965 | 0.878027 | 0.671298 | 50.8346% | 69.7849% | 36.0801% |
| 100-token prefix | 25,887 | 0.923264 | 0.805082 | 45.3896% | 62.7187% | 30.0846% |
| 200-token prefix | 14,228 | 0.945366 | 0.886284 | 39.5558% | 52.4670% | 23.1094% |

At 50 tokens, human false-machine increases from 3.3848% to 6.7473% and
machine false-human increases from 4.1176% to 9.5497%. This is evidence of
prefix-removal sensitivity, not a claim about all naturally short or edited
text. See the
[truncation robustness report](evaluation/mage_truncation_robustness.md).

## Intended use

Appropriate uses are limited to:

- reproducible research on detector calibration, abstention, and shift;
- local demonstrations of uncertainty-first inference and privacy-safe service
  design; and
- exploratory analysis where a knowledgeable human reviews the input, output,
  warnings, and known evaluation gaps.

The output must not be used as the sole or primary basis for academic
discipline, employment, fraud allegations, moderation sanctions, legal action,
or any other consequential decision. It is not designed for writer or model
attribution, plagiarism detection, fact checking, quality scoring,
psychological inference, or surveillance.

## Input and output behavior

The current contract accepts one nonblank Unicode string up to 100,000 code
points and does not truncate or rewrite it. The vectorizer scores the complete
accepted input. Text below 50 whitespace tokens receives a low-evidence
warning, but warnings do not alter the frozen category. Inputs outside the
observed development length range or with out-of-profile formatting also
receive warnings.

The calibrated number estimates a target frequency under the MAGE validation
setup. It is not a universal probability that a person or model authored the
text. Submitted text is not returned, logged, or persisted by default; runtime
monitoring stores process-local aggregates only. See the
[inference contract](inference/contract.md) and
[monitoring contract](operations/monitoring.md).

## Performance and environment

| Measured batch | Rows scored | Seconds | Records/second |
| --- | ---: | ---: | ---: |
| Validation | 50,509 | 15.148 | 3,334.271 |
| Frozen test | 50,567 | 15.528 | 3,256.566 |
| Raw OOD files, including repeated controls | 3,924 | 1.548 | 2,535.609 |
| Ghostbuster external | 20,991 | 14.024 | 1,496.805 |

These are batch measurements from the audited Windows workstation, not API
latency or production-capacity guarantees. The host had 8 logical processors,
15.82 GiB RAM, Intel HD Graphics 630, and no CUDA tooling. Training used Python
3.14.6, scikit-learn 1.9.0, NumPy 2.5.1, SciPy 1.18.0, and Joblib 1.5.3.
Peak training or service memory was not measured, and no GPU requirement or
GPU performance claim is made.

## Known limitations and unmeasured areas

- The first cycle is English-only and MAGE-specific. Multilingual behavior is
  unknown.
- Lexical features can learn domain, topic, source, and formatting artifacts
  instead of authorship-relevant signals.
- Short-text and domain errors are severe, and abstention does not eliminate
  false classifications.
- OOD calibration and human false-machine behavior degrade substantially.
- Mixed-authorship, quoted passages, code, tables, non-English text, and most
  edit/adversarial conditions were not evaluated.
- The full-data BERT-Tiny candidate improved in-distribution metrics but failed
  the MAGE OOD gate (0.558414 ROC AUC, 0.390341 ECE, and error rates above 18%);
  it is not the shipped runtime model.
- Ghostbuster supplies cross-dataset evidence, but its 12.5334% aggregate and
  24.7485% student-essay human false-machine rates prohibit a safety claim.
- The bounded qualitative review is single-reviewer and excerpt-based; it does
  not estimate error-pattern prevalence or establish causal model features.
- The free portfolio deployment passed its technical acceptance checks, but no
  production-user monitoring study or model-drift outcome evaluation has
  occurred.

The model must not be described as production-ready, externally validated, or
capable of proving authorship. The project-wide
[responsible-AI and use policy](RESPONSIBLE_AI.md) governs every presentation
and downstream use of these results.

## Reproduction and change control

With the pinned ignored data, predictions, and artifacts present, the existing
records can be verified with:

```powershell
python scripts/train_baselines.py --verify-only
python scripts/evaluate_baselines.py --verify-only
python scripts/calibrate_baseline.py --verify-only
python scripts/evaluate_frozen_test.py --verify-only
python scripts/evaluate_ood.py --verify-only
python scripts/run_domain_holdouts.py --verify-only
python scripts/run_generator_holdouts.py --verify-only
python scripts/evaluate_ghostbuster.py --verify-only
python scripts/check_model_card.py
```

The published-test result is immutable for this version. A replacement model
or threshold policy must start a separately identified train, calibration, and
evaluation cycle and pass the gates in
[`docs/MODEL_SELECTION.md`](MODEL_SELECTION.md).
