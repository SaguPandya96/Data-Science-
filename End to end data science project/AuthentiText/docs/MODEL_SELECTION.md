# First-cycle model selection

## Decision

Retain the word unigram/bigram TF-IDF logistic model with the frozen isotonic
calibrator and abstention thresholds as AuthentiText's version 1 **local
research baseline**. Reject the majority and length-only controls. Do not
approve any evaluated model as a reliable authorship detector or high-stakes
production system.

A full-data BERT-Tiny candidate was trained, calibrated on disjoint validation
roles, frozen, and evaluated once on the published test and MAGE development
OOD roles. Reject it for version 1 deployment: its in-distribution improvement
does not compensate for its severe OOD ranking, calibration, and error
regression.

## Candidate evidence

All three implemented candidates were trained on the 287,843-row sanitized
training partition and evaluated on the same 50,509 validation rows.

| Candidate | Artifact | Validation ROC AUC | Validation AP | Threshold-0.5 human FPR | Batch throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority/prevalence | 106 B | 0.500000 | 0.493001 | 1.000000 | Not meaningful |
| Length logistic | 928 B | 0.529177 | 0.531670 | 0.518549 | 69,586 records/s |
| Word TF-IDF logistic | 1,578,216 B | 0.814070 | 0.828215 | 0.266089 | 3,334 records/s |
| BERT-Tiny | 18,262,086 B | 0.851317 | 0.868015 | 0.342354 | 629 records/s on frozen test |

The word model is the only candidate with materially useful ranking. Its raw
0.5 threshold is rejected; the selected product contract is the separately
calibrated three-way policy.

## Frozen-policy evidence

On the 50,567-row sanitized in-distribution test, the selected baseline has ROC
AUC 0.806435 and AP 0.821832. Isotonic Brier score is 0.177779 and 15-bin ECE is
0.009833. The abstaining policy covers 42.9292% of records and is uncertain on
57.0708%.

The decisive results are not within both intended 5% error points:

- human false-machine: 5.2391% (1,343 / 25,634);
- machine false-human: 5.9880% (1,493 / 24,933); and
- decisive accuracy: 86.9357%.

Aggregate performance conceals severe subgroup failures. HellaSwag human
false-machine rate is 13.7303%; Yelp machine false-human rate is 25.4417%.
Below 50 whitespace tokens, decisive accuracy is 58.8441%, human false-machine
is 9.6887%, and machine false-human is 11.9406%.

On 3,162 exact-content-deduplicated MAGE development OOD texts, ROC AUC drops to
0.697370, calibration ECE rises to 0.218292, and 66.1607% are uncertain. Human
false-machine rate is 12.3360%. The likely-machine rate is 40.1250% for GPT-4,
30.0000% for GPT-4 paraphrases, and 21.1250% for machine-paraphrased human text.

Nine independently trained and calibrated leave-one-domain-out folds have a
median ROC AUC of 0.702483 and median uncertainty of 67.9364%. Median human
false-machine rate is 13.4251%; five folds exceed 10%, and the maximum is
23.0015% on Yelp. These within-MAGE results confirm that refitting outside a
domain does not remove the baseline's domain dependence.

Twenty-seven independently trained and calibrated exact-generator holdouts
have median ROC AUC 0.803724 and median uncertainty 65.0098%. Fourteen folds
fall below the frozen in-distribution ROC AUC. Every fold has ECE above 27%,
and all five Flan-T5 folds exceed a 10% machine false-human rate, reaching
18.7726% on `flan_t5_small`. Related generator families can remain in training,
so these results measure exact-identifier transfer rather than family or
external generalization.

On 20,991 overlap-gated Ghostbuster external records, the unchanged policy has
ROC AUC 0.828929 but ECE 0.152084. It is uncertain on 42.9660%, calls 12.5334%
of human records likely machine, and calls 1.3001% of machine records likely
human. Human false-machine rises to 24.7485% on student essays. Only 21.5000%
of Claude records reach likely machine. The high 0.962187 AP is
prevalence-sensitive because 85.7463% of the external rows are machine. These
results add cross-dataset evidence but strengthen, rather than remove, the
prohibition on consequential use.

These failures prevent a production or high-stakes recommendation even though
the lexical model is small, fast in batches, reproducible, and easy to inspect.

## Transformer evaluation decision

The pinned two-layer BERT-Tiny candidate trained on all 287,843 sanitized rows
for three epochs on a hosted CPU runner. Its saved model reload matched exactly,
and the hash-frozen model, isotonic calibrator, and thresholds were selected
without test data. On 50,567 test rows it reached 0.851966 ROC AUC, 0.866533 AP,
0.009303 ECE, and 44.5627% coverage. Human false-machine was 4.7554% and machine
false-human was 3.1645%, both better than the lexical baseline.

The generalization gate failed decisively. On the same 3,162 deduplicated MAGE
OOD texts, transformer ROC AUC was 0.558414 versus 0.697370 for the lexical
baseline. ECE was 0.390341 versus 0.218292. Human false-machine rose to 19.0289%
and machine false-human to 18.3750%, compared with 12.3360% and 3.7500% for the
baseline. The transformer was also roughly five times slower in the measured
batch comparisons and produced an 18.26 MB artifact instead of 1.58 MB.

These results are retained without retuning. BERT-Tiny is a completed and
rejected candidate, not deferred work and not the runtime model.

## Selection rationale

The word model remains the shipped research baseline because it:

- dominates both implemented controls on validation ranking;
- has a frozen, independently audited calibration and abstention contract;
- fits the verified CPU and memory environment;
- scores tens of thousands of texts in a reproducible batch within seconds;
- produces a compact 1.58 MB artifact; and
- supports hash verification and text-free prediction audits.

Its selection is conditional on preserving all warnings, the uncertain state,
artifact provenance, and the prohibition against treating results as proof.
There is no fallback to the weaker controls when the lexical model is uncertain.

## Gate for a future candidate

Any transformer or other candidate must be a new evaluation cycle. Before it
can replace version 1, it must:

1. declare pretrained-weight identity, license, architecture, maximum length,
   truncation policy, environment, seed, and measured resource use;
2. train only on the sanitized training role and use new predefined validation
   roles for calibration and policy selection;
3. compare human false-machine, machine false-human, coverage, calibration,
   short-text, domain, generator, and MAGE OOD outcomes—not only aggregate AUC;
4. freeze its decision rule before any new final evaluation;
5. preserve text-free predictions and complete hash-linked reports; and
6. demonstrate a meaningful safety/generalization gain large enough to justify
   its runtime, dependency, and maintenance cost.

Until those gates are met, AuthentiText remains an honest baseline and research
interface, not a validated production detector.
