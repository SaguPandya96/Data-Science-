# MAGE source-holdout regimes

The source-holdout report defines development-time generalization regimes over
the sanitized `id` split. It stores selectors and measured counts only; it does
not duplicate record text. The complete text-free results are in
[`mage_source_regimes.json`](../../data/metadata/mage_source_regimes.json).

## Reproduce

```powershell
python scripts/define_source_regimes.py
python scripts/define_source_regimes.py --check
```

The command first verifies all three sanitized split files by byte size and
SHA-256. It then parses and counts their source metadata. A check run recomputes
the complete report and requires byte-for-byte equality.

## Domain-holdout execution

The leave-one-domain-out runner materializes only one ignored fold at a time,
checks its selector counts against this report, fits the three baseline
controls, and calibrates the selected word TF-IDF model using only the
domain-excluded validation role. It does not materialize or score the held
domain's published-test text until that fold's model, calibration method, and
thresholds are fixed. Temporary selected text is removed after each fold;
models and text-free predictions remain ignored local artifacts.

Run or verify the complete nine-fold family with:

```powershell
python scripts/run_domain_holdouts.py
python scripts/run_domain_holdouts.py --verify-only
```

All nine real folds have executed and passed artifact and prediction
verification. The measured ranking, calibration, abstention, and cross-class
error results are in the
[leave-one-domain-out evaluation](../evaluation/mage_domain_holdouts.md).

## Observed source vocabulary

All 288 retained source values match one of two forms:

- `<domain>_human`
- `<domain>_machine_<strategy>_<generator>`

The parser observes nine domains, 27 exact generator identifiers, and the three
strategies `continuation`, `specified`, and `topical`. Nine source values are
human and 279 are machine. Generator identifiers are opaque upstream metadata;
for example, the observed `gpt-3.5-trubo` spelling is preserved rather than
silently corrected.

## Leave-one-domain-out

For each domain, published train and validation rows from that domain are
excluded. The test role contains only published test rows from the held domain.
Records from other domains' published test partitions are excluded rather than
moved. All nine regimes contain both targets in train, validation, and test.

| Measure across nine regimes | Minimum | Maximum |
| --- | ---: | ---: |
| Train rows | 235,490 | 265,211 |
| Validation rows | 43,980 | 45,667 |
| Test rows | 4,788 | 6,537 |
| Test human rows | 2,400 | 3,292 |
| Test machine rows | 2,250 | 3,254 |

## Leave-one-exact-generator-out

For each exact generator identifier, its machine rows are excluded from
published train and validation. The test role contains that generator's
published-test machine rows across every available strategy and domain, plus
all 25,634 published-test human rows. Other generators' test rows are excluded.
All 27 regimes contain both targets in every active role.

| Measure across 27 regimes | Minimum | Maximum |
| --- | ---: | ---: |
| Train rows | 272,084 | 282,858 |
| Validation rows | 48,547 | 49,920 |
| Test rows | 26,244 | 27,598 |
| Test human rows | 25,634 | 25,634 |
| Test held-generator machine rows | 610 | 1,964 |

The generator tests are intentionally imbalanced because every available human
negative is retained for false-positive analysis. Accuracy alone will be
misleading; later evaluation must include balanced accuracy, AUROC, average
precision, precision/recall, and false-positive rates with explicit supports.

The generator runner fits, calibrates, and evaluates an independent policy per
regime, materializes held-generator test text only after that policy is frozen,
and removes selected text after scoring. It checkpoints after every completed
fold so the longer 27-fold run can be verified and resumed without silently
reusing incomplete work:

```powershell
python scripts/run_generator_holdouts.py
python scripts/run_generator_holdouts.py --resume
python scripts/run_generator_holdouts.py --verify-only
```

All 27 real folds have executed, checkpointed, and passed a separate
artifact/prediction verification pass. The measured ranking, calibration,
abstention, and cross-class error results are in the
[leave-one-exact-generator-out evaluation](../evaluation/mage_generator_holdouts.md).

## Limits

The source field supplies no separate prompt identifier, so these selectors
cannot prove prompt-disjointness. Exact generator IDs also do not establish
architecture families: two related identifiers can occur on opposite sides of
a generator fold. These regimes measure the metadata-supported questions they
state and no broader claim.
