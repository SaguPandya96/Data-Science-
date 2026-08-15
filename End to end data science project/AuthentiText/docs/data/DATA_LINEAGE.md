# Data lineage

## MAGE OOD acquisition slice

The first acquired data is a deliberately small, development-only slice from
the public [MAGE dataset](https://huggingface.co/datasets/yaful/MAGE). The
release is pinned to commit
`342663f0a2b775455c023f5d36a1341ff0ec5402`, last modified upstream on
2024-05-22. This acquisition contains two files:

| File | Bytes | SHA-256 | Rows | Observed label counts |
| --- | ---: | --- | ---: | --- |
| `test_ood_set_gpt.csv` | 2,191,864 | `b9fe1faaa45fbbad446f527da12d42e3e9e3bbd207f7ea054e81ec8e922016d2` | 1,562 | `0`: 800; `1`: 762 |
| `test_ood_set_gpt_para.csv` | 3,614,100 | `a587909d155447ea2768cbd8f695a7febb65cdd7414da5c3b5ff6d6397463725` | 2,362 | `0`: 1,600; `1`: 762 |

Both files were retrieved and independently hashed on 2026-08-08. Their
observed header is `text,label,src`; neither file had an empty `text` value in
the initial aggregate inspection. The official MAGE README defines raw label
`0` as machine-generated and raw label `1` as human-written. AuthentiText's
canonical target will use machine-generated as the positive class, so a later
reproducible transformation will map raw `0` to canonical `1` and raw `1` to
canonical `0`; the acquired files remain unchanged.

No transformation has been applied. The raw CSVs live under `data/raw/mage/`
and are ignored by Git. The versioned
[`mage_ood_manifest.json`](../../data/metadata/mage_ood_manifest.json) records
the pinned URLs, byte sizes, SHA-256 digests, upstream Git blob identifiers, and
expected columns. Re-run acquisition or local verification with:

```powershell
python scripts/download_data.py
python scripts/download_data.py --verify-only
```

Aggregate profiles contain no source text and are generated deterministically:

```powershell
python scripts/profile_data.py
python scripts/profile_data.py --check
```

The first profile validates 3,924 rows: 2,400 machine-generated and 1,524
human-written after interpreting the raw labels. All required fields are
present and nonblank. It also finds 762 exact-text groups spanning both files;
all carry raw label `1` (human-written) and none has a label conflict. The OOD
files are therefore dependent views, not independent evaluation samples. Any
combined analysis must group these repeats or remove the duplicated copy.
The raw schema also has no stable record identifier; future processed records
must receive deterministic IDs derived from stable source and content fields,
not mutable CSV row numbers.

These files are development stress tests for generator and paraphrase shift.
They are not the untouched external evaluation set and must not be used to
claim cross-dataset generalization. Ghostbuster remains sealed for that role.

The frozen-policy [OOD evaluation](../evaluation/mage_ood.md) verifies both raw
file hashes, derives stable record IDs from the dataset/revision/file identity,
content hash, label, and source, and writes a text-free ignored prediction
artifact. Its committed
[`mage_ood_evaluation_report.json`](../../data/metadata/mage_ood_evaluation_report.json)
reports both files separately and removes one copy of each of the 762 repeated
human controls for combined metrics. The evaluation does not transform the raw
CSVs or read the published in-distribution test partition.

The MAGE repository and Hugging Face card state Apache-2.0. Because MAGE
aggregates human text from other corpora, upstream source terms must still be
reviewed before redistribution. Raw data is therefore not committed or
packaged.

## MAGE development partitions

The official `train.csv`, `valid.csv`, and `test.csv` partitions use the same
pinned dataset revision and raw schema as the OOD slice. Their separate
[`mage_development_manifest.json`](../../data/metadata/mage_development_manifest.json)
records upstream Git LFS SHA-256 identities and byte sizes. All three files
were retrieved and independently verified on 2026-08-08:

| File | Bytes | SHA-256 | Observed rows |
| --- | ---: | --- | ---: |
| `train.csv` | 403,744,528 | `07b95008964d6be412aa764aa1e515d652c345305cebc522a7612c6015ba373a` | 319,071 |
| `valid.csv` | 72,276,577 | `005e200957a4da551c8ac37079f43811db9c32f5141e3bdabb1caa257588691f` | 56,792 |
| `test.csv` | 71,739,623 | `dcb8a8a8de1459d382ac2da5e0f1c39452e8bc47d186d7393f3ed4308f762ac7` | 56,819 |

The acquisition check observed 432,682 rows in total, with the expected
`text,label,src` columns and no null or blank text values or malformed rows.
Detailed class, source, length, and duplication results are generated in the
separate [development profile](mage_development_profile.md). Acquire or verify
these files with:

```powershell
python scripts/download_data.py `
  --manifest data/metadata/mage_development_manifest.json
python scripts/download_data.py `
  --manifest data/metadata/mage_development_manifest.json `
  --verify-only
```

These partitions are development data, not the sealed cross-dataset final
evaluation. Records whose source belongs to WritingPrompts will be identified
and excluded before model fitting, tuning, calibration, threshold selection,
or internal reporting because Ghostbuster contains the same upstream domain.

## Ghostbuster sealed external corpus

The official Ghostbuster data repository is pinned to commit
`86ebd72590556a81622986fab736ab9227a948af`. The deterministic selector found
all 21,000 main-corpus files reported in the paper across three domains and
seven conditions. The selected source files total 70,601,168 bytes. Six human
essay files are blank upstream and are explicitly recorded and excluded; there
were no UTF-8 failures. The resulting 20,994 records contain 2,994 human and
18,000 machine texts.

Raw files and the prepared 27,319,331-byte gzip JSON Lines corpus remain
ignored. The committed
[`ghostbuster_main_manifest.json`](../../data/metadata/ghostbuster_main_manifest.json)
records the input manifest hash, output SHA-256, exact selection, blank paths,
and text-free aggregate profile. The [preparation protocol](ghostbuster_main.md)
is reproducible and has passed both creation-time verification and a separate
verify-only run. It does not score the frozen detector. Exact, normalized, and
near-overlap checks against sanitized MAGE remain a mandatory gate before the
one-time external evaluation.

That gate subsequently passed and reproduced byte-for-byte. Across 388,919
sanitized MAGE and 20,994 usable Ghostbuster records, it found no exact,
normalized, or confirmed 0.8 word-5-gram-Jaccard cross-dataset match. The
bounded lexical pass tested 100,013 candidate pairs; its skipped-block and
tokenless-record limitations are recorded in the text-free
[`ghostbuster_overlap_report.json`](../../data/metadata/ghostbuster_overlap_report.json).
Three redundant internal normalized Ghostbuster copies are excluded by stable
record ID, leaving 20,991 records eligible for frozen scoring.

The frozen scorer subsequently ran exactly once and its verification-only mode
recomputed all metrics from the ignored text-free predictions. The committed
[`ghostbuster_evaluation_report.json`](../../data/metadata/ghostbuster_evaluation_report.json)
links the prepared input, overlap report, base model, calibrator, thresholds,
and 20,991 prediction IDs. The full [external evaluation](../evaluation/ghostbuster_external.md)
reports the observed calibration and subgroup failures without retuning.

## Processed MAGE development data

[`mage_cleaning_report.json`](../../data/metadata/mage_cleaning_report.json)
records the deterministic raw-to-processed transformation. Text is preserved
exactly after UTF-8 decoding. Sources beginning with `wp_` are excluded with
reason `shared_upstream_domain_with_sealed_external_corpus`; no other row is
removed. MAGE raw labels are retained as metadata and mapped to the canonical
machine-positive target.

Each output record contains exact-text `content_id`, partition-specific
`record_id`, dataset revision, partition, source, raw label, canonical target,
and text. The processed gzip JSON Lines files remain under
`data/processed/mage_development/` and are ignored by Git. Their byte sizes and
SHA-256 identities are versioned in the report.

| Partition | Input rows | Excluded | Output rows | Target 1: machine | Target 0: human |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 319,071 | 31,159 | 287,912 | 200,950 | 86,962 |
| Validation | 56,792 | 6,214 | 50,578 | 24,912 | 25,666 |
| Test | 56,819 | 6,236 | 50,583 | 24,941 | 25,642 |
| **Total** | **432,682** | **43,609** | **389,073** | **250,803** | **138,270** |

Generate or verify the processed files with the locked environment active:

```powershell
python scripts/clean_data.py
python scripts/clean_data.py --verify-only
```

The 52 exact validation/test duplicate groups are intentionally retained in
this canonical layer with shared content IDs. The later split transformation
applies and reconciles a deterministic cross-partition policy.

The subsequent [exact and normalized leakage analysis](mage_leakage_analysis.md)
measures both raw content-ID equality and NFKC/case/whitespace-normalized
equality without emitting source text. Its versioned report is the input to the
near-duplicate audit and final split policy.

The [bounded high-overlap audit](mage_near_duplicate_audit.md) then applies a
deterministic source-balanced sample and explicit candidate caps. Its report is
kept separate so sampled lexical findings cannot be confused with the
population-wide exact/normalized census.

## Sanitized MAGE in-distribution split

The [in-distribution split transformation](mage_id_split.md) consumes the
canonical records, all normalized-equality relationships, and the 12 confirmed
sampled high-overlap edges. It keeps records in their published partitions,
drops conflicting-target components, and otherwise keeps one deterministic
representative using train, test, then validation precedence. The ignored
outputs are under `data/processed/mage_splits/id/`; the versioned
[`mage_id_split_report.json`](../../data/metadata/mage_id_split_report.json)
records full row reconciliation and output byte identities.

| Partition | Canonical input | Dropped | Split output | Target 1: machine | Target 0: human |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 287,912 | 69 | 287,843 | 200,889 | 86,954 |
| Validation | 50,578 | 69 | 50,509 | 24,901 | 25,608 |
| Test | 50,583 | 16 | 50,567 | 24,933 | 25,634 |
| **Total** | **389,073** | **154** | **388,919** | **250,723** | **138,196** |

The [source-holdout regime record](mage_source_regimes.md) parses all 288
retained upstream source identifiers and defines selector-only domain and
generator regimes over these outputs. No additional text copies are created.
Its versioned report records measured role, target, and source counts for all
36 regimes.

## Leave-one-domain-out experiment artifacts

The nine domain selectors are executed directly from the verified sanitized
split. For each fold, train and validation rows from the held domain are
excluded. Held-domain published-test rows are not materialized until after that
fold's model, calibration method, and thresholds are fixed. Selected fold text
is deleted after scoring.

The ignored `artifacts/generalization/domain/` tree contains 72 files totaling
73.95 MiB: models, calibrators, and text-free validation/test predictions. The
committed
[`mage_domain_holdout_report.json`](../../data/metadata/mage_domain_holdout_report.json)
records each selected input identity, artifact/prediction identity,
configuration, timing, calibration evidence, test metric, and validation flag.
The [evaluation report](../evaluation/mage_domain_holdouts.md) presents the
measured comparison. Reproduce or verify it with:

```powershell
python scripts/run_domain_holdouts.py
python scripts/run_domain_holdouts.py --verify-only
```

## Leave-one-exact-generator-out experiment artifacts

The 27 exact-generator selectors are executed directly from the same verified
sanitized split. Each fold excludes held-generator machine rows from train and
validation. Its test role retains all published-test human rows plus only the
held generator's machine rows, and is not materialized until that fold's model,
calibrator, and thresholds are frozen. Selected text is deleted after scoring.

The ignored `artifacts/generalization/generator/` tree contains 216 files
totaling 269.41 MiB. The committed
[`mage_generator_holdout_report.json`](../../data/metadata/mage_generator_holdout_report.json)
records selector identities, checkpoints, artifact and prediction identities,
timing, calibration evidence, test metrics, and validation flags. The
[evaluation report](../evaluation/mage_generator_holdouts.md) presents the
measured comparison. Reproduce, resume, or verify it with:

```powershell
python scripts/run_generator_holdouts.py
python scripts/run_generator_holdouts.py --resume
python scripts/run_generator_holdouts.py --verify-only
```

## Paired MAGE prefix-truncation artifacts

The prefix-truncation stress test reads the verified sanitized MAGE test
directly and does not create another text dataset. For each prespecified
budget, records longer than the budget are paired with an in-memory prefix that
ends at the budget's last non-whitespace token. The original bytes before that
cut are preserved, and the transformed text is discarded after scoring.

The ignored `artifacts/predictions/robustness/mage_truncation_pairs.jsonl.gz`
contains 77,080 text-free paired rows totaling 5,333,750 bytes with SHA-256
`bc3aad34fe515013b3bf7c9e4008cc9f963fffbd2283f31b105749cc1ba6ffb8`.
The committed
[`mage_truncation_robustness_report.json`](../../data/metadata/mage_truncation_robustness_report.json)
links the sanitized input, base model, calibrator, conditions, aggregate
metrics, and prediction identity. Run or verify it with:

```powershell
python scripts/evaluate_truncation_robustness.py
python scripts/evaluate_truncation_robustness.py --verify-only
```

## Aggregate drift reference

[`mage_drift_reference.json`](../../data/metadata/mage_drift_reference.json) is
derived only from the sanitized validation split, its text-free stored baseline
predictions, and the frozen calibrator. The builder verifies all three inputs
against their existing report hashes and joins them by record ID. The committed
artifact uses 33,647 validation rows for its aggregate reference and the other
16,862 rows for disjoint false-alert and domain-shift audits. It contains fixed
aggregate distributions, empirical thresholds, and backtest results—no text or
per-record values. The published test partition is not an input. Rebuild or
verify it with:

```powershell
python scripts/build_drift_reference.py
python scripts/build_drift_reference.py --verify-only
```
