# Ghostbuster sealed main corpus

Ghostbuster is the cross-dataset external evaluation corpus. Its model,
calibrator, and two abstaining policy thresholds were frozen before acquisition.
No Ghostbuster text may be used for feature design, model fitting, calibration,
threshold selection, or exploratory error review before the one-time evaluation.

The source is the [official data repository](https://github.com/vivek3141/ghostbuster-data),
pinned to commit `86ebd72590556a81622986fab736ab9227a948af`. The
[NAACL paper](https://aclanthology.org/2024.naacl-long.95/) reports 21,000 main-corpus
documents in Table 5: 1,000 human, 5,000 ChatGPT, and 1,000 Claude documents in
each of three domains. The repository licenses the data under CC BY 3.0.

The deterministic selector includes exactly 1,000 numeric `.txt` files from
each combination of:

- `essay`, `reuter`, and `wp`;
- `human`, `gpt`, `claude`, `gpt_prompt1`, `gpt_prompt2`, `gpt_semantic`, and
  `gpt_writing`.

For Reuters, the selected files are one directory below the condition, grouped
by author. The selector excludes nested log-probability files, prompt files,
`perturb`, `perturb_old`, `other`, and every auxiliary evaluation set. This
yields the paper's 3,000 human and 18,000 machine main-corpus records without
mixing in additional conditions.

Clone the repository into the ignored raw-data location and check out the
pinned revision:

```powershell
git clone https://github.com/vivek3141/ghostbuster-data `
  data/raw/ghostbuster/repository
git -C data/raw/ghostbuster/repository checkout `
  86ebd72590556a81622986fab736ab9227a948af
```

Prepare or verify the ignored deterministic gzip JSON Lines file with:

```powershell
python scripts/prepare_ghostbuster.py
python scripts/prepare_ghostbuster.py --verify-only
```

Preparation verifies the pinned Git revision and clean tracked tree, hashes
every selected input, rejects missing or non-UTF-8 files, excludes and records
blank source documents, maps machine text to target `1`, and records
deterministic record, exact-content, and normalized-content identifiers.
Duplicate record identifiers are rejected. The committed manifest contains
only counts, identities, source-relative paths for excluded blanks, and
aggregate length statistics; raw text remains ignored. Preparation explicitly
performs no model scoring.

The real preparation and a separate verify-only pass completed on 2026-08-10.
The selector found all 21,000 declared files (70,601,168 bytes) and no UTF-8
failures. Six human essay files are blank upstream and were excluded by the
documented rule, leaving 20,994 usable records: 2,994 human and 18,000 machine.
The three domain counts are 6,994 student essays, 7,000 news articles, and
7,000 creative-writing documents. The ignored prepared file is 27,319,331
bytes with SHA-256
`27238c202fd7f3b5a27620193cf0a8f24d5b5c034fe5547915eadd06cd3c0a66`.
The text-free
[`ghostbuster_main_manifest.json`](../../data/metadata/ghostbuster_main_manifest.json)
records the complete identities and reconciliation.

Before the one-time frozen evaluation, a separate overlap gate must compare the
prepared corpus with every sanitized MAGE development partition using exact,
normalized, and declared near-duplicate checks. Any detected overlap must be
resolved by a versioned rule without inspecting detector outcomes. The frozen
evaluation may run only after that gate passes.

Run or reproduce that gate with:

```powershell
python scripts/audit_external_overlap.py
python scripts/audit_external_overlap.py --check
```

The gate performs population exact and normalized comparisons across all
388,919 sanitized MAGE records and every usable Ghostbuster record. Its bounded
near check uses population-wide candidate blocking with bottom-eight word
5-gram hashes plus prefix and suffix blocks, followed by exact Jaccard
calculation at a frozen 0.8 threshold. Blocks with more than 100 external
members and total candidate pairs beyond 2,000,000 are explicit safety bounds.
Every confirmed external match, and every redundant internal normalized copy,
is excluded by record ID before scoring. This rule uses no detector output.

The real audit and a complete `--check` reproduction passed on 2026-08-10.
Population equality checks found zero exact and zero normalized cross-dataset
pairs. The bounded population blocking pass retained 195,255 blocks and tested
100,013 candidate pairs; it found zero cross-dataset pairs at or above 0.8
word-5-gram Jaccard. Six blocks with 1,146 total external memberships exceeded
the declared 100-member limit and were skipped. One MAGE record had no Unicode
word tokens and therefore participated in equality checks but not lexical
blocking. These limitations mean the near check is strong lexical evidence,
not a semantic-overlap census.

Ghostbuster itself contains one exact-duplicate group and three normalized
duplicate groups. Applying the predeclared normalized-copy rule excludes three
records (two human and one machine), leaving 20,991 independent external rows
for scoring. The text-free
[`ghostbuster_overlap_report.json`](../../data/metadata/ghostbuster_overlap_report.json)
contains the stable excluded IDs and verified input hashes. No model scoring or
outcome review occurred during the audit.

The one-time scorer refuses to overwrite an existing evaluation report. Its
first invocation verifies the preparation manifest, source repository, overlap
gate, frozen base-model hash, calibration linkage, and validation-derived
thresholds before applying the three exclusions. Later invocations are
verification-only:

```powershell
python scripts/evaluate_ghostbuster.py
python scripts/evaluate_ghostbuster.py --verify-only
```

The prediction artifact is ignored and contains stable IDs, targets, metadata,
raw scores, calibrated scores, and categories, but no text. Verification
recomputes every metric and proves that prediction IDs equal the prepared IDs
minus the overlap report's exclusion set. External outcomes may document model
limitations but may not change the model, calibrator, or thresholds.

The one-time run and its verification-only replay completed on 2026-08-10.
Results and limitations are preserved in the
[frozen external evaluation](../evaluation/ghostbuster_external.md).
