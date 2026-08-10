# MAGE bounded high-overlap audit

This audit looks for cross-partition lexical near duplicates beyond the exact
and normalized equality census. It is intentionally bounded and must not be
read as a population-wide paraphrase analysis. The deterministic, text-free
results are in
[`mage_near_duplicate_report.json`](../../data/metadata/mage_near_duplicate_report.json).
The report exposes the 12 confirmed relationships as hashed record-ID edges so
the split transformation can consume them without storing source text.

## Reproduce

```powershell
python scripts/audit_near_duplicates.py
python scripts/audit_near_duplicates.py --check
```

## Scope and method

The sample selects the lowest 20 stable record IDs in each partition/source
stratum. This yields 17,280 records: 5,760 from each partition, covering all 288
retained source values and all 864 partition/source strata. It is 4.441% of the
389,073-record population.

The sample contains 16,740 machine-positive and 540 human-negative records.
That imbalance follows from giving equal weight to every source value when most
MAGE source values are machine generators. The audit is source-balanced, not
class-representative; its confirmed-pair count is not a population-rate
estimate.

Candidate generation uses bottom-8 BLAKE2b-64 hashes of normalized word
5-shingles plus matching eight-token prefix and suffix blocks. Blocks above 100
members are skipped. Candidate pairs from different partitions are confirmed
with Jaccard similarity over the full hashed shingle sets at a threshold of
0.80. Pairs already equal under the normalized census are excluded.

## Executed results

| Measure | Result |
| --- | ---: |
| Candidate blocks | 142,356 |
| Candidate pairs | 5,863 |
| Normalized-equal candidate pairs excluded | 1 |
| Oversized blocks skipped | 1 |
| Memberships in skipped block | 104 |
| Confirmed high-overlap pairs | 12 |
| Connected groups | 9 |
| Sampled rows in groups | 21 |
| Largest group | 3 rows |
| Conflicting-target pairs | 0 |

Confirmed pairs comprise 6 train/test, 2 train/validation, and 4
validation/test pairs. Confirmed similarity ranges from 0.8125 to 0.964286.

## Interpretation and limits

The executed sample demonstrates that high lexical overlap exists beyond the
normalized-equality groups. It does not establish how common such overlap is in
the full corpus. Candidate blocking can miss pairs with no retained block key,
one oversized block was not expanded, and word-shingle Jaccard does not measure
semantic paraphrase. These limitations remain part of every later leakage and
generalization claim.

The [sanitized in-distribution split](mage_id_split.md) groups every
normalized-equal record and these confirmed sampled high-overlap pairs.
Source-held-out regimes provide an additional defense against source-specific
relationships that a bounded pair audit cannot exhaustively enumerate.
