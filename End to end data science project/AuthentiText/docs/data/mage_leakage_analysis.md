# MAGE exact and normalized leakage analysis

This analysis covers all 389,073 processed MAGE development records. The
deterministic, text-free
[`mage_leakage_report.json`](../../data/metadata/mage_leakage_report.json)
revalidates every processed content hash, record ID, target mapping, input file
identity, and population count before measuring overlap.

## Reproduce

```powershell
python scripts/analyze_leakage.py
python scripts/analyze_leakage.py --check
```

## Exact decoded text

| Measure | Result |
| --- | ---: |
| Unique exact-text content IDs | 389,021 |
| Duplicate groups | 52 |
| Rows in duplicate groups | 104 |
| Cross-partition groups | 52 |
| Validation–test groups | 52 |
| Conflicting-target groups | 0 |

Every exact duplicate is one validation/test pair. No exact duplicate involves
the training partition, occurs only within a partition, or carries conflicting
targets.

## Normalized exact text

The normalized key applies Unicode NFKC compatibility normalization,
case-folding, and whitespace collapsing before hashing. It does not remove
punctuation or perform semantic comparison.

| Measure | Result |
| --- | ---: |
| Unique normalized values | 388,932 |
| Duplicate groups | 141 |
| Rows in duplicate groups | 282 |
| Within-partition groups | 76 |
| Cross-partition groups | 65 |
| Rows in cross-partition groups | 130 |
| Normalization-only cross-partition groups | 13 |
| Conflicting-target groups | 1 |

The 65 cross-partition groups comprise 6 train/validation, 6 train/test, and 53
validation/test groups. The single conflicting-target group is inside the
training partition. It is a data finding, not an analyzer failure, and must be
removed or otherwise resolved before model fitting.

## Source overlap

All 288 retained source values appear in train, validation, and test. The
published partitions are therefore source-overlapping in-distribution splits.
They cannot by themselves support claims about unseen domains, generators, or
source families. Separate deterministic source-group regimes are required.

## Implications and remaining work

Any reuse of the published partitions must operate on normalized-text groups,
not individual rows. A conflict policy and partition precedence will be fixed
only after the bounded near-duplicate audit so removals reconcile once. Exact
and normalized equality do not detect paraphrase, partial copying, templates,
or shared prompts; those remain explicitly unmeasured in this report.
