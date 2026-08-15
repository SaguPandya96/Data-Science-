# MAGE development profile

This report interprets the deterministic, text-free
[`mage_development_profile.json`](../../data/metadata/mage_development_profile.json).
It describes the exact pinned files acquired on 2026-08-08; it does not report
model performance.

## Reproduce

```powershell
python scripts/profile_data.py `
  --manifest data/metadata/mage_development_manifest.json `
  --output data/metadata/mage_development_profile.json
python scripts/profile_data.py `
  --manifest data/metadata/mage_development_manifest.json `
  --output data/metadata/mage_development_profile.json `
  --check
```

## Composition and quality

| Partition | Rows | Machine | Human | Source values | Median characters | P95 characters | Median whitespace tokens | P95 whitespace tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 319,071 | 225,753 | 93,318 | 322 | 675 | 4,394 | 115 | 783 |
| Validation | 56,792 | 27,993 | 28,799 | 322 | 681 | 4,399 | 115 | 785 |
| Test | 56,819 | 28,078 | 28,741 | 322 | 677 | 4,382 | 115 | 782 |
| **Total** | **432,682** | **281,824** | **150,858** | **322** | **676** | **4,393** | **115** | **783** |

The raw labels were interpreted using MAGE's official mapping: `0` is
machine-generated and `1` is human-written. All required fields are present and
nonblank. The profile found no malformed rows, unknown labels, or exact-text
label conflicts. A whitespace token is simply a run separated by Python's
`str.split()`; it is a descriptive length measure, not a model tokenizer.

## WritingPrompts exclusion

Ghostbuster contains the same upstream WritingPrompts domain, so all MAGE
source values beginning with `wp_` are excluded from development before any
fitting, tuning, calibration, threshold selection, or internal evaluation. The
rule matches 34 observed source values and 43,609 rows:

| Partition | Machine excluded | Human excluded | Total excluded | Rows remaining |
| --- | ---: | ---: | ---: | ---: |
| Train | 24,803 | 6,356 | 31,159 | 287,912 |
| Validation | 3,081 | 3,133 | 6,214 | 50,578 |
| Test | 3,137 | 3,099 | 6,236 | 50,583 |
| **Total** | **31,021** | **12,588** | **43,609** | **389,073** |

After this source exclusion, 250,803 machine and 138,270 human rows remain.
These are population counts, not a chosen training sample.

## Exact-text dependency

The official validation and test partitions share 52 exact-text groups
containing 104 rows. Fifty groups carry the human raw label and two carry the
machine raw label; none conflicts. No exact duplicate group involving the
training partition was observed. The official validation and test files must
therefore not be treated as fully independent until processing applies a
deterministic cross-partition policy.

This check hashes the exact decoded text. Unicode-normalized and near-duplicate
checks, prompt relationships, templates, and metadata artifacts remain for the
dedicated leakage-analysis phase.
