# MAGE sanitized in-distribution split

The `id` split is the leakage-sanitized form of MAGE's published development
partitions. It is intended for the first in-distribution baseline, not for
cross-domain or cross-generator generalization claims. The deterministic,
text-free reconciliation is in
[`mage_id_split_report.json`](../../data/metadata/mage_id_split_report.json).
The split files remain ignored under `data/processed/mage_splits/id/`.

## Reproduce and verify

```powershell
python scripts/create_splits.py
python scripts/create_splits.py --verify-only
```

Creation first verifies the canonical input hashes from the cleaning report.
Verification checks every output byte size, SHA-256, record count, target count,
partition field, and cross-file record-ID uniqueness.

## Component policy

The transformation computes the NFKC, case-folded, whitespace-collapsed value
of every one of the 389,073 canonical records. Records with the same normalized
value are connected. The 12 hashed record-ID edges confirmed by the bounded
high-overlap audit are then added to the same graph.

For each connected component:

1. If it contains both canonical targets, every member is dropped with reason
   `conflicting_target_component`.
2. Otherwise, one member is retained using partition precedence `train`,
   `test`, then `validation`.
3. If multiple members remain tied within that partition, the lowest stable
   record ID wins.

Records retain their upstream partition; none are moved. The precedence keeps
known training copies out of evaluation and prefers the final internal test
over validation when the two published holdouts overlap. This is a policy
choice recorded for reproduction, not a claim that the upstream split is
scientifically optimal.

## Executed reconciliation

The graph contains 388,920 components. There are 149 multi-record components,
including one conflicting-target component and nine components containing the
12 sampled high-overlap edges. The transformation drops 152 redundant members
and both members of the conflicting component.

| Partition | Input | Grouped overlap | Target conflict | Output |
| --- | ---: | ---: | ---: | ---: |
| Train | 287,912 | 67 | 2 | 287,843 |
| Validation | 50,578 | 69 | 0 | 50,509 |
| Test | 50,583 | 16 | 0 | 50,567 |
| **Total** | **389,073** | **152** | **2** | **388,919** |

| Partition | Human target 0 | Machine target 1 | Output SHA-256 |
| --- | ---: | ---: | --- |
| Train | 86,954 | 200,889 | `9bb04cac540ac2aad1249adbd7cf1023a6da538eff5519a7bb11024ffb4c6918` |
| Validation | 25,608 | 24,901 | `64a997db40e98059389fed9e1dd593015e45da68f2dddd22d67e098c6eadec39` |
| Test | 25,634 | 24,933 | `0fe309466be85f146e37bfdbf1fee30024286193b8756972a52a6d313827d44d` |

All 389,073 inputs reconcile to 388,919 outputs plus 154 documented removals.

## Scope limitation

Normalized equality is a population census, but the additional lexical edges
come from a deterministic 4.441% source-balanced sample. The split removes all
known relationships supplied to it; it does not prove that no unsampled
near-duplicate, paraphrase, template, or semantic relationship remains. Later
source-group regimes address a different and broader generalization risk.
