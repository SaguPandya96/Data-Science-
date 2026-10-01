# Round 11: other encoders of mpnet's size

Every encoder runs under `ms-marco-MiniLM-L-6-v2 (int8)` on the top 20. 200 personas, k = 5. Intervals are 95% persona bootstrap.

**Outcome: e5-base-v2 at w = 48.0 adopted.**

## Choosing the encoder (mean clean hit over `dev`, `holdout`, `holdout2`, `holdout3`, `holdout4`, `holdout5`, `holdout6`)

Current: `all-mpnet-base-v2` at w = 12.0, 88.9%.

| Encoder | MB | w = 4.0 | w = 8.0 | w = 12.0 | w = 16.0 | w = 24.0 | w = 32.0 | w = 48.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gte-base` | 435.8 | 87.6% | 87.8% | 87.8% | 88.0% | 88.5% | 88.6% | 88.6% |
| `multi-qa-mpnet-base-dot-v1` | 435.8 | 89.4% | 90.0% | 90.1% | 90.0% | 89.8% | 89.5% | 89.1% |
| `e5-base-v2` | 435.8 | 88.3% | 88.8% | 89.6% | 89.9% | 90.3% | 90.4% | 90.4% |
| `nomic-embed-text-v1.5` | 547.3 | 88.2% | 89.1% | 89.6% | 89.7% | 89.8% | 89.7% | 89.3% |
| `snowflake-arctic-embed-m-v1.5` | 435.9 | 86.4% | 86.2% | 86.1% | 85.9% | 85.2% | 84.6% | 83.8% |
| `bge-base-en-v1.5 (with query instruction)` | 435.8 | 87.1% | 87.1% | 87.1% | 87.0% | 86.9% | 86.7% | 86.6% |

## New held-out set (all)

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `keel+rerank` | 91.1% (90.7 to 91.6) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+other` | 92.4% (92.0 to 92.9) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+other` minus:

- `keel+rerank`: +1.3 pts (+0.7 to +1.8)

## New held-out set: direct

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `keel+rerank` | 99.9% (99.8 to 100.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+other` | 99.9% (99.8 to 100.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+other` minus:

- `keel+rerank`: +0.0 pts (+0.0 to +0.0)

## New held-out set: indirect

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `keel+rerank` | 82.3% (81.5 to 83.2) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+other` | 84.9% (84.0 to 85.8) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+other` minus:

- `keel+rerank`: +2.6 pts (+1.4 to +3.7)
