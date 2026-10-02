# Round 12: larger encoders

Every encoder runs under `ms-marco-MiniLM-L-6-v2 (int8)` on the top 20. 200 personas, k = 5. Intervals are 95% persona bootstrap.

**Outcome: e5-large-v2 at w = 64.0 adopted.**

## Choosing the encoder (mean clean hit over `dev`, `holdout`, `holdout2`, `holdout3`, `holdout4`, `holdout5`, `holdout6`, `holdout7`)

Current: `e5-base-v2` at w = 48.0, 90.6%.

| Encoder | MB | w = 4.0 | w = 8.0 | w = 12.0 | w = 16.0 | w = 24.0 | w = 32.0 | w = 48.0 | w = 64.0 | w = 96.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `e5-large-v2` | 1336.9 | 88.2% | 88.8% | 89.5% | 90.1% | 90.7% | 90.8% | 91.1% | 91.2% | 91.2% |
| `bge-large-en-v1.5 (with query instruction)` | 1336.9 | 86.9% | 86.8% | 86.9% | 86.7% | 86.3% | 86.0% | 85.5% | 85.2% | 84.8% |
| `mxbai-embed-large-v1` | 1336.9 | 87.6% | 87.9% | 87.7% | 87.5% | 87.2% | 86.8% | 86.3% | 86.0% | 85.5% |

## New held-out set (all)

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `keel+e5` | 90.1% (89.8 to 90.5) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+larger` | 93.9% (93.5 to 94.3) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+larger` minus:

- `keel+e5`: +3.8 pts (+3.4 to +4.2)

## New held-out set: direct

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `keel+e5` | 100.0% (100.0 to 100.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+larger` | 100.0% (100.0 to 100.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+larger` minus:

- `keel+e5`: +0.0 pts (+0.0 to +0.0)

## New held-out set: indirect

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `keel+e5` | 80.2% (79.5 to 81.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+larger` | 87.8% (87.1 to 88.6) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+larger` minus:

- `keel+e5`: +7.6 pts (+6.7 to +8.4)
