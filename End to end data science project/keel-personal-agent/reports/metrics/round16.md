# Round 16: an int8 e5-large-v2

Every encoder runs under `ms-marco-MiniLM-L-6-v2 (int8)` on the top 20. 200 personas, k = 5. Intervals are 95% persona bootstrap.

**Outcome: e5-large-v2 (int8) at w = 64.0 adopted.**

## Choosing the encoder (mean clean hit over `dev`, `holdout`, `holdout2`, `holdout3`, `holdout4`, `holdout5`, `holdout6`, `holdout7`, `holdout8`)

Current: `e5-large-v2` at w = 64.0, 91.5%.

| Encoder | MB | w = 4.0 | w = 8.0 | w = 12.0 | w = 16.0 | w = 24.0 | w = 32.0 | w = 48.0 | w = 64.0 | w = 96.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `e5-large-v2 (int8)` | 337.0 | 88.5% | 89.1% | 89.7% | 90.1% | 90.6% | 90.9% | 91.2% | 91.3% | 91.4% |

## New held-out set (all)

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `keel+e5large` | 93.4% (92.9 to 93.8) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+e5large(int8)` | 93.0% (92.5 to 93.4) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+e5large(int8)` minus:

- `keel+e5large`: -0.4 pts (-0.7 to -0.0)

## New held-out set: direct

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `keel+e5large` | 97.5% (97.1 to 97.9) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+e5large(int8)` | 97.2% (96.8 to 97.7) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+e5large(int8)` minus:

- `keel+e5large`: -0.2 pts (-0.6 to +0.1)

## New held-out set: indirect

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `keel+e5large` | 89.2% (88.5 to 90.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+e5large(int8)` | 88.7% (87.9 to 89.5) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+e5large(int8)` minus:

- `keel+e5large`: -0.5 pts (-1.1 to +0.1)
