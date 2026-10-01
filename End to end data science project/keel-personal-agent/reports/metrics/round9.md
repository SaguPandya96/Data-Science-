# Round 9: smaller encoders

Every encoder runs under `ms-marco-MiniLM-L-6-v2 (int8)` on the top 20. 200 personas, k = 5. Intervals are 95% persona bootstrap.

**Outcome: no smaller encoder came within 0.5 points on the tuning sets; mpnet kept.**

## Choosing the encoder (mean clean hit over `dev`, `holdout`, `holdout2`, `holdout3`, `holdout4`, `holdout5`, `holdout6`)

Current: `all-mpnet-base-v2` at w = 12.0, 88.9%.

| Encoder | MB | w = 4.0 | w = 8.0 | w = 12.0 | w = 16.0 | w = 24.0 | w = 32.0 | w = 48.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `all-MiniLM-L6-v2` | 90.4 | 87.8% | 87.7% | 87.3% | 87.2% | 86.8% | 86.6% | 86.4% |
| `bge-small-en-v1.5` | 132.9 | 85.8% | 85.3% | 85.1% | 84.6% | 84.1% | 83.7% | 83.1% |
| `all-MiniLM-L12-v2` | 133.1 | 87.9% | 87.5% | 87.1% | 87.0% | 86.8% | 86.5% | 86.1% |
| `gte-small` | 133.1 | 87.3% | 87.3% | 87.3% | 87.5% | 87.6% | 87.5% | 87.3% |

Within 0.5 points: none.

The held-out set was not used.
