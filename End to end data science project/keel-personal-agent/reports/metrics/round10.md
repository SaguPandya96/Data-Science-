# Round 10: MiniLM below weight 4

Every encoder runs under `ms-marco-MiniLM-L-6-v2 (int8)` on the top 20. 200 personas, k = 5. Intervals are 95% persona bootstrap.

**Outcome: no smaller encoder came within 0.5 points on the tuning sets; mpnet kept.**

## Choosing the encoder (mean clean hit over `dev`, `holdout`, `holdout2`, `holdout3`, `holdout4`, `holdout5`, `holdout6`)

Current: `all-mpnet-base-v2` at w = 12.0, 88.9%.

| Encoder | MB | w = 0.5 | w = 1.0 | w = 2.0 | w = 3.0 | w = 4.0 |
| --- | --- | --- | --- | --- | --- | --- |
| `all-MiniLM-L6-v2` | 90.4 | 87.2% | 87.3% | 87.4% | 87.6% | 87.8% |
| `all-MiniLM-L12-v2` | 133.1 | 87.4% | 87.4% | 87.5% | 87.9% | 87.9% |

Within 0.5 points: none.

The held-out set was not used.
