# Round 13: other encoders of e5-large's size

Every encoder runs under `ms-marco-MiniLM-L-6-v2 (int8)` on the top 20. 200 personas, k = 5. Intervals are 95% persona bootstrap.

**Outcome: no encoder beat e5-large-v2 on the tuning sets (best: gte-large at w = 32.0); e5-large-v2 kept.**

## Choosing the encoder (mean clean hit over `dev`, `holdout`, `holdout2`, `holdout3`, `holdout4`, `holdout5`, `holdout6`, `holdout7`, `holdout8`)

Current: `e5-large-v2` at w = 64.0, 91.5%.

| Encoder | MB | w = 4.0 | w = 8.0 | w = 12.0 | w = 16.0 | w = 24.0 | w = 32.0 | w = 48.0 | w = 64.0 | w = 96.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `e5-large` | 1336.9 | 86.8% | 86.7% | 86.7% | 86.7% | 86.7% | 86.7% | 86.4% | 85.8% | 85.0% |
| `snowflake-arctic-embed-l` | 1336.9 | 87.6% | 87.9% | 88.2% | 88.4% | 88.5% | 88.5% | 88.0% | 87.6% | 87.1% |
| `gte-large` | 1336.9 | 88.1% | 88.3% | 88.6% | 88.9% | 89.3% | 89.5% | 89.4% | 89.3% | 89.1% |

The held-out set was not used.
