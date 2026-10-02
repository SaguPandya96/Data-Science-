# Round 14: encoders of about 2.2 GB

Every encoder runs under `ms-marco-MiniLM-L-6-v2 (int8)` on the top 20. 200 personas, k = 5. Intervals are 95% persona bootstrap.

**Outcome: no encoder beat e5-large-v2 on the tuning sets (best: multilingual-e5-large at w = 64.0); e5-large-v2 kept.**

## Choosing the encoder (mean clean hit over `dev`, `holdout`, `holdout2`, `holdout3`, `holdout4`, `holdout5`, `holdout6`, `holdout7`, `holdout8`)

Current: `e5-large-v2` at w = 64.0, 91.5%.

| Encoder | MB | w = 4.0 | w = 8.0 | w = 12.0 | w = 16.0 | w = 24.0 | w = 32.0 | w = 48.0 | w = 64.0 | w = 96.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `multilingual-e5-large` | 2235.9 | 88.5% | 89.0% | 89.4% | 89.6% | 90.3% | 90.8% | 91.2% | 91.3% | 91.2% |
| `multilingual-e5-large-instruct` | 2236.1 | 88.1% | 88.4% | 88.6% | 89.0% | 89.6% | 90.0% | 90.3% | 90.3% | 90.3% |
| `snowflake-arctic-embed-l-v2.0` | 2267.6 | 86.6% | 86.2% | 86.0% | 85.7% | 85.1% | 84.6% | 83.9% | 83.5% | 83.1% |

The held-out set was not used.
