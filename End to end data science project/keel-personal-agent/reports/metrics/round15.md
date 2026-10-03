# Round 15: newer encoder architectures

Every encoder runs under `ms-marco-MiniLM-L-6-v2 (int8)` on the top 20. 200 personas, k = 5. Intervals are 95% persona bootstrap.

**Outcome: no encoder beat e5-large-v2 on the tuning sets (best: gte-modernbert-base at w = 24.0); e5-large-v2 kept.**

## Choosing the encoder (mean clean hit over `dev`, `holdout`, `holdout2`, `holdout3`, `holdout4`, `holdout5`, `holdout6`, `holdout7`, `holdout8`)

Current: `e5-large-v2` at w = 64.0, 91.5%.

| Encoder | MB | w = 4.0 | w = 8.0 | w = 12.0 | w = 16.0 | w = 24.0 | w = 32.0 | w = 48.0 | w = 64.0 | w = 96.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gte-modernbert-base` | 596.4 | 88.8% | 89.4% | 89.9% | 90.1% | 90.2% | 90.2% | 90.0% | 89.9% | 89.7% |
| `modernbert-embed-base` | 596.5 | 88.2% | 88.9% | 88.8% | 88.8% | 88.5% | 88.4% | 88.2% | 88.0% | 87.8% |
| `modernbert-embed-large` | 1579.6 | 87.8% | 88.4% | 88.4% | 88.2% | 87.8% | 87.5% | 87.1% | 86.9% | 86.5% |
| `gte-large-en-v1.5` | 1745.6 | 87.8% | 88.1% | 88.3% | 88.3% | 87.8% | 87.5% | 87.0% | 86.6% | 86.0% |

The held-out set was not used.
