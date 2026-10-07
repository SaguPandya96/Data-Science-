# Round 17: the reranker's setting for the current encoder

First stage: `e5-large-v2 (int8)` at w = 64.0. Reranker: `ms-marco-MiniLM-L-6-v2 (int8)`. 200 personas, k = 5. Intervals are 95% persona bootstrap.

**Outcome: the current setting is within the tie margin of the best on the tuning sets; the current 20 candidates at weight 2.0 kept.**

## Choosing the setting (mean clean hit over `dev`, `holdout`, `holdout2`, `holdout3`, `holdout4`, `holdout5`, `holdout6`, `holdout7`, `holdout8`, `holdout9`)

Current: 20 candidates at weight 2.0.

| Candidates | w = 0.5 | w = 1.0 | w = 2.0 | w = 4.0 | w = 8.0 | w = 16.0 |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 91.49% | 91.54% | 91.55% | 91.31% | 90.99% | 90.54% |
| 20 | 91.50% | 91.55% | 91.50% | 91.17% | 90.80% | 90.18% |
| 30 | 91.50% | 91.55% | 91.42% | 90.95% | 90.49% | 89.75% |
| 40 | 91.50% | 91.55% | 91.41% | 90.89% | 90.34% | 89.46% |

The held-out set was not used.
