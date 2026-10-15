# Round 18: other cross-encoder rerankers

First stage: `e5-large-v2 (int8)` at w = 64.0, top 20 reranked. 200 personas, k = 5. Intervals are 95% persona bootstrap.

**Outcome: jina-reranker-v1-turbo-en (int8) at weight 0.5 adopted.**

## Choosing the reranker (mean clean hit over `dev`, `holdout`, `holdout2`, `holdout3`, `holdout4`, `holdout5`, `holdout6`, `holdout7`, `holdout8`, `holdout9`, `holdout10`)

Current: `ms-marco-MiniLM-L-6-v2 (int8)` at weight 2.0.

| Reranker | w = 0.5 | w = 1.0 | w = 2.0 | w = 4.0 | w = 8.0 |
| --- | --- | --- | --- | --- | --- |
| `ms-marco-MiniLM-L-6-v2 (int8)` | 91.31% | 91.37% | 91.33% | 90.97% | 90.47% |
| `jina-reranker-v1-turbo-en (int8)` | 92.43% | 92.23% | 91.40% | 90.30% | 89.32% |
| `ms-marco-MiniLM-L-12-v2 (int8)` | 91.23% | 91.28% | 91.08% | 90.73% | 90.07% |

## New held-out set (all)

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `keel+rerank` | 88.5% (87.9 to 89.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+rerank(other)` | 91.1% (90.6 to 91.6) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+rerank(other)` minus:

- `keel+rerank`: +2.6 pts (+2.1 to +3.1)

## New held-out set: direct

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `keel+rerank` | 94.8% (94.3 to 95.3) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+rerank(other)` | 96.8% (96.3 to 97.2) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+rerank(other)` minus:

- `keel+rerank`: +2.0 pts (+1.5 to +2.5)

## New held-out set: indirect

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `keel+rerank` | 82.1% (81.2 to 83.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+rerank(other)` | 85.4% (84.6 to 86.2) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+rerank(other)` minus:

- `keel+rerank`: +3.3 pts (+2.4 to +4.2)
