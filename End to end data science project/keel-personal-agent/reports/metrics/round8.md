# Round 8: quantized models

200 personas, k = 5. Intervals are 95% persona bootstrap.

**Outcome: encoder fp32, reranker int8 adopted (no worse than the current models).**

## Model sizes

| Model | MB |
| --- | --- |
| `all-mpnet-base-v2` | 435.8 |
| `all-mpnet-base-v2 (int8)` | 110.1 |
| `ms-marco-MiniLM-L-6-v2` | 91.0 |
| `ms-marco-MiniLM-L-6-v2 (int8)` | 23.1 |

## Choosing the variant (mean clean hit over `dev`, `holdout`, `holdout2`, `holdout3`, `holdout4`, `holdout5`)

Current models: 88.6%.

| Variant | Encoder weight | Mean clean hit |
| --- | --- | --- |
| encoder int8, reranker fp32 | 8.0 (best) | 86.3% |
| encoder int8, reranker fp32 | 12.0 | 86.0% |
| encoder int8, reranker fp32 | 16.0 | 85.9% |
| encoder fp32, reranker int8 | 12.0 (best) | 88.4% |
| encoder int8, reranker int8 | 8.0 (best) | 86.1% |
| encoder int8, reranker int8 | 12.0 | 85.9% |
| encoder int8, reranker int8 | 16.0 | 85.8% |

Within 0.5 points: encoder fp32, reranker int8.

## New held-out set (all)

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `keel+rerank` | 91.5% (91.1 to 92.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+rerank(int8)` | 91.7% (91.2 to 92.1) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+rerank(int8)` minus:

- `keel+rerank`: +0.1 pts (-0.0 to +0.3)

## New held-out set: direct

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `keel+rerank` | 99.6% (99.4 to 99.8) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+rerank(int8)` | 99.5% (99.3 to 99.8) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+rerank(int8)` minus:

- `keel+rerank`: -0.1 pts (-0.2 to +0.1)

## New held-out set: indirect

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `keel+rerank` | 83.5% (82.6 to 84.3) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+rerank(int8)` | 83.8% (83.0 to 84.6) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+rerank(int8)` minus:

- `keel+rerank`: +0.3 pts (+0.0 to +0.7)
