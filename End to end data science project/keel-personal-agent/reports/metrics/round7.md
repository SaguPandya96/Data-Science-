# Round 7: cross-encoder reranking

First stage: Keel with `all-mpnet-base-v2` at weight 12.0. 200 personas, k = 5. Intervals are 95% persona bootstrap.

**Outcome: ms-marco-MiniLM-L-6-v2 on the top 20, weight 2.0 adopted.**

Chosen on already-seen sets: `ms-marco-MiniLM-L-6-v2` on the top 20, weight 2.0.

## Choosing the setting (mean clean hit over `dev`, `holdout`, `holdout2`, `holdout3`, `holdout4`)

Without reranking: 87.3%.

### `ms-marco-MiniLM-L-6-v2`

| Short list | w = 0.25 | w = 0.5 | w = 1.0 | w = 2.0 | w = 4.0 | w = only |
| --- | --- | --- | --- | --- | --- | --- |
| top 10 | 87.6% | 87.9% | 88.3% | 88.3% | 88.0% | 86.5% |
| top 20 | 87.7% | 88.0% | 88.5% | 88.6% | 88.5% | 86.2% |
| top 30 | 87.7% | 88.0% | 88.5% | 88.6% | 88.4% | 85.8% |

### `ms-marco-MiniLM-L-12-v2`

| Short list | w = 0.25 | w = 0.5 | w = 1.0 | w = 2.0 | w = 4.0 | w = only |
| --- | --- | --- | --- | --- | --- | --- |
| top 10 | 87.5% | 87.7% | 87.9% | 88.0% | 87.9% | 86.0% |
| top 20 | 87.5% | 87.7% | 87.9% | 88.0% | 87.8% | 85.0% |
| top 30 | 87.5% | 87.7% | 87.9% | 88.0% | 87.7% | 84.2% |

## New held-out set (all)

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `keel+larger` | 85.8% (85.4 to 86.3) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+rerank` | 88.1% (87.7 to 88.5) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+rerank` minus:

- `keel+larger`: +2.3 pts (+1.9 to +2.6)

## New held-out set: direct

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `keel+larger` | 100.0% (100.0 to 100.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+rerank` | 99.9% (99.8 to 100.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+rerank` minus:

- `keel+larger`: -0.1 pts (-0.2 to +0.0)

## New held-out set: indirect

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `keel+larger` | 71.6% (70.7 to 72.5) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+rerank` | 76.2% (75.4 to 77.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+rerank` minus:

- `keel+larger`: +4.6 pts (+3.9 to +5.3)
