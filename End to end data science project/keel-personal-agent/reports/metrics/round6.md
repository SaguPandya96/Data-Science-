# Round 6: larger transformer encoders

200 personas, k = 5. Intervals are 95% persona bootstrap.

**Outcome: all-mpnet-base-v2 at w = 12.0 adopted.**

Chosen on already-seen sets: `all-mpnet-base-v2` at weight 12.0.

## Choosing the encoder and weight (mean clean hit over `dev`, `holdout`, `holdout2`, `holdout3`)

| Encoder | w = 2.0 | w = 4.0 | w = 6.0 | w = 8.0 | w = 12.0 |
| --- | --- | --- | --- | --- | --- |
| `all-MiniLM-L6-v2` | 85.0% | 86.2% | 86.6% | 86.4% | 86.0% |
| `bge-base-en-v1.5` | 82.5% | 83.4% | 83.6% | 83.8% | 83.8% |
| `all-mpnet-base-v2` | 85.3% | 86.5% | 86.8% | 86.8% | 86.8% |

## New held-out set (all)

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `transformer` | 48.9% (47.6 to 50.2) | 28.0% (26.8 to 29.2) | 2.2% (1.9 to 2.5) |
| `keel+transformer` | 85.5% (85.1 to 85.9) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `larger` | 52.7% (51.4 to 54.2) | 31.1% (29.7 to 32.5) | 2.9% (2.5 to 3.3) |
| `keel+larger` | 89.3% (89.0 to 89.5) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+larger` minus:

- `keel+transformer`: +3.8 pts (+3.4 to +4.2)
- `transformer`: +40.4 pts (+39.1 to +41.6)
- `larger`: +36.6 pts (+35.1 to +37.9)

## New held-out set: direct

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `transformer` | 61.4% (59.8 to 63.2) | 36.7% (35.1 to 38.2) | 2.5% (2.0 to 2.9) |
| `keel+transformer` | 100.0% (100.0 to 100.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `larger` | 61.9% (60.3 to 63.6) | 36.6% (35.0 to 38.2) | 1.8% (1.4 to 2.3) |
| `keel+larger` | 100.0% (100.0 to 100.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+larger` minus:

- `keel+transformer`: +0.0 pts (+0.0 to +0.0)
- `transformer`: +38.6 pts (+36.8 to +40.2)
- `larger`: +38.1 pts (+36.4 to +39.7)

## New held-out set: indirect

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `transformer` | 36.3% (34.9 to 37.8) | 19.4% (18.2 to 20.6) | 1.9% (1.5 to 2.4) |
| `keel+transformer` | 71.0% (70.1 to 71.8) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `larger` | 43.5% (42.1 to 45.0) | 25.7% (24.3 to 27.0) | 4.0% (3.5 to 4.6) |
| `keel+larger` | 78.5% (78.0 to 79.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+larger` minus:

- `keel+transformer`: +7.6 pts (+6.8 to +8.4)
- `transformer`: +42.2 pts (+40.8 to +43.7)
- `larger`: +35.0 pts (+33.5 to +36.5)
