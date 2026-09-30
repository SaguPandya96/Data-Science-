# Round 5: a wider weight grid for MiniLM

Encoder `all-MiniLM-L6-v2`. 200 personas, k = 5. Intervals are 95% persona bootstrap.

**Outcome: w = 6.0 adopted.** Chosen weight 6.0; current weight 2.0.

## Choosing the weight (clean hit on already-seen question sets)

| Weight | `dev` | `holdout` | `holdout2` | Mean |
| --- | --- | --- | --- | --- |
| 1.0 | 97.5% | 78.5% | 75.5% | 83.8% |
| 2.0 | 98.8% | 81.7% | 74.9% | 85.2% |
| 3.0 | 99.6% | 83.5% | 74.4% | 85.9% |
| 4.0 | 99.8% | 84.2% | 74.3% | 86.1% |
| 6.0 (chosen) | 99.8% | 85.0% | 74.2% | 86.3% |
| 8.0 | 99.6% | 85.0% | 74.1% | 86.2% |
| 12.0 | 99.6% | 84.8% | 73.9% | 86.1% |
| 16.0 | 99.6% | 84.7% | 73.8% | 86.0% |

## Every weight on the new held-out set (reported, not used to choose)

| Weight | All | Direct | Indirect |
| --- | --- | --- | --- |
| 1.0 | 83.1% | 99.7% | 66.5% |
| 2.0 | 84.4% | 100.0% | 68.8% |
| 3.0 | 84.5% | 100.0% | 69.0% |
| 4.0 | 86.2% | 100.0% | 72.5% |
| 6.0 | 87.4% | 100.0% | 74.8% |
| 8.0 | 86.8% | 100.0% | 73.7% |
| 12.0 | 85.7% | 100.0% | 71.3% |
| 16.0 | 85.1% | 100.0% | 70.1% |

## New held-out set (all)

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `transformer` | 49.4% (48.2 to 50.7) | 27.7% (26.4 to 28.9) | 2.4% (2.1 to 2.7) |
| `keel+embed` | 67.6% (67.1 to 68.1) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+transformer` | 84.4% (84.0 to 84.8) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+transformer(tuned)` | 87.4% (87.0 to 87.8) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+transformer(tuned)` minus:

- `keel+transformer`: +3.0 pts (+2.6 to +3.4)
- `transformer`: +38.0 pts (+36.8 to +39.2)
- `keel+embed`: +19.8 pts (+19.2 to +20.4)

## New held-out set: direct

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `transformer` | 60.1% (58.6 to 61.7) | 34.3% (32.8 to 35.8) | 4.7% (4.2 to 5.2) |
| `keel+embed` | 92.0% (91.5 to 92.6) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+transformer` | 100.0% (100.0 to 100.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+transformer(tuned)` | 100.0% (100.0 to 100.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+transformer(tuned)` minus:

- `keel+transformer`: +0.0 pts (+0.0 to +0.0)
- `transformer`: +39.9 pts (+38.3 to +41.4)
- `keel+embed`: +8.0 pts (+7.4 to +8.5)

## New held-out set: indirect

| Arm | Clean hit | Stale shown | Allergy shown |
| --- | --- | --- | --- |
| `transformer` | 38.6% (37.2 to 40.1) | 21.0% (19.8 to 22.2) | 0.1% (0.0 to 0.3) |
| `keel+embed` | 43.2% (42.2 to 44.1) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+transformer` | 68.8% (67.9 to 69.7) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |
| `keel+transformer(tuned)` | 74.8% (73.9 to 75.7) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) |

`keel+transformer(tuned)` minus:

- `keel+transformer`: +6.0 pts (+5.2 to +6.8)
- `transformer`: +36.2 pts (+34.8 to +37.6)
- `keel+embed`: +31.6 pts (+30.5 to +32.8)
