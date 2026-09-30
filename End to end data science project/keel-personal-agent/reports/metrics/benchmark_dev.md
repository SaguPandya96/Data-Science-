# Memory benchmark: dev templates

200 personas, 3,200 questions, 102 memories per persona on average (87 to 120), 38% of questions about a detail that changed. k = 5. Intervals are 95% persona bootstrap.

**Pass rule (clean_hit beats recent, lexical): PASSED**

## All questions

| Arm | Clean hit | Hit | Stale shown | Allergy shown | Tokens |
| --- | --- | --- | --- | --- | --- |
| `none` | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 0 |
| `recent` | 4.6% (3.9 to 5.2) | 4.6% (3.9 to 5.2) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 84 |
| `lexical` | 33.8% (32.3 to 35.3) | 39.4% (37.7 to 41.1) | 15.6% (14.4 to 16.8) | 0.0% (0.0 to 0.0) | 27 |
| `keel` | 92.5% (91.7 to 93.2) | 92.5% (91.7 to 93.2) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 74 |
| `full` | 62.2% (60.6 to 63.8) | 100.0% (100.0 to 100.0) | 37.8% (36.2 to 39.4) | 100.0% (100.0 to 100.0) | 1664 |

## Details that changed value

| Arm | Clean hit | Hit | Stale shown | Allergy shown | Tokens |
| --- | --- | --- | --- | --- | --- |
| `none` | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 0 |
| `recent` | 12.3% (10.6 to 14.0) | 12.3% (10.6 to 14.0) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 84 |
| `lexical` | 15.1% (13.0 to 17.3) | 30.0% (27.2 to 32.8) | 41.5% (38.4 to 44.5) | 0.0% (0.0 to 0.0) | 29 |
| `keel` | 91.5% (89.6 to 93.2) | 91.5% (89.6 to 93.2) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 75 |
| `full` | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 100.0% (100.0 to 100.0) | 100.0% (100.0 to 100.0) | 1664 |

## Keel minus baseline, clean hit (paired)

| Baseline | All questions | Changed details |
| --- | --- | --- |
| `recent` | +87.9 pts (+86.9 to +88.9) | +79.2 pts (+76.8 to +81.6) |
| `lexical` | +58.7 pts (+57.2 to +60.1) | +76.4 pts (+74.0 to +78.7) |

## Ablation (all questions)

| Arm | Clean hit | Hit | Stale shown | Allergy shown | Tokens |
| --- | --- | --- | --- | --- | --- |
| `keel` | 92.5% (91.7 to 93.2) | 92.5% (91.7 to 93.2) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 74 |
| `keel - supersession` | 58.7% (57.0 to 60.4) | 91.1% (90.2 to 91.9) | 33.8% (32.2 to 35.3) | 77.1% (71.2 to 82.6) | 73 |
| `keel - recency & importance` | 92.3% (91.5 to 93.0) | 92.3% (91.5 to 93.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 77 |
| `keel - synonyms` | 79.9% (78.8 to 81.1) | 79.9% (78.8 to 81.1) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 73 |
| `keel - key indexing` | 73.7% (72.4 to 75.0) | 73.7% (72.4 to 75.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 76 |
| `keel - duplicate suppression` | 91.5% (90.6 to 92.3) | 91.5% (90.6 to 92.3) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 75 |
| `keel - always-on constraints` | 92.9% (92.2 to 93.7) | 92.9% (92.2 to 93.7) | 0.0% (0.0 to 0.0) | 10.6% (8.0 to 13.5) | 77 |

## Key noise: share of updates written under a different key

| Noise | `recent` | `lexical` | `keel` | `keel` on changed details |
| --- | --- | --- | --- | --- |
| 0% | 4.6% (3.9 to 5.2) | 33.8% (32.3 to 35.3) | 92.5% (91.7 to 93.2) | 91.5% (89.6 to 93.2) |
| 25% | 4.6% (3.9 to 5.2) | 33.8% (32.3 to 35.3) | 82.3% (81.0 to 83.6) | 65.2% (62.1 to 67.9) |
| 50% | 4.6% (3.9 to 5.2) | 33.8% (32.3 to 35.3) | 73.8% (72.2 to 75.3) | 42.9% (40.1 to 45.4) |
| 100% | 4.6% (3.9 to 5.2) | 33.8% (32.3 to 35.3) | 59.4% (57.7 to 61.1) | 5.3% (4.1 to 6.5) |

## Memories allowed in the prompt (clean hit, all questions)

| k | `recent` | `lexical` | `keel` |
| --- | --- | --- | --- |
| 1 | 0.7% (0.4 to 1.0) | 31.1% (29.6 to 32.5) | 82.4% (81.2 to 83.6) |
| 3 | 2.7% (2.2 to 3.2) | 33.6% (32.2 to 35.1) | 90.6% (89.7 to 91.4) |
| 5 | 4.6% (3.9 to 5.2) | 33.8% (32.3 to 35.3) | 92.5% (91.7 to 93.2) |
| 10 | 8.3% (7.4 to 9.2) | 33.9% (32.4 to 35.4) | 93.7% (93.0 to 94.5) |
