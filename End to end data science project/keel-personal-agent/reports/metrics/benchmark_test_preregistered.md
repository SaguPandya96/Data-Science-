# Memory benchmark: test templates

200 personas, 3,200 questions, 102 memories per persona on average (87 to 120), 38% of questions about a detail that changed. k = 5. Intervals are 95% persona bootstrap.

**Pass rule (clean_hit beats recent, lexical): PASSED**

## All questions

| Arm | Clean hit | Hit | Stale shown | Allergy shown | Tokens |
| --- | --- | --- | --- | --- | --- |
| `none` | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 0 |
| `recent` | 4.6% (3.9 to 5.2) | 4.6% (3.9 to 5.2) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 84 |
| `lexical` | 8.1% (7.2 to 9.0) | 8.7% (7.8 to 9.7) | 4.6% (3.8 to 5.3) | 1.1% (0.8 to 1.4) | 26 |
| `keel` | 59.7% (58.4 to 60.9) | 59.7% (58.4 to 60.9) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 75 |
| `full` | 62.2% (60.6 to 63.8) | 100.0% (100.0 to 100.0) | 37.8% (36.2 to 39.4) | 100.0% (100.0 to 100.0) | 1664 |

## Details that changed value

| Arm | Clean hit | Hit | Stale shown | Allergy shown | Tokens |
| --- | --- | --- | --- | --- | --- |
| `none` | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 0 |
| `recent` | 12.3% (10.6 to 14.0) | 12.3% (10.6 to 14.0) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 84 |
| `lexical` | 4.8% (3.6 to 6.0) | 6.3% (5.0 to 7.6) | 11.5% (9.7 to 13.3) | 1.0% (0.5 to 1.6) | 27 |
| `keel` | 69.2% (66.6 to 72.0) | 69.2% (66.6 to 72.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 75 |
| `full` | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 100.0% (100.0 to 100.0) | 100.0% (100.0 to 100.0) | 1664 |

## Keel minus baseline, clean hit (paired)

| Baseline | All questions | Changed details |
| --- | --- | --- |
| `recent` | +55.1 pts (+53.8 to +56.5) | +57.0 pts (+54.1 to +60.1) |
| `lexical` | +51.6 pts (+50.1 to +53.0) | +64.5 pts (+61.8 to +67.5) |

## Ablation (all questions)

| Arm | Clean hit | Hit | Stale shown | Allergy shown | Tokens |
| --- | --- | --- | --- | --- | --- |
| `keel` | 59.7% (58.4 to 60.9) | 59.7% (58.4 to 60.9) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 75 |
| `keel - supersession` | 40.4% (38.9 to 41.9) | 58.0% (56.7 to 59.2) | 19.5% (18.3 to 20.8) | 78.0% (72.4 to 83.2) | 74 |
| `keel - recency & importance` | 55.4% (54.2 to 56.6) | 55.4% (54.2 to 56.6) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 79 |
| `keel - synonyms` | 31.8% (30.7 to 32.8) | 31.8% (30.7 to 32.8) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 74 |
| `keel - key indexing` | 50.6% (49.3 to 52.0) | 50.6% (49.3 to 52.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 76 |
| `keel - duplicate suppression` | 57.9% (56.7 to 59.1) | 57.9% (56.7 to 59.1) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 76 |
| `keel - always-on constraints` | 59.1% (57.8 to 60.4) | 59.1% (57.8 to 60.4) | 0.0% (0.0 to 0.0) | 16.0% (12.7 to 19.6) | 77 |

## Key noise: share of updates written under a different key

| Noise | `recent` | `lexical` | `keel` | `keel` on changed details |
| --- | --- | --- | --- | --- |
| 0% | 4.6% (3.9 to 5.2) | 8.1% (7.2 to 9.0) | 59.7% (58.4 to 60.9) | 69.2% (66.6 to 72.0) |
| 25% | 4.6% (3.9 to 5.2) | 8.1% (7.2 to 9.0) | 54.2% (52.7 to 55.6) | 55.4% (52.4 to 58.5) |
| 50% | 4.6% (3.9 to 5.2) | 8.1% (7.2 to 9.0) | 49.5% (47.8 to 51.0) | 43.3% (40.3 to 46.2) |
| 100% | 4.6% (3.9 to 5.2) | 8.1% (7.2 to 9.0) | 42.4% (40.8 to 44.0) | 24.9% (22.3 to 27.5) |

## Memories allowed in the prompt (clean hit, all questions)

| k | `recent` | `lexical` | `keel` |
| --- | --- | --- | --- |
| 1 | 0.7% (0.4 to 1.0) | 5.9% (5.1 to 6.8) | 42.2% (41.0 to 43.6) |
| 3 | 2.7% (2.2 to 3.2) | 7.4% (6.5 to 8.3) | 55.0% (53.8 to 56.2) |
| 5 | 4.6% (3.9 to 5.2) | 8.1% (7.2 to 9.0) | 59.7% (58.4 to 60.9) |
| 10 | 8.3% (7.4 to 9.2) | 8.3% (7.4 to 9.2) | 64.8% (63.5 to 66.2) |
