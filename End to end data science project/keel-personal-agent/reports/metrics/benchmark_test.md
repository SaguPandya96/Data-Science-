# Memory benchmark: test templates

200 personas, 3,200 questions, 102 memories per persona on average (87 to 120), 38% of questions about a detail that changed. k = 5. Intervals are 95% persona bootstrap.

**Pass rule (clean_hit beats recent, lexical): PASSED**

## All questions

| Arm | Clean hit | Hit | Stale shown | Allergy shown | Tokens |
| --- | --- | --- | --- | --- | --- |
| `none` | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 0 |
| `recent` | 4.6% (3.9 to 5.2) | 4.6% (3.9 to 5.2) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 84 |
| `lexical` | 9.8% (8.8 to 10.8) | 10.4% (9.4 to 11.5) | 5.0% (4.2 to 5.8) | 1.1% (0.8 to 1.4) | 29 |
| `keel` | 62.6% (61.4 to 63.9) | 62.6% (61.4 to 63.9) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 75 |
| `full` | 62.2% (60.6 to 63.8) | 100.0% (100.0 to 100.0) | 37.8% (36.2 to 39.4) | 100.0% (100.0 to 100.0) | 1664 |

## Details that changed value

| Arm | Clean hit | Hit | Stale shown | Allergy shown | Tokens |
| --- | --- | --- | --- | --- | --- |
| `none` | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 0 |
| `recent` | 12.3% (10.6 to 14.0) | 12.3% (10.6 to 14.0) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 84 |
| `lexical` | 5.5% (4.3 to 6.8) | 7.1% (5.7 to 8.5) | 12.6% (10.7 to 14.5) | 1.0% (0.5 to 1.6) | 30 |
| `keel` | 67.8% (65.2 to 70.6) | 67.8% (65.2 to 70.6) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 75 |
| `full` | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 100.0% (100.0 to 100.0) | 100.0% (100.0 to 100.0) | 1664 |

## Keel minus baseline, clean hit (paired)

| Baseline | All questions | Changed details |
| --- | --- | --- |
| `recent` | +58.1 pts (+56.7 to +59.5) | +55.5 pts (+52.7 to +58.7) |
| `lexical` | +52.8 pts (+51.3 to +54.3) | +62.3 pts (+59.6 to +65.2) |

## Ablation (all questions)

| Arm | Clean hit | Hit | Stale shown | Allergy shown | Tokens |
| --- | --- | --- | --- | --- | --- |
| `keel` | 62.6% (61.4 to 63.9) | 62.6% (61.4 to 63.9) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 75 |
| `keel - supersession` | 43.5% (42.0 to 44.9) | 60.8% (59.5 to 62.1) | 19.7% (18.4 to 20.9) | 77.9% (72.3 to 83.2) | 74 |
| `keel - recency & importance` | 58.6% (57.3 to 59.8) | 58.6% (57.3 to 59.8) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 78 |
| `keel - synonyms` | 34.7% (33.5 to 35.8) | 34.7% (33.5 to 35.8) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 75 |
| `keel - key indexing` | 51.7% (50.4 to 53.1) | 51.7% (50.4 to 53.1) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 76 |
| `keel - duplicate suppression` | 60.8% (59.6 to 62.2) | 60.8% (59.6 to 62.2) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 76 |
| `keel - always-on constraints` | 62.3% (61.0 to 63.7) | 62.3% (61.0 to 63.7) | 0.0% (0.0 to 0.0) | 15.3% (12.1 to 18.9) | 77 |

## Key noise: share of updates written under a different key

| Noise | `recent` | `lexical` | `keel` | `keel` on changed details |
| --- | --- | --- | --- | --- |
| 0% | 4.6% (3.9 to 5.2) | 9.8% (8.8 to 10.8) | 62.6% (61.4 to 63.9) | 67.8% (65.2 to 70.6) |
| 25% | 4.6% (3.9 to 5.2) | 9.8% (8.8 to 10.8) | 57.5% (56.1 to 58.9) | 55.0% (52.0 to 58.1) |
| 50% | 4.6% (3.9 to 5.2) | 9.8% (8.8 to 10.8) | 53.0% (51.3 to 54.6) | 43.5% (40.4 to 46.6) |
| 100% | 4.6% (3.9 to 5.2) | 9.8% (8.8 to 10.8) | 46.5% (44.9 to 48.1) | 26.1% (23.4 to 28.9) |

## Memories allowed in the prompt (clean hit, all questions)

| k | `recent` | `lexical` | `keel` |
| --- | --- | --- | --- |
| 1 | 0.7% (0.4 to 1.0) | 7.1% (6.2 to 7.9) | 43.7% (42.4 to 45.1) |
| 3 | 2.7% (2.2 to 3.2) | 8.7% (7.8 to 9.7) | 57.9% (56.7 to 59.2) |
| 5 | 4.6% (3.9 to 5.2) | 9.8% (8.8 to 10.8) | 62.6% (61.4 to 63.9) |
| 10 | 8.3% (7.4 to 9.2) | 10.0% (9.0 to 11.0) | 68.2% (66.9 to 69.6) |
