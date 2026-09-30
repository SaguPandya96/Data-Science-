# Round 4: transformer sentence encoders

200 personas, k = 5. Intervals are 95% persona bootstrap.

**Pass rule (`keel+transformer` beats `keel+embed` on `holdout2`): PASSED**

Chosen on dev: `all-MiniLM-L6-v2` at weight 2.0.

## Encoder and weight, chosen on dev (clean hit)

| Encoder | 0.25 | 0.5 | 1.0 | 2.0 |
| --- | --- | --- | --- | --- |
| `all-MiniLM-L6-v2` | 93.7% | 95.6% | 97.5% | 98.8% |
| `bge-small-en-v1.5` | 93.0% | 94.2% | 96.0% | 98.1% |

## Third held-out set (all)

| Arm | Clean hit | Stale shown | Allergy shown | Tokens |
| --- | --- | --- | --- | --- |
| `recent` | 4.6% (3.9 to 5.2) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 84 |
| `lexical` | 20.4% (19.6 to 21.3) | 9.8% (9.1 to 10.6) | 0.0% (0.0 to 0.0) | 28 |
| `embedding` | 35.0% (33.9 to 36.0) | 17.2% (16.3 to 18.1) | 1.6% (1.3 to 1.9) | 81 |
| `transformer` | 44.9% (43.7 to 46.1) | 25.3% (24.2 to 26.5) | 1.2% (0.9 to 1.5) | 80 |
| `keel` | 64.3% (63.8 to 64.8) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 74 |
| `keel+embed` | 66.9% (66.3 to 67.4) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 75 |
| `keel+transformer` | 74.9% (74.5 to 75.3) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 76 |

`keel+transformer` minus:

- `keel+embed`: +8.0 pts (+7.5 to +8.6)
- `transformer`: +30.0 pts (+28.8 to +31.2)
- `embedding`: +39.9 pts (+38.8 to +41.0)
- `keel`: +10.6 pts (+10.0 to +11.1)
- `lexical`: +54.5 pts (+53.6 to +55.4)
- `recent`: +70.3 pts (+69.5 to +71.1)

## Third held-out set: direct

| Arm | Clean hit | Stale shown | Allergy shown | Tokens |
| --- | --- | --- | --- | --- |
| `recent` | 4.6% (3.9 to 5.2) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 84 |
| `lexical` | 34.4% (32.9 to 35.9) | 14.8% (13.6 to 16.1) | 0.0% (0.0 to 0.0) | 28 |
| `embedding` | 54.0% (52.4 to 55.5) | 25.0% (23.6 to 26.3) | 1.3% (0.9 to 1.8) | 80 |
| `transformer` | 61.7% (60.1 to 63.3) | 36.3% (34.7 to 37.9) | 2.2% (1.8 to 2.8) | 78 |
| `keel` | 98.6% (98.2 to 98.9) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 73 |
| `keel+embed` | 98.8% (98.4 to 99.1) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 74 |
| `keel+transformer` | 100.0% (100.0 to 100.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 75 |

`keel+transformer` minus:

- `keel+embed`: +1.2 pts (+0.9 to +1.6)
- `transformer`: +38.3 pts (+36.7 to +39.9)
- `embedding`: +46.0 pts (+44.5 to +47.6)
- `keel`: +1.4 pts (+1.1 to +1.8)
- `lexical`: +65.6 pts (+64.1 to +67.1)
- `recent`: +95.4 pts (+94.8 to +96.1)

## Third held-out set: indirect

| Arm | Clean hit | Stale shown | Allergy shown | Tokens |
| --- | --- | --- | --- | --- |
| `recent` | 4.6% (3.9 to 5.2) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 84 |
| `lexical` | 6.4% (5.8 to 7.0) | 4.8% (4.2 to 5.5) | 0.0% (0.0 to 0.0) | 28 |
| `embedding` | 16.0% (14.8 to 17.1) | 9.4% (8.5 to 10.2) | 1.8% (1.4 to 2.3) | 83 |
| `transformer` | 28.1% (26.8 to 29.4) | 14.4% (13.4 to 15.4) | 0.1% (0.0 to 0.3) | 82 |
| `keel` | 30.1% (29.2 to 31.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 74 |
| `keel+embed` | 35.0% (33.9 to 36.0) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 76 |
| `keel+transformer` | 49.8% (48.9 to 50.6) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 78 |

`keel+transformer` minus:

- `keel+embed`: +14.8 pts (+13.8 to +15.9)
- `transformer`: +21.7 pts (+20.4 to +23.0)
- `embedding`: +33.8 pts (+32.5 to +35.2)
- `keel`: +19.8 pts (+18.8 to +20.7)
- `lexical`: +43.4 pts (+42.5 to +44.2)
- `recent`: +45.2 pts (+44.2 to +46.3)

## Round 2 held-out set (already seen)

| Arm | Clean hit | Stale shown | Allergy shown | Tokens |
| --- | --- | --- | --- | --- |
| `recent` | 4.6% (3.9 to 5.2) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 84 |
| `lexical` | 18.9% (18.0 to 19.9) | 11.0% (10.2 to 11.8) | 0.0% (0.0 to 0.0) | 26 |
| `embedding` | 39.2% (38.1 to 40.4) | 20.1% (19.1 to 21.2) | 3.4% (2.9 to 3.9) | 83 |
| `transformer` | 49.3% (48.0 to 50.6) | 29.8% (28.4 to 31.1) | 1.7% (1.4 to 2.0) | 80 |
| `keel` | 64.4% (63.9 to 64.8) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 75 |
| `keel+embed` | 69.1% (68.6 to 69.6) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 76 |
| `keel+transformer` | 81.7% (81.2 to 82.2) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 77 |

`keel+transformer` minus:

- `keel+embed`: +12.6 pts (+11.9 to +13.3)
- `transformer`: +32.4 pts (+31.1 to +33.8)
- `embedding`: +42.5 pts (+41.2 to +43.8)
- `keel`: +17.3 pts (+16.7 to +18.0)
- `lexical`: +62.8 pts (+61.7 to +63.9)
- `recent`: +77.1 pts (+76.3 to +78.0)

## Hit rate per held-out wording

| Detail | Style | Question | `keel+embed` | `keel+transformer` |
| --- | --- | --- | --- | --- |
| allergy | direct | Tell me my allergies. | 100% | 100% |
| allergy | indirect | What should be on the medical alert bracelet I'm ordering? | 100% | 100% |
| car | direct | Which car is mine? | 100% | 100% |
| car | indirect | What should I tell the valet to bring around? | 14% | 0% |
| diet | direct | What diet am I on? | 100% | 100% |
| diet | indirect | Which of the potluck dishes will I actually want to try? | 0% | 0% |
| dining_budget | direct | How big is my dining-out budget? | 100% | 100% |
| dining_budget | indirect | Should I say yes to a pricey tasting menu this month? | 100% | 100% |
| doctor | direct | Who's my GP? | 100% | 100% |
| doctor | indirect | Whose office should get my updated insurance card? | 0% | 2% |
| employer | direct | What's the name of the company I work at? | 80% | 100% |
| employer | indirect | Whose quarterly all-hands meeting do I attend? | 6% | 14% |
| favorite_cuisine | direct | Which cuisine do I love most? | 100% | 100% |
| favorite_cuisine | indirect | What kind of cookbook would make a good gift for me? | 10% | 9% |
| gym_days | direct | On which days do I go to the gym? | 100% | 100% |
| gym_days | indirect | Which days should my spotter expect me? | 30% | 100% |
| home_city | direct | Which city do I live in? | 100% | 100% |
| home_city | indirect | If I say 'let's grab brunch downtown', which downtown do I mean? | 42% | 88% |
| job_title | direct | What's my job title? | 100% | 100% |
| job_title | indirect | How would a coworker introduce me to a new client? | 20% | 15% |
| language_goal | direct | Which foreign language am I working on? | 100% | 100% |
| language_goal | indirect | Which app course should I open for my daily streak? | 31% | 33% |
| partner_name | direct | Who is my partner? | 100% | 100% |
| partner_name | indirect | Whose name goes next to mine on the joint lease? | 99% | 100% |
| pet | direct | Tell me about my pet. | 100% | 100% |
| pet | indirect | Who will the house sitter be looking after? | 25% | 27% |
| race_goal | direct | What race is my goal? | 100% | 100% |
| race_goal | indirect | Which bib number pickup should I put in my calendar? | 22% | 10% |
| sibling | direct | What's my brother's or sister's name? | 100% | 100% |
| sibling | indirect | Who could be my kids' aunt or uncle on my side? | 8% | 100% |
| wake_time | direct | What time does my alarm ring? | 100% | 100% |
| wake_time | indirect | Can we schedule a sunrise hike without me oversleeping? | 52% | 100% |
