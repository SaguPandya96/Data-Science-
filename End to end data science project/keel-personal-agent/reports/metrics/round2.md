# Round 2: embedding search

Embedder: `wordllama-l2_supercat-256`. 200 personas, k = 5. Intervals are 95% persona bootstrap.

**Pass rule (`keel+embed` beats `keel` on `holdout`): PASSED**

## Embedding weight, chosen on dev

| Weight | Dev clean hit |
| --- | --- |
| 0.25 | 93.2% |
| 0.5 (chosen) | 94.6% |
| 1.0 | 94.6% |
| 2.0 | 94.1% |

## New held-out wordings (all)

| Arm | Clean hit | Stale shown | Allergy shown | Tokens |
| --- | --- | --- | --- | --- |
| `recent` | 4.6% (3.9 to 5.2) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 84 |
| `lexical` | 18.9% (18.0 to 19.9) | 11.0% (10.2 to 11.8) | 0.0% (0.0 to 0.0) | 26 |
| `embedding` | 39.2% (38.1 to 40.4) | 20.1% (19.1 to 21.2) | 3.4% (2.9 to 3.9) | 83 |
| `keel` | 64.4% (63.9 to 64.8) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 75 |
| `keel+embed` | 69.1% (68.6 to 69.6) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 76 |
| `full` | 62.2% (60.6 to 63.8) | 37.8% (36.2 to 39.4) | 100.0% (100.0 to 100.0) | 1664 |

`keel+embed` minus:

- `keel`: +4.7 pts (+4.3 to +5.1)
- `embedding`: +29.9 pts (+28.5 to +31.2)
- `lexical`: +50.1 pts (+49.1 to +51.2)
- `recent`: +64.5 pts (+63.8 to +65.3)

## New held-out wordings: direct

| Arm | Clean hit | Stale shown | Allergy shown | Tokens |
| --- | --- | --- | --- | --- |
| `recent` | 4.6% (3.9 to 5.2) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 84 |
| `lexical` | 30.7% (29.2 to 32.2) | 16.4% (15.2 to 17.6) | 0.0% (0.0 to 0.0) | 28 |
| `embedding` | 55.6% (54.0 to 57.2) | 27.5% (26.2 to 28.8) | 4.2% (3.6 to 5.0) | 81 |
| `keel` | 90.9% (90.3 to 91.5) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 74 |
| `keel+embed` | 95.5% (95.0 to 95.9) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 75 |
| `full` | 62.2% (60.6 to 63.8) | 37.8% (36.2 to 39.4) | 100.0% (100.0 to 100.0) | 1664 |

`keel+embed` minus:

- `keel`: +4.6 pts (+4.1 to +5.0)
- `embedding`: +39.8 pts (+38.1 to +41.4)
- `lexical`: +64.8 pts (+63.3 to +66.3)
- `recent`: +90.9 pts (+90.2 to +91.6)

## New held-out wordings: indirect

| Arm | Clean hit | Stale shown | Allergy shown | Tokens |
| --- | --- | --- | --- | --- |
| `recent` | 4.6% (3.9 to 5.2) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 84 |
| `lexical` | 7.2% (6.4 to 7.9) | 5.7% (5.0 to 6.4) | 0.0% (0.0 to 0.0) | 25 |
| `embedding` | 22.8% (21.5 to 24.0) | 12.8% (11.8 to 13.8) | 2.6% (1.9 to 3.3) | 85 |
| `keel` | 37.8% (37.1 to 38.6) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 77 |
| `keel+embed` | 42.7% (41.7 to 43.7) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 78 |
| `full` | 62.2% (60.6 to 63.8) | 37.8% (36.2 to 39.4) | 100.0% (100.0 to 100.0) | 1664 |

`keel+embed` minus:

- `keel`: +4.8 pts (+4.2 to +5.6)
- `embedding`: +19.9 pts (+18.5 to +21.5)
- `lexical`: +35.5 pts (+34.4 to +36.6)
- `recent`: +38.1 pts (+37.0 to +39.2)

## Round 1 test wordings (already seen)

| Arm | Clean hit | Stale shown | Allergy shown | Tokens |
| --- | --- | --- | --- | --- |
| `recent` | 4.6% (3.9 to 5.2) | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) | 84 |
| `lexical` | 9.8% (8.8 to 10.8) | 5.0% (4.2 to 5.8) | 1.1% (0.8 to 1.4) | 29 |
| `embedding` | 27.5% (26.0 to 28.8) | 11.8% (10.7 to 12.8) | 1.7% (1.3 to 2.2) | 84 |
| `keel` | 62.6% (61.4 to 63.9) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 75 |
| `keel+embed` | 66.9% (65.6 to 68.1) | 0.0% (0.0 to 0.0) | 100.0% (100.0 to 100.0) | 77 |
| `full` | 62.2% (60.6 to 63.8) | 37.8% (36.2 to 39.4) | 100.0% (100.0 to 100.0) | 1664 |

`keel+embed` minus:

- `keel`: +4.2 pts (+3.5 to +5.0)
- `embedding`: +39.4 pts (+37.8 to +40.9)
- `lexical`: +57.1 pts (+55.7 to +58.5)
- `recent`: +62.3 pts (+61.0 to +63.7)

## Hit rate per held-out wording

| Detail | Style | Question | `keel` | `keel+embed` |
| --- | --- | --- | --- | --- |
| allergy | direct | Which allergies do I have? | 100% | 100% |
| allergy | indirect | Is there anything I should tell the waiter before ordering? | 100% | 100% |
| car | direct | What car do I own? | 100% | 100% |
| car | indirect | What model should I tell the mechanic I'm bringing in? | 24% | 31% |
| diet | direct | Do I follow any particular diet? | 100% | 100% |
| diet | indirect | Can you pick a sandwich for me at the deli? | 13% | 28% |
| dining_budget | direct | What's my monthly restaurant budget? | 100% | 100% |
| dining_budget | indirect | Can I afford another fancy dinner out this month? | 100% | 100% |
| doctor | direct | What's the name of my doctor? | 100% | 100% |
| doctor | indirect | Who should the pharmacy send my prescription questions to? | 12% | 38% |
| employer | direct | Which company do I work for now? | 100% | 100% |
| employer | indirect | Whose logo is on my work badge? | 100% | 100% |
| favorite_cuisine | direct | What's my favorite type of food? | 100% | 100% |
| favorite_cuisine | indirect | What should we order in tonight to cheer me up? | 16% | 14% |
| gym_days | direct | What days are my workouts? | 24% | 100% |
| gym_days | indirect | Which evenings should you keep clear so I can get my reps in? | 11% | 10% |
| home_city | direct | What city do I call home? | 100% | 100% |
| home_city | indirect | Which local news station should I watch for my weather? | 2% | 6% |
| job_title | direct | What is my current job? | 98% | 98% |
| job_title | indirect | What do I tell people I do when they ask at parties? | 0% | 0% |
| language_goal | direct | Which language am I learning at the moment? | 100% | 100% |
| language_goal | indirect | Which country's movies should I watch with subtitles for practice? | 100% | 100% |
| partner_name | direct | What's the name of my partner? | 100% | 100% |
| partner_name | indirect | Who am I sharing the hotel room with on our getaway? | 16% | 26% |
| pet | direct | What kind of pet do I own? | 100% | 100% |
| pet | indirect | Who is the vet appointment for? | 12% | 28% |
| race_goal | direct | Which race am I training for? | 100% | 100% |
| race_goal | indirect | What finish line am I working toward this season? | 0% | 2% |
| sibling | direct | Do I have any siblings? | 100% | 100% |
| sibling | indirect | Who else grew up in the same house as me? | 0% | 0% |
| wake_time | direct | When do I usually get up? | 32% | 29% |
| wake_time | indirect | Is a 6 am call too early for me? | 100% | 100% |
