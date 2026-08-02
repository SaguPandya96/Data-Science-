# GitHub Open-Source Repository Recommendation System

[Open the fully executed notebook](github_repository_recommender.ipynb)

An end-to-end recommendation-systems portfolio project built from real public GitHub data. It profiles developers from public stars and ownership, represents repositories with TF–IDF text plus language/topic/activity/quality signals, excludes already-seen projects, and evaluates ranked recommendations against random, popularity, and language-only baselines.

## Why this project matters

Repository discovery is usually popularity-led. That helps developers find famous projects, but not necessarily projects aligned with their skills, interests, or contribution goals. This project tests whether public activity and repository metadata can retrieve held-out interests more effectively while retaining coverage, diversity, novelty, and transparent explanations.

## Executed result

The committed reduced-mode run contains **5 public developers, 796 public repositories, and 449 real interactions**. All five histories use timestamped REST stars and chronological splits, with the five most recent stars held out per user.

| Model | Precision@5 | Recall@10 | Hit Rate@10 | MAP@10 | NDCG@10 | Coverage | Diversity | Novelty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0730 | 0.9864 | 0.6423 |
| Popularity | 0.0400 | 0.0400 | 0.2000 | 0.0400 | 0.0678 | 0.0263 | 0.8946 | 0.1161 |
| **Language only** | **0.0800** | **0.0800** | **0.4000** | **0.0800** | **0.1357** | **0.0701** | 0.9300 | 0.4181 |
| Content based | 0.0400 | 0.0800 | 0.2000 | 0.0180 | 0.0476 | 0.0599 | 0.4081 | 0.6959 |
| Hybrid | 0.0400 | 0.0400 | 0.2000 | 0.0133 | 0.0339 | 0.0730 | 0.7506 | 0.5238 |

The **language-only model** achieved the highest test NDCG@10 (`0.1357`) and Hit Rate@10 (`0.40`). Content beat random but trailed both popularity and language-only NDCG@10; the tuned hybrid also did not win. Those negative results are preserved rather than hidden. They suggest primary language is the strongest signal in this small sample and that the richer rankers need more users and better metadata. These are descriptive results for five users, not population-level performance claims.

## Data

Primary data comes from the official [GitHub REST API](https://docs.github.com/en/rest): repository search, public user profiles, and timestamped public starred repositories. GitHub's public raw-content host supplies bounded README and contribution-file checks. The first unauthenticated pass reached the hourly core limit; cached retry after reset completed all five histories through the REST API. The committed modelling tables therefore contain no HTML-derived interactions.

Committed sample files in `data/sample/` make the core workflow reproducible without API calls. The collection manifest records API version, time, queries, sources, failures, row counts, and limitations. Raw API caches are intentionally ignored.

## Methodology

1. Validate and clean public developer, repository, language, and interaction tables.
2. Report missingness, duplicates, timestamps, inactivity, metadata gaps, API failures, and rate limits.
3. Build weighted developer profiles from training-only owned/starred interactions.
4. Clean descriptions, bounded README text, topics, and language tokens without aggressive stemming.
5. Fit TF–IDF variants on item metadata; select configuration on validation NDCG@10.
6. Generate eligible unseen candidates and score content, language, topic, activity, quality, and popularity separately.
7. Tune hybrid weights on validation data only.
8. Compare random, popularity, language, content, and hybrid models on untouched test interactions.
9. Generate feature-grounded explanations, error examples, cold-start design, and robustness diagnostics.

Collaborative filtering was rejected quantitatively: only five developers were observed and most repositories had a single-user interaction signal. The project therefore does not force an unreliable factorization model.

## Selected configuration

- TF–IDF: descriptions + README + topics + language, unigrams, `min_df=2`, `max_df=0.95`, sublinear TF, maximum 12,000 features.
- Validation-selected hybrid weights: content `0.45`, language `0.20`, topic `0.15`, activity `0.08`, quality `0.08`, popularity `0.04`.
- Production recommendations use the tuned hybrid for multi-objective exploration; offline model comparison truthfully identifies pure content as the best held-out model.
- Initial interaction assumptions: contributed `5`, owned `4`, forked `3`, starred `2`; equal weights are tested for robustness.

## Example recommendations

| Developer | Top repository | Feature-grounded explanation |
|---|---|---|
| `karpathy` | [`scikit-learn/scikit-learn`](https://github.com/scikit-learn/scikit-learn) | Python is strong in the public profile; topics overlap with machine learning and Python; activity is recent. |
| `gaearon` | [`bcherny/json-schema-to-typescript`](https://github.com/bcherny/json-schema-to-typescript) | TypeScript is strong in the profile; topics overlap with TypeScript; activity is recent. |
| `hadley` | [`REditorSupport/languageserver`](https://github.com/REditorSupport/languageserver) | R is strong in the profile; repository topics overlap with R; activity is recent. |

All 50 ranked examples and component scores are in [`outputs/recommendations.csv`](outputs/recommendations.csv).

## Visualizations

![Model comparison](outputs/figures/model_comparison.png)

![Popularity concentration](outputs/figures/popularity_concentration.png)

The notebook also saves developer activity, stars, forks, issues, languages, metadata availability, interaction, and long-tail charts individually under `outputs/figures/`.

## Repository structure

```text
GitHub Open-Source Repository Recommendation System/
├── github_repository_recommender.ipynb
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
├── data/
│   ├── raw/                 # ignored API cache
│   ├── processed/           # notebook-generated validated tables
│   └── sample/              # committed real public data
├── outputs/
│   ├── figures/
│   ├── recommendations.csv
│   ├── model_comparison.csv
│   ├── evaluation_metrics.csv
│   └── data_quality_summary.csv
└── models/
    ├── tfidf_vectorizer.joblib
    └── repository_feature_matrix.joblib
```

## Installation and execution

Python 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook github_repository_recommender.ipynb
```

For a clean non-interactive run:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 github_repository_recommender.ipynb
```

The notebook defaults to the committed sample and runs without a token. To scale collection, copy `.env.example` to `.env`, place a GitHub token only in the local `.env`, and use the reusable client in sections 8–9:

```text
GITHUB_TOKEN=your_token_here
```

Never commit `.env`; it is ignored. Authenticated collection should increase page/developer limits incrementally and preserve caching.

## Outputs and reproducibility

- `evaluation_metrics.csv` and `model_comparison.csv`: calculated offline ranking metrics.
- `recommendations.csv`: repository URLs, ranks, all score components, and explanations.
- `data_quality_summary.csv`: every check, impact, action, and reason.
- `models/`: fitted vectorizer and sparse repository matrix.
- `data/processed/`: normalized real-data tables regenerated by the notebook.

The committed notebook has actual outputs and passed a fresh Python 3.11 top-to-bottom execution. Randomness is seeded at `42`; activity calculations are anchored to collection time.

## Limitations and responsible use

- Five selected developers are not representative of GitHub.
- All five histories are chronological, but five users are still far too few for stable generalization.
- Public stars are noisy proxies for relevance and do not prove contribution intent.
- Repository search/history defines a sample catalog; the system does not rank every GitHub repository.
- README checks are partial, and byte-level language plus issue-label signals are unavailable in reduced mode.
- Public availability does not justify sensitive-trait inference. Users should be able to inspect, correct, and delete feedback profiles.

## Future improvements

Use an authenticated, consent-aware 50–150 developer sample and 1,000–5,000 repository catalog; fetch language-byte and issue-label signals; add reliable contribution events; bootstrap confidence intervals over multiple temporal windows; learn ranking weights from explicit feedback; and reconsider collaborative filtering only when item co-support is sufficient.

## License

Released under the [MIT License](LICENSE).
