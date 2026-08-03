# Amazon Review Intelligence & Recommender System

[Back to the project index](../../README.md)

An end-to-end data science project on the [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) dataset (McAuley Lab, UCSD): 701,528 reviews across 112,565 beauty products.

It covers the full lifecycle: ingestion → EDA → feature engineering → NLP → recommender systems → explainability → a served API and demo UI, with tests and CI.

---

## The problem

A beauty catalogue with over 112,000 products is far more than any shopper will browse. Most customers arrive without a specific item in mind, and whether they buy anything depends on whether the store can put the right handful of products in front of them. That makes product discovery a revenue problem, not a cosmetic one.

Customer reviews are the most obvious raw material for solving it. They are abundant, they carry explicit 1-5 star preferences, and they cost nothing to collect. The natural plan is to learn what each customer likes from their review history and personalize what they see.

**This project asks whether that plan actually works, and answers three questions a team would need settled before committing engineering time to it:**

| Question | Why it matters commercially |
|---|---|
| Can we personalize recommendations from review history? | Determines whether to build a personalization system or keep serving the same best-sellers to everyone |
| Can we predict how a customer will rate a product they have not bought? | Would let us surface products a customer is likely to love, and flag ones likely to disappoint before they buy |
| Can we automatically detect unhappy customers from what they write? | Review volume is far beyond manual reading; catching dissatisfaction early is what enables intervention |

**The answers, briefly:** no, no, and yes. Two of the three are negative results, and each is backed by a baseline comparison rather than a headline metric taken at face value. The recommendation is therefore to *stop* investing in review-based personalization and redirect that effort toward collecting denser behavioural data, which is a more useful conclusion than a model that looks good in a notebook and disappoints in production.

**The most interesting results in this project are the negative ones.** Two models that looked good by their headline metric turned out, under proper baselines and ablation, not to do what their metric implied. Finding that is the work.

---

## Headline findings

**1. A popularity baseline beat the personalized recommender - and then the baseline turned out to be mostly an evaluation artifact.**

Matrix factorization scored Hit Rate@10 of 0.131. Recommending best-sellers to everyone scored 0.237. But the standard evaluation protocol samples negative items *uniformly*, which quietly flatters popularity: the held-out item is something a real person bought, while uniform negatives are mostly long-tail items nobody buys.

Re-running with popularity-weighted negatives, so every candidate is plausibly purchasable, collapsed the gap:

| Ranker | HR@10 (uniform negatives) | HR@10 (popularity-weighted negatives) |
|---|---|---|
| Random guessing | 0.105 | 0.105 |
| **Popularity** | **0.237** | **0.110** |
| Matrix factorization | 0.131 | 0.086 |
| Hybrid (MF + popularity) | 0.214 | 0.103 |

Under the honest protocol, popularity barely beats random and matrix factorization does worse than random. **No model here learned meaningful personalization.** Reporting only the uniform-sampling number, which is the common default, would have published a 0.237 that mostly measured how the negatives were drawn.

**2. The rating model's accuracy came almost entirely from the review text - which only exists after the customer has already decided their rating.**

| Feature set | Test RMSE | vs. baseline |
|---|---|---|
| Predict the global mean (baseline) | 1.309 | - |
| **Pre-review features only** (user/item history, verified purchase) | **1.264** | **3.4%** |
| Review text only (sentiment score) | 0.839 | 35.9% |
| All features | 0.824 | 37.0% |

The 37% improvement is real, but it does not mean we can predict how a customer *will* rate a product. Strip out the review text and the model is barely better than guessing the mean. So this is a **rating-inference** model (infer a star rating from written feedback), not a preference-prediction model. That is still useful for imputing ratings on feedback that arrives without stars, or flagging rating/text mismatch as a review-manipulation signal. But it is a different product, and shipping it under the wrong description would have been wrong.

**3. What does work: content-based item similarity.** It compares product text rather than user history, so data sparsity doesn't hurt it and it handles brand-new inventory on day one.

```
Seed: Oral-B Professional Care Deluxe Electric Toothbrush
  0.761  Oral-B Smartseries 4000 Professional Care Rechargeable Electric Toothbrush
  0.723  Oral-B Professional Care 1000 Power Toothbrush
  0.659  Oral-B Vitality Sonic Rechargeable Electric Toothbrush
```

**4. Sentiment classification works - but its headline accuracy is another trap.** The model scores 0.81 accuracy; always predicting "positive" scores **0.80**. Judged on accuracy it looks worthless.

The real result is on the minority classes, where the baseline is useless: it never flags an unhappy customer at all.

| Metric | Always-predict-positive | Model |
|---|---|---|
| Accuracy | 0.801 | 0.81 |
| Macro F1 | 0.297 | 0.63 |
| Recall on negative reviews | 0.00 | 0.68 |

For the business use case (surfacing dissatisfied customers), that last row is the entire value. The weak class is neutral (3-star), which is genuinely ambiguous rather than a modeling failure: those reviews mix praise and complaint, and the confusion is with adjacent classes rather than sign errors.

---

## Why personalization failed here: the root cause

The EDA answered this before any model was trained.

- **92.3% of users wrote exactly one review** (583,553 of 631,986).
- Mean reviews per user: **1.11**.
- User-item matrix density: **0.00099%**.

Collaborative filtering learns from co-occurrence: it needs users who rated several items so it can find patterns across them. With one interaction per user there is nothing to learn from, and the latent factors stay near their random initialization.

Segmenting the evaluation by user activity confirmed it isn't only a cold-start problem - popularity won in **every** segment, including users with 5+ interactions (HR@10 0.184 vs 0.132 for MF):

| Segment | Users | Popularity | Matrix factorization |
|---|---|---|---|
| Cold start (0 interactions) | 207 | 0.198 | 0.077 |
| 1 interaction | 12,099 | 0.233 | 0.139 |
| 2–4 interactions | 2,194 | 0.226 | 0.132 |
| 5+ interactions | 553 | 0.184 | 0.132 |

**The fix is more data, not a better model.** No algorithm recovers a signal that isn't there. Real recommender systems train on clicks, cart-adds, and purchases, which are dense implicit feedback, rather than on reviews. Reviews are written by a small, self-selected minority of buyers, and note that a review-only dataset cannot even measure what that fraction is: customers who never write one leave no trace in it.

---

## What this means for the product

Mapping each finding to the decision it drives:

| Finding | Recommendation |
|---|---|
| Personalization from reviews does not beat best-sellers | Do not build review-based personalization for this category. Serve popularity on discovery surfaces, and invest the engineering time in capturing behavioural signals instead |
| Rating prediction only works once the review exists | Do not promise "products you'll love" from this model. Redeploy it as rating inference on unstarred feedback, and as a rating/text mismatch detector |
| Negative-review detection works (68% recall vs 0%) | Ship it. Route flagged reviews to customer support for follow-up, which is the one place here with clear near-term value |
| Content-based similarity works and needs no user history | Ship it for "similar products" surfaces, including day-one coverage for new inventory |

The commercially useful output is the negative result. Knowing that review data cannot support personalization in this category prevents a team from spending a quarter building a system that would have lost to a popularity ranker in production.

## What I'd do next

1. **Change the input, not the algorithm.** Use implicit feedback (views, cart-adds, purchases) instead of reviews, and re-test collaborative filtering on a denser category (Electronics, Books) where it can get a fair trial. Only a fraction of buyers write a review; essentially all of them browse and click, so behavioural logs cover far more customers and give far more events per customer.
2. **Ship what works now** - content-based similarity for "similar products", popularity for cold-start home surfaces, negative-review flagging for support.
3. **Fix the offline metric before trusting any future model.** Standardize on popularity-weighted negatives, then validate online with an A/B test. Offline ranking metrics are a proxy for customer behaviour, and this project is a case study in how far a proxy can drift.

---

## Architecture

```
data/raw/          Amazon Reviews 2023 All_Beauty (JSONL.gz, downloaded)
   │
   ├── src/data/download.py       fetch reviews + product metadata
   ├── src/data/preprocess.py     clean, dedupe, filter, time-based leave-one-out split
   ├── src/features/              engineered features (leakage-safe historical aggregates)
   │
   ├── src/models/sentiment.py         TF-IDF + Logistic Regression
   ├── src/models/rating_predictor.py  HistGradientBoosting + exact SHAP
   ├── src/models/recommender.py       SGD matrix factorization + TF-IDF content-based
   ├── src/models/evaluate.py          baselines, both sampling protocols, segmented eval
   │
   ├── src/train.py               pipeline entry point, writes models/metrics.json
   │
   ├── src/api/main.py            FastAPI serving layer
   └── src/app/streamlit_app.py   demo UI
```

**Modeling decisions worth noting:**

- **Time-based leave-one-out split.** Each user's most recent review is held out. Review volume grows sharply over time, so a random split would let the model train on future reviews to predict past ones.
- **Leakage-safe features.** User/item average ratings are computed as expanding means over *prior* rows only. A user's first review falls back to the global mean; using their own mean there would leak the current row's rating into its own feature. There's a regression test for this.
- **Matrix factorization written from scratch** (numpy SGD, ~60 lines) rather than pulled from a library: no compiled dependencies, and the prediction decomposes into `global_mean + user_bias + item_bias + dot(user_vec, item_vec)`, which makes it inspectable.
- **Exact SHAP.** `TreeExplainer` with no background dataset uses the tree-path-dependent algorithm, where contributions reconcile with the prediction to floating-point precision. Supplying a background sample switches SHAP to an approximate interventional method that fails its own additivity check on this model; supplying the explained row as its own background silently returns all-zero contributions. A test asserts additivity holds.

---

## API

```bash
uvicorn src.api.main:app --reload
```

Interactive docs at `http://127.0.0.1:8000/docs`.

| Endpoint | Purpose |
|---|---|
| `GET /recommend/{user_id}` | Top-N recommendations; reports which strategy served them (collaborative / content / popularity) and whether the user is cold-start |
| `GET /similar/{parent_asin}` | Content-based similar products |
| `POST /sentiment` | Sentiment label + probabilities for review text |
| `POST /predict_rating` | Rating inference with per-feature SHAP contributions |
| `GET /health` | Liveness + which models are loaded |

```bash
curl -X POST http://127.0.0.1:8000/sentiment -H "Content-Type: application/json" -d '{"text":"Broke after two days, total waste of money"}'
```

```json
{"label": "negative", "score": -0.9976,
 "probabilities": {"negative": 0.9977, "neutral": 0.0022, "positive": 0.0001}}
```

The `/predict_rating` response ranks feature contributions by absolute SHAP value, which is how finding #2 surfaced - `sentiment_score` contributes ~10x more than every other feature combined.

---

## Setup

Python 3.11.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate     # macOS / Linux
```

```bash
.venv\Scripts\activate        # Windows
```

```bash
pip install -r requirements.txt
```

Download the data (about 134 MB), build the processed splits, then train everything:

```bash
python -m src.data.download && python -m src.data.preprocess && python -m src.train
```

`src.train` runs the whole pipeline - features, both models, the recommender, the ablation, the baseline comparison, and the segmented evaluation - and writes every metric in this README to `models/metrics.json`.

```bash
streamlit run src/app/streamlit_app.py
```

The demo UI has four tabs: recommendations (try an unknown user ID to see the cold-start path), similar products, sentiment, and SHAP-explained rating inference. It requires the API to be running.

---

## Tests

```bash
pytest tests -v
```

37 tests covering the split logic, leakage-safety of engineered features, matrix factorization behaviour, evaluation-metric correctness, and all API endpoints. API tests self-skip when model artifacts aren't present, so CI passes on a fresh clone without the dataset. Linting is `ruff`; CI runs both on every push.

Two of these tests were written before the bugs they caught: the leakage test caught user averages including their own row, and the SHAP additivity test caught an explainer returning all-zero contributions while still serving a plausible-looking prediction.

---

## Dataset

Amazon Reviews 2023, `All_Beauty` category. After filtering to users with ≥2 reviews and products with ≥3 reviews: **38,945 reviews, 15,053 users, 6,026 products**, spanning 2002–2023.

The filter is deliberately loose. Requiring 5+ reviews per user cut the data to 417 users - the long tail *is* the dataset, and modeling only its dense core would have meant modeling a population that doesn't exist.
