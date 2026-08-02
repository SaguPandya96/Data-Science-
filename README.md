# Data Science

This repository collects my data science study material and end-to-end projects.

## End-to-end projects

Each of these starts from a decision someone would actually have to make, and every model is reported against a baseline rather than on its own. In three of the four, the baseline won. Those results are kept as they came out.

### [Amazon Review Intelligence and Recommender System](End%20to%20end%20data%20science%20project/Amazon%20Review%20Intelligence%20and%20Recommender%20System/)

**Can 700k product reviews personalize what a shopper sees?** No. Matrix factorization hit Hit Rate@10 of 0.131 against 0.237 for simply recommending best-sellers, and it lost in every user-activity segment: 92.3% of users wrote exactly one review, leaving no co-occurrence signal. Re-running the evaluation with popularity-weighted negatives then collapsed the baseline's own advantage to 0.110 against a 0.105 random floor, showing most of it was an artifact of how negatives were sampled. Also covers sentiment classification and SHAP-explained rating inference, served through FastAPI with a Streamlit demo, tests and CI.

### [SupplyLens: Supplier Delivery Risk and Operational Decision Intelligence](End%20to%20end%20data%20science%20project/SupplyLens/)

**When a team cannot investigate every shipment, which ones deserve attention first?** Ranks public-health commodity shipments by the calibrated probability of arriving more than seven days late. Reviewing the top 20% of risk covered 296 of 1,479 test shipments and caught 56 severe delays, a 1.76x lift over a volume-matched policy, with probabilities calibrated to a Brier score of 0.096. Logistic regression was kept over gradient boosting because the boost did not clear a pre-declared improvement margin, and a learned lead-time model was rejected outright for losing to the scheduled delivery date.

### [GitHub Open-Source Repository Recommendation System](End%20to%20end%20data%20science%20project/GitHub%20Open-Source%20Repository%20Recommendation%20System/)

**Can a developer's public history surface repositories that popularity alone would bury?** Across 5 developers, 796 repositories and 449 real interactions, a language-only baseline beat both the content-based and hybrid rankers, taking NDCG@10 of 0.136 and Hit Rate@10 of 0.40. Reported as measured, with the small-sample limitation stated rather than generalized into a population claim.

### [Crypto Alternative Data Forecasting](End%20to%20end%20data%20science%20project/crypto-alternative-data-forecasting/)

**Does Bitcoin-news sentiment improve a next-day direction model?** For the data available, no. The market-only model was below chance on the held-out period, adding sentiment did not rescue it, and the available GDELT history was too short to support a serious multi-year test. The result was kept rather than tuned until the conclusion changed.

## Study topics

- Python for data science
- Feature engineering for data science
- Feature selection for data science
- Statistics for data science
- Machine learning for data science

