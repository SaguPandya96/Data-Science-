# Data Science

Four end-to-end data science projects, plus the study material behind them.

Every project answers a real question, and every model is measured against a simple alternative to check it is actually adding value. **In three of the four, the simple alternative won.** Those results are published as they came out.

That is the point of this repository. A model that looks impressive on its own can still be useless, and the only way to know is to compare it against something basic. Each project below shows that comparison.

---

## The projects

### 1. Amazon Review Intelligence and Recommender System

**The question:** can we use 700,000 product reviews to work out what each shopper likes, and show them products picked for them?

**The answer: no, and it is worth understanding why.** The personalized model put the right product in its top 10 about **13%** of the time. Simply showing everybody the current best-sellers, with no personalization at all, worked **24%** of the time. The simple approach was nearly twice as good.

The reason turned out to be in the data, not the code. **92% of shoppers had written only one review each.** To learn someone's taste you need to see several things they liked; with a single review there is nothing to learn from.

Then a second problem appeared. The way this kind of model is normally tested was quietly making the best-seller approach look better than it is. After fixing the test so it was fair, the best-seller advantage largely disappeared as well: **neither approach was genuinely learning anything about individual shoppers.**

The useful conclusion for a business: do not spend a quarter building personalization on review data. Collect what customers click and buy instead, which is far richer.

The project also detects unhappy customers from their written review, which does work and is worth shipping, and explains every prediction it makes so nothing is a black box.

*Built with:* Python, scikit-learn, SHAP, FastAPI, Streamlit, automated tests and CI.
*Technical detail:* Hit Rate@10 0.131 for matrix factorization vs 0.237 popularity, collapsing to 0.086 and 0.110 under popularity-weighted negative sampling against a 0.105 random floor.

[Open the project](End%20to%20end%20data%20science%20project/Amazon%20Review%20Intelligence%20and%20Recommender%20System/)

---

### 2. SupplyLens: Supplier Delivery Risk

**The question:** a team receives far more shipments than it can possibly check. Which ones should it look at first?

**The answer: this one worked.** The model scores every shipment by how likely it is to arrive more than a week late. Checking only the riskiest **1 in 5** shipments caught **56** of the badly delayed ones, roughly **twice as many** as checking the same number of shipments at random.

Two decisions in this project matter as much as the result. A more complex model was tested and **rejected**, because it did not beat the simpler one by enough to justify itself. And a second model that tried to predict delivery times was **thrown out entirely** once it turned out to be worse than just reading the supplier's own promised date.

*Built with:* Python, scikit-learn, calibrated probability estimates, reproducible data pipeline, monitoring.
*Technical detail:* ROC-AUC 0.696, PR-AUC 0.166 at 10.8% prevalence; top-20% policy reviewed 296 of 1,479 shipments, captured 56 severe delays, 1.76x lift; Brier 0.096 after isotonic calibration.

[Open the project](End%20to%20end%20data%20science%20project/SupplyLens/)

---

### 3. GitHub Open-Source Repository Recommendation System

**The question:** can a developer's public activity be used to suggest projects they would find useful, including smaller ones that never reach the front page?

**The answer: not from this data.** Several approaches were compared, and the winner was the simplest one tested: **just matching the programming language** a developer already works in. The more elaborate methods did worse.

This one comes with an honest limitation attached. It was measured on **5 developers**, which is far too small a sample to claim it holds generally. That limitation is stated in the project rather than quietly left out.

*Built with:* Python, GitHub public API, ranking evaluation, explainable recommendations.
*Technical detail:* 5 developers, 796 repositories, 449 interactions; language-only baseline NDCG@10 0.136, Hit Rate@10 0.40, beating content-based and hybrid rankers.

[Open the project](End%20to%20end%20data%20science%20project/GitHub%20Open-Source%20Repository%20Recommendation%20System/)

---

### 4. Bitcoin Direction Forecasting with News Sentiment

**The question:** does reading the mood of Bitcoin news help predict whether the price goes up or down tomorrow?

**The answer: no.** The model built on market data alone was already **worse than a coin flip** on data it had not seen. Adding news sentiment did not fix it, and there was not enough news history available to test the idea properly.

The result was published unchanged. It would have been easy to keep adjusting settings until some version looked profitable, and that is exactly how people fool themselves with financial models.

*Built with:* Python, VADER sentiment, GDELT news data, time-aware validation, backtesting.
*Technical detail:* 1,642 daily BTC-USD observations (2022-01-01 to 2026-06-30), 304 GDELT headlines; market-only model below chance out of sample, combined model no better.

[Open the project](End%20to%20end%20data%20science%20project/crypto-alternative-data-forecasting/)

---

## How these were built

The same discipline runs through all four:

- **Always compare against something simple.** Best-sellers, a coin flip, the supplier's promised date. A number means nothing on its own.
- **Test the way the real world works.** Train on the past and predict the future, never the reverse, so results are not flattered by information the model would not have had.
- **Question the test itself.** In the Amazon project the evaluation method turned out to be biased, which changed the conclusion entirely.
- **Publish the result that came out.** Three of these four say the sophisticated approach did not help. Reporting that is more useful than tuning until the answer looks good.

## Study material

Foundations I worked through alongside these projects:

- Python for data science
- Statistics for data science
- Feature engineering for data science
- Feature selection for data science
- Machine learning for data science
