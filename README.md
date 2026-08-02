# Data Science

End-to-end data science projects, plus the study material behind them.

Every project starts from a real question, and every model is measured against a simple alternative to check it is actually adding value. Where the simple alternative wins, that is what gets published.

---

## The projects

### SupplyLens: Supplier Delivery Risk

**The question:** a team receives far more shipments than it can possibly check. Which ones should it look at first?

**The answer: this one worked.** The model scores every shipment by how likely it is to arrive more than a week late. Checking only the riskiest **1 in 5** shipments caught **56** of the badly delayed ones, roughly **twice as many** as checking the same number at random. The output is a ranked review queue an operations team can work through.

Two decisions matter as much as the result. A more complex model was tested and **rejected** for not beating the simpler one by enough to justify itself. A second model predicting delivery times was **thrown out entirely** once it proved worse than just reading the supplier's promised date.

*Built with:* Python, scikit-learn, calibrated probabilities, reproducible pipeline, monitoring.
*Technical detail:* ROC-AUC 0.696, PR-AUC 0.166 at 10.8% prevalence; top-20% policy reviewed 296 of 1,479 shipments, captured 56 severe delays, 1.76x lift; Brier 0.096 after isotonic calibration.

[Open the project](End%20to%20end%20data%20science%20project/SupplyLens/)

---

### Amazon Review Intelligence and Recommender System

**The question:** can 700,000 product reviews tell us what each shopper likes, so we can show them products picked for them?

**The answer: no.** The personalized model put the right product in its top 10 about **13%** of the time. Showing everybody the current best-sellers, with no personalization at all, worked **24%** of the time.

The cause was the data, not the code: **92% of shoppers had written only one review each**, and one review is not enough to learn anyone's taste. Digging further, the standard way of testing these models was quietly flattering the best-seller approach too. After correcting it, that advantage also mostly vanished. The business conclusion is to stop investing in review-based personalization and collect what customers click and buy instead.

The project also flags unhappy customers from their written reviews, which does work and is worth shipping, and explains every prediction rather than leaving it a black box.

*Built with:* Python, scikit-learn, SHAP, FastAPI, Streamlit, automated tests and CI.
*Technical detail:* Hit Rate@10 0.131 for matrix factorization vs 0.237 popularity, collapsing to 0.086 and 0.110 under popularity-weighted negative sampling against a 0.105 random floor.

[Open the project](End%20to%20end%20data%20science%20project/Amazon%20Review%20Intelligence%20and%20Recommender%20System/)

---

### GitHub Open-Source Repository Recommendation System

**The question:** can a developer's public activity suggest useful projects, including smaller ones that never reach the front page?

**The answer: only partially, based on this dataset.** The simplest approach was matching repositories to programming languages a developer already uses. It performed best, while the content-based and hybrid approaches did not improve the recommendations.

It carries an honest limitation. This was measured on **5 developers**, far too small to claim it holds generally, and the project says so rather than quietly leaving it out.

*Built with:* Python, GitHub public API, ranking evaluation, explainable recommendations.
*Technical detail:* 5 developers, 796 repositories, 449 interactions; language-only baseline NDCG@10 0.136, Hit Rate@10 0.40, beating content-based and hybrid rankers.

[Open the project](End%20to%20end%20data%20science%20project/GitHub%20Open-Source%20Repository%20Recommendation%20System/)

---

### Bitcoin Direction Forecasting with News Sentiment

**The question:** does the mood of Bitcoin news help predict whether the price rises or falls tomorrow?

**The answer: no.** The model built on market data alone was already **worse than a coin flip** on data it had not seen, adding news sentiment did not fix it, and there was not enough news history to test the idea properly.

The result was published unchanged. Adjusting settings until some version looks profitable is exactly how people fool themselves with financial models.

*Built with:* Python, VADER sentiment, GDELT news data, time-aware validation, backtesting.
*Technical detail:* 1,642 daily BTC-USD observations (2022-01-01 to 2026-06-30), 304 GDELT headlines; market-only model below chance out of sample, combined model no better.

[Open the project](End%20to%20end%20data%20science%20project/crypto-alternative-data-forecasting/)

---

## How these were built

- **Always compare against something simple.** Best-sellers, a coin flip, the supplier's promised date. A number means nothing on its own.
- **Test the way the real world works.** Train on the past, predict the future, never the reverse.
- **Question the test itself.** In the Amazon project the evaluation method turned out to be biased, which changed the conclusion.
- **Publish what came out.** Reporting that a sophisticated approach did not help is more useful than tuning until it looks good.

## Study material

Python · Statistics · Feature engineering · Feature selection · Machine learning
