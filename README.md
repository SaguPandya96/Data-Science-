# Data Science

End-to-end data science projects, plus the study material behind them.

Every project starts from a real question, and every model is measured against a simple alternative to check it is actually adding value. Where the simple alternative wins, that is what gets published.

---

## The projects

### Signals in the Noise: Advertising Traffic Investigation

**The question:** when an advertising campaign suddenly looks unusual, is there enough evidence to send it for review without treating every anomaly as fraud?

**The answer: this produced a useful review workflow, with an important boundary.** I modeled the click and conversion behavior each campaign was expected to show based only on its earlier history, then ranked the largest unexplained changes for human review. In a 500,000-row chronological replay, the workflow reduced 5,902 held-out campaign windows to **11 evidence-rich review cases**. It recommends investigation, not automatic enforcement, because the public data does not contain real invalid-traffic labels.

The project also includes a separate controlled stress test for concentrated click bursts, low-and-slow activity, and impression floods, plus a benign popularity spike to test false alarms. The more complicated hybrid score did not beat the supervised baseline, so I did not promote it as the final model.

*Built with:* Python, SQL, SQLite, NumPy, chronological validation, reproducible pipeline, automated tests and CI.

*Technical detail:* 500,000 source rows covering the first 0.85 days; 15,131 observed campaign windows, 9,229 for training and 5,902 held out; held-out click rate 35.00% observed vs 35.60% expected, conversion-linked impression rate 4.82% observed vs 4.97% expected; 11 cases at the 99th-percentile review threshold.

[Open the project](End%20to%20end%20data%20science%20project/signals-in-the-noise-ad-traffic-investigation/)

---

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

### Adversarial Robustness of a Toxicity Classifier

**The question:** does an automated comment moderator still work when someone deliberately disguises what they wrote?

**The answer: it breaks, and the worse problem was somewhere else entirely.** Swapping eight Latin letters for Cyrillic ones that look identical on screen cut the share of toxic comments caught from **78% to 30%**. A Unicode cleanup step applied before the model fixed that completely, with no retraining and no new data.

The bigger finding came from testing the other half. Running ordinary, non-toxic comments through the same disguises showed that four of them were never evading the model at all. They push its score up on anything. Spacing out the letters made it flag **99%** of perfectly normal comments, against 1% untouched. The model is reacting to unusual formatting rather than to what a comment actually says, which means removing posts from people who have done nothing wrong. Cleanup does not help there, because nothing is being hidden.

The first version of this measurement concluded the model was fine. It used a threshold set to catch 95% of toxic comments, which sat so low that almost nothing was rejected and the number could barely move, and it scored only toxic examples so the false positive problem was invisible. Both mistakes are written up rather than quietly corrected.

*Built with:* Python, PyTorch, Hugging Face Transformers, bootstrap confidence intervals, 46 tests.

*Technical detail:* unitary/toxic-bert on 300 toxic and 300 benign civil_comments; at a 1% false positive threshold recall falls 0.780 to 0.303 under homoglyph substitution and returns to 0.780 with NFKC normalization; benign false positive rate rises 0.010 to 0.990 under character spacing, 0.960 under vowel repetition.

[Open the project](End%20to%20end%20data%20science%20project/evasion-gap/)

---

## How these were built

- **Always compare against something simple.** Best-sellers, a coin flip, the supplier's promised date. A number means nothing on its own.
- **Test the way the real world works.** Train on the past, predict the future, never the reverse.
- **Question the test itself.** In the Amazon project the evaluation method turned out to be biased, which changed the conclusion.
- **Publish what came out.** Reporting that a sophisticated approach did not help is more useful than tuning until it looks good.

## Study material

Python · Statistics · Feature engineering · Feature selection · Machine learning
