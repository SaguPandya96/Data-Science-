# Final conclusions

The held-out results do not support a useful next-day Bitcoin direction signal.

- The market-only logistic regression reached 0.467 accuracy and 0.461 ROC-AUC on 323 test days.
- The 95% moving-block bootstrap interval for accuracy was approximately 0.406 to 0.517.
- The combined model matched the market-only result; news sentiment did not add measured predictive value.
- The alternative-only model produced a constant, no-information forecast because the historical training folds contained no sentiment coverage.
- The 0.55 long/cash strategy lost 40.1% after transaction costs and slippage, with a -1.45 Sharpe ratio and 40.1% maximum drawdown.
- Walk-forward results varied across folds, which is inconsistent with a stable edge.

The main limitation is the mismatch between multi-year market history and the short rolling GDELT headline window. A credible follow-up needs point-in-time news across every training and evaluation period before more model complexity is justified.

The next study should begin with a licensed or archived headline history, a manually labelled sentiment-quality sample, cached FinBERT scores, and preregistered robustness choices. The current test set should remain closed to further tuning.
