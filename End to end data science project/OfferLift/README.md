# OfferLift: Email Experiment Readout and Uplift Targeting

**The question:** a retailer emailed a random two-thirds of its customers. Did the email
change what they did? And if the budget only covers part of the list next time, can a
model pick the customers worth emailing better than picking at random?

**Status: work in progress.** The experiment readout is complete. The uplift models have
had one untuned run, reported below as it came out.

## First-run results

Hillstrom MineThatData email test: 64,000 customers, three randomized arms, two-week window.
All numbers come from `reports/metrics/`, produced by `python scripts/run_analysis.py`.

**The experiment is trustworthy.** Arm sizes match the planned 1/3 split (sample ratio
mismatch p = 0.92) and every pre-treatment feature is balanced (largest standardized mean
difference 0.014, well under the 0.1 flag).

**Both emails work, and the men's email works better.**

| vs. no email | Visit rate | Conversion rate | Spend per customer |
| --- | --- | --- | --- |
| Control | 10.6% | 0.57% | $0.65 |
| Men's email | **+7.7 pts** (95% CI 7.0–8.3) | +0.68 pts (0.50–0.86) | +$0.77 (0.49–1.05) |
| Women's email | **+4.5 pts** (95% CI 3.9–5.2) | +0.31 pts (0.15–0.47) | +$0.42 (0.17–0.68) |

Both primary (visit) effects stay significant after Holm correction. The study could detect
a visit lift as small as 0.8 points, so it was well powered for the primary metric.

**CUPED did not help here, and that is a finding.** Adjusting spend for past-year spend
reduced variance by 0.04%. CUPED only works when the pre-period covariate predicts the
outcome, and a customer's past-year spend barely predicts whether they buy in a
particular two-week window.

**Overall, the uplift models did not rank customers better than random.** On the 12,784-customer test split
(men's email vs control, outcome = visit), every model's Qini coefficient is
indistinguishable from zero:

| Ranking | Qini coefficient | 95% bootstrap CI |
| --- | --- | --- |
| Random order | -0.0010 | -0.0027 to 0.0007 |
| T-learner, logistic | 0.0000 | -0.0015 to 0.0015 |
| T-learner, gradient boosting | -0.0002 | -0.0017 to 0.0016 |
| Transformed outcome, gradient boosting | -0.0003 | -0.0018 to 0.0015 |

**But at a small budget, one model does beat random targeting.** The budget table
compares each model's top-ranked customers with the same number picked at random. The
intervals come from a paired bootstrap (500 resamples of the test split), so model and
random are re-estimated on the same customers each time.

Extra visits per 1,000 customers, compared with random targeting at the same budget:

| Budget | T-learner, logistic | T-learner, gradient boosting | Transformed outcome |
| --- | --- | --- | --- |
| 10% | **+7.4** (3.2 to 11.5) | +1.9 (-2.5 to 5.7) | +2.0 (-2.9 to 5.8) |
| 20% | +2.5 (-2.7 to 7.7) | +3.8 (-1.2 to 8.8) | +2.0 (-2.9 to 7.6) |
| 30% | -1.7 (-6.9 to 3.4) | -1.7 (-7.0 to 4.3) | +0.2 (-5.4 to 6.7) |
| 50% | +0.6 (-5.4 to 6.2) | +0.4 (-5.9 to 7.1) | -1.9 (-7.0 to 4.1) |

Only one of the twelve cells has an interval above zero: the logistic T-learner at a 10%
budget, where its chosen customers visited 15.1 points more than control against 7.7
points for random. Two checks, run after seeing this result and not part of the pipeline:

- **Picking the best of twelve.** With a Bonferroni-corrected interval (alpha 0.05/12,
  2,000 resamples) it still clears zero, but only just: +7.4 (0.6 to 14.0).
- **A lucky split?** Refitting on five other 70/30 splits gave +8.2, +5.3, +2.4, +6.4
  and +3.9. All five are positive and three have intervals above zero. The effect looks
  real but probably smaller than the 7.4 seen on the original split.

So the model is good at finding a small group of highly responsive customers, and no
better than random beyond that. This is consistent with its flat Qini curve overall.

**Current recommendation, under the decision rule in `docs/ANALYSIS_PLAN.md`:** that rule
also requires the Qini interval to exclude zero, and it does not, so the recommendation is
still to send the men's email to randomly chosen customers. The 10% result is a
hypothesis for the next test, not a reason to change the rule after seeing the data.

## Next steps

- [x] Bootstrap intervals for the budget table, so the top-decile result can be judged.
- [ ] Confirm the 10% budget result: repeated cross-fitting inside the pipeline, with the
      budget and model fixed in advance.
- [ ] X-learner and an uplift tree, for coverage of the standard model families.
- [ ] Repeat on `conversion` and `spend`, and on the women's email.
- [ ] Scale check on the Criteo uplift dataset (about 14M rows).
- [ ] Streamlit page: pick a budget, see who gets emailed and the expected extra visits.
- [ ] Model card and a one-page business summary.

## How it is built

| Step | Code |
| --- | --- |
| Download with a pinned checksum | `scripts/download_data.py` |
| Schema and logic checks (e.g. no spend without a conversion) | `src/offerlift/data.py` |
| Sample ratio mismatch, balance, power, effects, CUPED, Holm | `src/offerlift/experiment.py` |
| T-learner and transformed-outcome uplift models | `src/offerlift/uplift.py` |
| Qini curve, Qini coefficient, bootstrap intervals | `src/offerlift/evaluation.py` |
| Budget targeting table with paired bootstrap intervals | `src/offerlift/policy.py` |
| Pre-committed analysis choices | `configs/config.yaml`, `docs/ANALYSIS_PLAN.md` |

The tests run on synthetic data where the true uplift is known. They check that the
methods recover it, that a skewed split fails the sample ratio check, and that outcomes
never enter the feature matrix.

## Run it

```bash
python -m pip install -e ".[dev]"
python scripts/download_data.py
python scripts/run_analysis.py
python -m pytest
```

*Built with:* Python, pandas, SciPy, scikit-learn, pytest, GitHub Actions.
