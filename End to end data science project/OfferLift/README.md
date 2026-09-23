# OfferLift: Email Experiment Readout and Uplift Targeting

**The question:** a retailer emailed a random two-thirds of its customers. Did the email
change what they did? And if the budget only covers part of the list next time, can a
model pick the customers worth emailing better than picking at random?

**Status: work in progress.** The experiment readout is complete. The uplift models have
had one untuned run, and it is reported below as it came out.

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

**The uplift models did not beat random targeting.** On the 12,784-customer test split
(men's email vs control, outcome = visit), every model's Qini coefficient is
indistinguishable from zero:

| Ranking | Qini coefficient | 95% bootstrap CI |
| --- | --- | --- |
| Random order | -0.0010 | -0.0027 to 0.0007 |
| T-learner, logistic | 0.0000 | -0.0015 to 0.0015 |
| T-learner, gradient boosting | -0.0002 | -0.0017 to 0.0016 |
| Transformed outcome, gradient boosting | -0.0003 | -0.0018 to 0.0015 |

At a 10% budget the logistic T-learner's top decile showed a 15.1-point lift against 7.7
for random targeting. That single number has no interval yet and sits inside a ranking
that is flat overall, so it is not evidence of anything until it is tested properly.

**Current recommendation, under the decision rule in `docs/ANALYSIS_PLAN.md`:** send the
men's email, choosing recipients at random, to as many customers as the budget allows.

## Next steps

- [ ] Bootstrap intervals for the budget table, so the top-decile result can be judged.
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
| Budget targeting table | `src/offerlift/policy.py` |
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
