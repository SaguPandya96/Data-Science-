# OfferLift: Email Experiment Readout and Uplift Targeting

**The question:** a retailer emailed a random two-thirds of its customers. Did the email
change what they did? And if the budget only covers part of the list next time, can a
model pick the customers worth emailing better than picking at random?

**Status: work in progress.** The experiment readout is complete, and the one positive
uplift result has been confirmed with a pre-committed test. The uplift models themselves
are still untuned, and every result is reported as it came out.

**Short on time?** The [one-page business summary](docs/BUSINESS_SUMMARY.md) gives the
decision, what it is worth, and how sure we are, without the statistics.

**[Try the live budget planner](https://offerlift.onrender.com)** (free hosting, so the first
load can take up to a minute while it wakes up). Locally, `make app` (or
`streamlit run app/app.py`) opens the same planner: set the
list size and budget to see who to email, the expected extra visits compared with random
targeting, and a downloadable email list for an uploaded customer file. It reads only the
committed results in `reports/metrics/`, so it needs no raw data or trained model.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/figures/segment_lift_dark.png">
  <img src="reports/figures/segment_lift_light.png" alt="Dot and interval chart of the men's email's effect on visit rate: +13.4 points for customers who bought both categories, against +6.9 for men's-only and +7.1 for women's-only buyers; +7.7 across all customers." width="760">
</picture>

## Results

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

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/figures/gain_curves_dark.png">
  <img src="reports/figures/gain_curves_light.png" alt="Line chart of extra visits per 1,000 customers compared with random targeting, by share of the list emailed. The logistic T-learner leads by up to 7.4 around 10% emailed; beyond about 20% all three models sit at or below random." width="760">
</picture>

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

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/figures/budget_gain_dark.png">
  <img src="reports/figures/budget_gain_light.png" alt="Dot and interval chart of each model's gain over random targeting at 10, 20, 30 and 50% budgets. Only the logistic T-learner at 10% has an interval above zero, at +7.4." width="760">
</picture>

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

**Confirmed with repeated cross-fitting.** Because the 10% result was the best of twelve
cells from a single split, it was re-tested with a design written into
`docs/ANALYSIS_PLAN.md` and `configs/config.yaml` before it was run: all 42,613 customers,
5-fold cross-fitting repeated 20 times, the logistic T-learner and the 10% budget fixed in
advance, and the pass rule set beforehand (the aggregated 95% interval must sit entirely
above zero).

| | Extra visits per 1,000 customers, vs random targeting |
| --- | --- |
| Confirmation (median of 20 cross-fits) | **+5.6** (95% CI 3.0 to 8.3), p < 0.01 |
| Range across the 20 repeats | +5.3 to +6.0; every repeat's own interval above zero |
| Original single split, for comparison | +7.4 (3.2 to 11.5) |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/figures/confirmation_dark.png">
  <img src="reports/figures/confirmation_light.png" alt="Forest plot: the original split estimate of +7.4 (3.2 to 11.5), the confirmation median of +5.6 (3.0 to 8.3), and all 20 repeated cross-fits between +5.3 and +6.0, every interval above zero." width="760">
</picture>

So the effect is real and smaller than the first split suggested, as the alternative
splits hinted. In practical terms, at a 10% budget random targeting earns about 7.7 extra
visits per 1,000 customers on the list, and model targeting earns about 13.3: roughly 1.7
times as many from the same number of emails.

The customers are the same ones that produced the original result, so this rules out a
lucky split or a lucky model fit, not something peculiar to this dataset. A fresh
experiment that emails the model's top 10% and a random 10% would settle that.

**Who the model targets: customers who bought both men's and women's merchandise.**
Taking the customers in the model's top 10% in at least half of the 20 confirmation
repeats (4,288 people), **every one of them** bought from both categories last year, and
they cover **99%** of the 4,330 customers who did. The model has in effect learned a
one-line rule. Those customers are also higher spenders (mean past-year spend $524 vs
$210), and all 4,330 both-category buyers spent at least $200.

The randomized data backs this up without using the model at all. Effect of the men's
email on the visit rate, by what the customer bought last year:

| Bought last year | Customers | Visit rate without email | Lift from the email |
| --- | --- | --- | --- |
| Both categories | 4,330 | 18.1% | **+13.4 pts** (10.9 to 16.0) |
| Men's only | 19,196 | 10.0% | +6.9 pts (6.0 to 7.9) |
| Women's only | 19,087 | 9.6% | +7.1 pts (6.1 to 8.0) |

Both-category buyers respond by 6.5 more points than everyone else (3.8 to 9.1). Their
extra conversion response is also clear, +0.9 points (0.1 to 1.7), while the extra spend
response is +$1.18 per customer with an interval that includes zero (-$0.19 to $2.54). The
women's email shows the same pattern, +7.1 vs +4.3 points, a
difference of 2.9 (0.3 to 5.4).

No other customer attribute adds anything. Spend of $200 or more and being a multichannel
customer look like signals at first, but among single-category buyers their extra lift is
+0.3 points (-1.1 to 1.8) and -0.2 points (-2.5 to 2.1): they only appeared to matter
because every both-category buyer is in those groups. Recent buyers (1-3 months) respond
slightly more, +1.3 points (-0.1 to 2.8), which is not conclusive. The full table for all
six customer attributes, and these follow-up checks, are in
`reports/metrics/targeting_profile.json`.

This analysis was exploratory and run after the confirmation, so it explains the confirmed
result rather than adding a new test. Because the rule picks almost exactly the customers
the model picks, the confirmed gain applies to it as well.

**Recommendation, under the rules in `docs/ANALYSIS_PLAN.md`:**

- **Budget of about 10% of the list or less:** send the men's email to customers who
  bought both men's and women's merchandise last year. This is the confirmed model
  policy written as a rule anyone can apply and check, with no model to maintain.
- **Larger budgets:** after those customers, choose the rest at random. No model beat
  random targeting beyond the top 10%, and the overall Qini interval still includes zero.

## Next steps

- [x] Bootstrap intervals for the budget table, so the top-decile result can be judged.
- [x] Confirm the 10% budget result: repeated cross-fitting inside the pipeline, with the
      budget and model fixed in advance.
- [x] Explain who the model targets (which customer features drive a high score).
- [ ] Test the rule prospectively: a new send comparing both-category buyers with a random
      group of the same size.
- [ ] X-learner and an uplift tree, for coverage of the standard model families.
- [ ] Repeat on `conversion` and `spend`, and on the women's email.
- [ ] Scale check on the Criteo uplift dataset (about 14M rows).
- [x] Streamlit page: pick a budget, see who gets emailed and the expected extra visits.
- [x] One-page business summary for a non-technical reader.
- [ ] Model card.

## How it is built

| Step | Code |
| --- | --- |
| Download with a pinned checksum | `scripts/download_data.py` |
| Schema and logic checks (e.g. no spend without a conversion) | `src/offerlift/data.py` |
| Sample ratio mismatch, balance, power, effects, CUPED, Holm | `src/offerlift/experiment.py` |
| T-learner and transformed-outcome uplift models | `src/offerlift/uplift.py` |
| Qini curve, Qini coefficient, bootstrap intervals | `src/offerlift/evaluation.py` |
| Budget targeting table with paired bootstrap intervals | `src/offerlift/policy.py` |
| Repeated cross-fitting confirmation | `src/offerlift/confirmation.py` |
| Targeting profile and segment effects | `src/offerlift/segments.py` |
| README charts, light and dark, drawn from `reports/metrics/` | `scripts/build_figures.py` |
| Budget planner logic and Streamlit app | `src/offerlift/planner.py`, `app/app.py` |
| Pre-committed analysis choices | `configs/config.yaml`, `docs/ANALYSIS_PLAN.md` |

The tests run on synthetic data where the true uplift is known. They check that the
methods recover it, that a skewed split fails the sample ratio check, and that outcomes
never enter the feature matrix.

## Run it

```bash
python -m pip install -e ".[dev,app]"
python scripts/download_data.py
python scripts/run_analysis.py
python scripts/build_figures.py
python -m pytest
streamlit run app/app.py
```

*Built with:* Python, pandas, SciPy, scikit-learn, matplotlib, Streamlit, pytest, GitHub Actions.
