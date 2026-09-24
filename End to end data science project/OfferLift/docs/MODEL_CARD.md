# Model card: OfferLift targeting

What the targeting model is, what it was checked against, and where I wouldn't trust it.
The full analysis is in the [README](../README.md); every number here comes from
`reports/metrics/`.

## What it is

A T-learner uplift model: two logistic regressions, one fitted on customers who got the
men's email and one on customers who got no email. A customer's score is the difference
between the two predicted visit probabilities, which estimates how much the email changes
that person's chance of visiting in the next two weeks.

In practice the model reduces to a rule. Its top 10% is almost exactly the customers who
bought both men's and women's products last year, so the recommended policy is that rule,
not the model. This card covers both.

| | |
| --- | --- |
| Code | `src/offerlift/uplift.py` (`t_learner_logistic`) |
| Outcome | Visited the site within two weeks (0/1) |
| Inputs | Months since last purchase, past-year spend, bought men's (0/1), bought women's (0/1), new customer (0/1), area type, purchase channel |
| Not used | Anything measured after the email, and `history_segment` (a binned copy of past-year spend) |
| Training data | 29,829 customers (70% of the men's email and no-email groups, stratified by group and outcome, seed 42) |
| Held-out data | 12,784 customers |

## Intended use

Deciding who gets the men's email when the budget covers **about 10% of the list or less**.
Beyond that, send to randomly chosen customers.

It is not meant for:

- **Bigger budgets.** Past the first 10 to 20% of the list, no model did better than random.
- **Other emails or offers.** It was trained on one email. The women's email shows the same
  pattern for both-category buyers, but I didn't fit or test a model for it.
- **Predicting who will visit.** Uplift is the change caused by the email, not the chance of
  visiting. Customers likely to visit anyway can score low.
- **Anything beyond marketing emails,** such as pricing, credit or eligibility decisions.

## How it performed

| Check | Result |
| --- | --- |
| Ranking the whole list (Qini coefficient, held-out 30%) | 0.0000 (95% CI -0.0015 to 0.0015), no better than random |
| Top 10% vs a random 10% (held-out 30%) | +7.4 extra visits per 1,000 customers (3.2 to 11.5) |
| Top 10% vs a random 10% (confirmation: 5-fold cross-fitting repeated 20 times, all 42,613 customers) | **+5.6** extra visits per 1,000 customers (3.0 to 8.3), positive in all 20 repeats |
| The same comparison for 20%, 30% and 50% budgets | Every interval includes zero |

The confirmation test and its pass rule were written down before it ran
([`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md)). The two gradient boosting models I compared it
with never beat random at any budget.

## Who it picks

Customers in the top 10% in at least half of the confirmation repeats: 4,288 people.

- All of them bought both men's and women's products last year, and they include 99% of
  the 4,330 customers who did.
- Their average past-year spend is $524, against $210 for everyone else. Every
  both-category buyer spent over $200.
- They bought slightly more recently (4.8 months ago vs 5.9 for everyone else) and are
  more often new customers (61% vs 49%).

In the randomized data, the men's email raised this group's visit rate by 13.4 points
(10.9 to 16.0), against about 7 for everyone else. Spend and channel add nothing once
purchase categories are accounted for.

## Limits and risks

- **One test, one retailer, 2008.** The confirmation re-used the same customers, so it
  rules out a lucky split, not something peculiar to this data. A new send comparing
  both-category buyers with a random group of the same size is the real check.
- **Only visits were modeled.** Purchases show the same pattern, but the extra spend for
  this group isn't statistically clear. The model doesn't show that it raises revenue.
- **The group can change.** If product ranges, the email or the customer base change, the
  both-category effect may not hold. Re-check it after any major change.
- **Who gets left out.** The data has no demographic fields, so I couldn't check whether the
  rule treats groups of people differently. "Bought men's and women's products" may
  partly track household makeup. For an informational email that barely matters; if the
  email carried a discount, the rule would decide who gets the discount, which is worth
  reviewing first.

## Maintenance

The model is re-fitted every time `scripts/run_analysis.py` runs, with fixed seeds, so the
numbers above can be reproduced exactly. The budget planner app doesn't use the model at
all: it uses the rule and the group effects stored in `reports/metrics/`.
