# OfferLift: business summary

*One page for a marketing or product lead. The technical readout is in the
[project README](../README.md).*

## The decision

**When the email budget covers about 10% of the list or less, send the men's email to
customers who bought both men's and women's merchandise last year. Fill any budget
beyond that with customers chosen at random.**

That is a rule anyone can apply in a CRM filter. No model is needed to run it.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../reports/figures/segment_lift_dark.png">
  <img src="../reports/figures/segment_lift_light.png" alt="The men's email raises visit rate by 13.4 points for customers who bought both categories, against 6.9 and 7.1 for single-category buyers." width="700">
</picture>

## What we found

1. **The email works.** In a randomized test of 64,000 customers, the men's email raised
   the two-week visit rate from 10.6% to 18.3% (+7.7 points). It also raised purchases
   and spend. The women's email worked too, but less well (+4.5 points).
2. **One group responds about twice as strongly.** Customers who bought from both
   categories last year, about 10% of the list, gain +13.4 visit points from the email.
   Everyone else gains about +7.
3. **Beyond that group, targeting doesn't help.** We tested three machine learning
   models. Past the first 10 to 20% of the list, none of them picked customers better
   than choosing at random. What the best model learned turns out to be the rule above.

## What it's worth

For a list of 100,000 customers and a budget of 10,000 emails:

| Who gets the email | Extra site visits in two weeks |
| --- | --- |
| 10,000 customers chosen at random | about 770 |
| The 10,000 customers the rule picks | about 1,320 |
| **Difference** | **about +560** (95% range: +300 to +830) |

That is about **1.7 times as many extra visits from the same number of emails**.

Purchases seem to follow the same pattern. Both-category buyers' purchase rate rose by
1.5 points from the email, against 0.6 for everyone else. That comes from a follow-up
analysis rather than the main test, so treat it as supporting evidence. The extra spend
per customer is also higher, but not by a statistically clear margin.

## How sure we are

- **The test itself is sound.** The groups were the planned size and looked alike before
  the email, so the differences come from the email, not from who was in each group.
- **The targeting result was checked with a test fixed in advance.** It first appeared as
  the best of 12 comparisons on one data split, which can happen by luck. So we wrote down
  a stricter test and its pass rule before running it: 20 repeated re-splits of all 42,613
  customers. It passed in every repeat, with a smaller gain than first seen.
- **The main limit:** all of this comes from one 2008 test at one retailer. We re-checked
  the result on the same customers, which rules out a lucky split but not something
  specific to this data.

## Recommended next step

**Run a small confirmation send before rolling the rule out.** Split the next campaign's
emails evenly between both-category buyers and a random group, and hold back a small
random group from each with no email, so each group's response can be measured. If
both-category buyers again gain clearly more visits than the random group, adopt the
rule. This fits within a normal campaign's budget; only the choice of who gets the
emails changes.

## What we would not do

- **Build or maintain a targeting model for this email.** At small budgets a simple rule
  does the same job. At larger budgets no model beat random selection.
- **Use CUPED to shorten future tests of this kind.** Past-year spend barely predicts spend
  in a two-week window, so it reduced noise by only 0.04%. A different pre-test measure
  would be needed.

---

*Source: Hillstrom MineThatData email test (64,000 customers, three randomized groups,
two-week window). Every figure here comes from `reports/metrics/`, produced by
`python scripts/run_analysis.py`.*
