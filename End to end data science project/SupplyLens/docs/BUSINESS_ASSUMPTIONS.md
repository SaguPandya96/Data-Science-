# Business Framing and Assumptions

## Decision problem

Supply-chain operations teams need to identify shipments likely to arrive substantially after schedule so scarce review and intervention capacity can be directed toward the highest-risk cases.

## Stakeholders

- Supply-chain operations teams managing review queues.
- Procurement and supplier-management teams investigating recurring performance patterns.
- Transportation teams reviewing modes and destinations.
- Inventory planners using historical variability indicators.
- Data science and engineering teams operating, monitoring, and retraining the pipeline.

## Decision and action

- **Prediction unit:** one ASN/DN shipment after source lines are aggregated.
- **Prediction point:** scheduled-delivery commitment, before actual delivery.
- **Model output:** calibrated probability that delivery will occur more than seven calendar days after schedule.
- **Policy:** review the top 20% of a scoring batch by stable risk rank.
- **Potential actions:** contact a supplier, verify a milestone, review transport status, or escalate a planning exception.

The project does not claim that any action was taken or adopted.

## Success measures

Modeling measures emphasize ranking under imbalance: PR-AUC, ROC-AUC, Brier score, calibration error, precision, recall, lift, and severe delays captured at fixed capacity. Accuracy is not a primary measure.

Operational measures include queue size, severe delays captured, false negatives, represented shipment value, supplier minimum-volume eligibility, and planning-variability indicators.

## Target threshold

Executed prevalence analysis found:

| Definition | Positive shipments | Prevalence |
|---|---:|---:|
| More than 0 days late | 800 | 11.38% |
| More than 3 days late | 657 | 9.35% |
| More than 7 days late | 509 | 7.24% |
| More than 14 days late | 326 | 4.64% |

More than seven days was selected before modeling because it represents a substantial miss while retaining 509 positive shipments for temporal learning and evaluation. The choice was not made to maximize model performance.

## Financial scenario assumptions

The project has no observed review cost, missed-delay exposure, or intervention outcome. The scenario calculator therefore uses configurable assumptions:

- Review cost: $50 per reviewed shipment.
- Missed-severe-delay exposure: $2,500 per captured severe delay.
- Intervention success rate: 25%.

At the selected test-period capacity, the scenario computes $14,800 review cost, $35,000 expected avoided exposure, and $20,200 expected net value, with a 10.57% break-even success rate. These are scenario outputs, not observed costs or realized savings. Sensitivity results are in `reports/tables/business_impact_sensitivity.csv`.

## Replenishment assumptions

The data does not contain on-hand inventory, demand forecasts, reorder points, service levels, stockout costs, or order-up-to quantities. `replenishment_risk_indicators.csv` summarizes historical quantity and lead-time variability. The displayed safety-stock value is a scenario proxy using historical shipped quantity and observed lead-time variability; it is not an inventory recommendation.

## Causal-readiness boundary

The diagnostic treatment is fulfillment through a regional distribution center versus direct drop. Propensity-model AUC is 0.942, 40.8% of records fall outside 0.05–0.95 overlap, and maximum absolute standardized imbalance remains 0.462 after weighting. Unobserved inventory availability, urgency, contract, and routing fields are missing. No causal effect is reported.

A credible evaluation would require treatment-decision timestamps, inventory availability, requested service level, urgency, route options, transport capacity, contract constraints, and intervention history. A randomized mode/fulfillment pilot or a defensible natural experiment would be preferred.

## Scope

In scope: retrospective delivery-risk ranking, supplier performance summaries, lead-time variability, capacity policies, reproducible scoring, monitoring examples, and assumption-based scenarios.

Out of scope: automatic supplier sanctions, price negotiation, route optimization, causal effect claims, inventory optimization, current operational deployment, and forecasts beyond the source program's population.
