# Deployment and Monitoring

## Deployment design

SupplyLens is packaged as a batch-scoring system:

1. A scheduler or analyst exports one row per shipment using the scoring contract.
2. `scripts/score.py` validates the file and rejects prohibited outcome fields.
3. The serialized bundle applies the fitted feature preprocessing, logistic classifier, and isotonic calibrator.
4. Probabilities are ranked within the batch.
5. The exact top 20% receive `review_flag = 1`.
6. The scored file is delivered to a human review workflow.

The Streamlit application reads generated artifacts for retrospective exploration. It is not a transactional workflow engine and has no external API dependency.

## Artifact versioning

Each deployable bundle should be registered with:

- Code commit and release tag.
- Dataset checksum and processed-data validation report.
- Target definition and prediction-time contract.
- Training, validation, and test boundaries.
- Feature schema and category-frequency baselines.
- Model and calibration method.
- Review capacity.
- Final metrics and segment report.

`models/supplylens_model.joblib` is reproducible and intentionally ignored by version control. A production registry should store it with immutability and access controls.

## Predeployment checks

- Verify every feature is available at the scoring event in the target system.
- Run the complete test suite and scoring-contract test.
- Score a labeled shadow batch and compare schema, missingness, category rates, score distribution, capacity, calibration, and segment recall.
- Require an operational owner to approve actions and escalation procedures.
- Document rollback to the previous model and to a non-model business rule.

## Monitoring plan

| Area | Measure | Frequency | Alert condition | Response |
|---|---|---|---|---|
| Schema | Missing or renamed required columns; prohibited outcome columns | Every batch | Any occurrence | Reject batch and notify data owner. |
| Dates | Invalid or missing scheduled dates | Every batch | Any occurrence | Reject affected batch. |
| Ranges | Negative quantity, value, or line count; non-finite score | Every batch | Any occurrence | Reject and investigate upstream mapping. |
| Nulls | Per-column missing rate | Every batch / weekly trend | More than 10 percentage points above baseline | Review upstream feed and retrain only after root cause. |
| Categories | Unseen-category rate | Every batch | More than 5% for a modeled categorical field | Review mapping and segment performance. |
| Numeric drift | PSI for quantity, value, weight, freight | Weekly or monthly | PSI ≥ 0.20 | Investigate population shift; shadow-test retraining candidate. |
| Mix drift | Supplier, destination, mode, fulfillment, product frequency distance | Weekly or monthly | Context-dependent; initial investigation at 0.20 | Review operational or portfolio change. |
| Prediction drift | Score mean, quantiles, PSI, and high-risk rate | Every batch / weekly trend | PSI ≥ 0.20 or high-risk rate changes by more than 30% relative | Verify inputs and calibration. |
| Capacity | Review-queue size | Every batch | Not equal to configured ceiling of 20% | Stop delivery and inspect ranking logic. |
| Performance | PR-AUC, top-20% precision/recall/lift, Brier, false-negative rate | Monthly or after at least 200 labels | Material deterioration against registered test range | Triage by data/label drift and consider retraining. |
| Segments | Precision, recall, false-negative rate, Brier by eligible segment | Monthly/quarterly | Persistent gap with adequate volume | Investigate representation, data quality, and feature availability. |

Configuration defaults are in `configs/config.yaml`. Alert levels are starting assumptions, not empirically optimized service-level commitments.

## Executed drift example

Train-to-test monitoring produced:

- Prediction PSI: 0.297.
- Supplier frequency distance: 0.226.
- Destination frequency distance: 0.222.
- Shipment-mode frequency distance: 0.177.
- Weight PSI: 0.114.
- Freight-cost PSI: 0.095.

These retrospective results illustrate a meaningful population shift and support the temporal evaluation design. They are not live alerts.

## Labels and delayed performance

Actual delivery labels arrive after the scheduled date. Monitoring must distinguish:

- Immediate input and score monitoring.
- Delayed performance monitoring after outcomes mature.
- Backfill rules that prevent partially observed periods from understating delay.

Performance calculation should wait until the observation window is complete and should exclude neither late labels nor difficult segments.

## Retraining criteria

Consider retraining when a drift alert persists for two monitoring periods, at least 200 mature labels exist, top-20% recall or PR-AUC deteriorates materially, calibration error rises, or the operational prediction event changes. Retraining is not automatic: the candidate must pass data validation, temporal validation, calibration, segment review, scoring-contract tests, and shadow deployment.

## Rollback

1. Stop publishing new scored files.
2. Restore the last registered bundle and configuration.
3. Re-score the affected batch after schema validation.
4. If no safe model remains, use the documented supplier-rate business rule or manual review.
5. Record the incident, root cause, affected batches, and recovery validation.

## Ownership assumptions

- Operations owns review capacity, intervention playbooks, and exception handling.
- Data engineering owns source contracts and batch delivery.
- Model owners own performance, drift analysis, retraining, release documentation, and rollback readiness.
- Supplier-management and responsible-use reviewers own appropriate interpretation and protection against automatic punitive use.

