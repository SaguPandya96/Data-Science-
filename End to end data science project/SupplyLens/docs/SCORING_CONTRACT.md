# Production Scoring Contract

## Command

```bash
python scripts/score.py --input path/to/shipments.csv --output path/to/scored_shipments.csv
```

PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\score.py --input .\path\to\shipments.csv --output .\path\to\scored_shipments.csv
```

## Unit of observation

One row must represent one ASN/DN shipment at scheduled-delivery commitment. Input rows must already be aggregated to this grain using the same rules as `supplylens.data.clean_shipments`.

## Required columns

| Column | Type | Missing policy |
|---|---|---|
| `shipment_id` | unique string | Never missing or duplicated. |
| `scheduled_delivery_date` | ISO date or parseable date | Never missing. |
| `po_sent_date` | ISO date | May be missing. |
| `country` | string | Converted to `Unknown` when missing. |
| `managed_by` | string | Converted to `Unknown`. |
| `fulfill_via` | string | Converted to `Unknown`. |
| `vendor_inco_term` | string | Converted to `Unknown`. |
| `shipment_mode` | string | Converted to `Unknown`. |
| `product_group` | string | Converted to `Unknown`. |
| `sub_classification` | string | Converted to `Unknown`. |
| `supplier` | string | Converted to `Unknown`. |
| `molecule_test_type` | string | Converted to `Unknown`. |
| `brand` | string | Converted to `Unknown`. |
| `dosage_form` | string | Converted to `Unknown`. |
| `manufacturing_site` | string | Converted to `Unknown`. |
| `first_line_designation` | string | Converted to `Unknown`. |
| `total_quantity` | non-negative number | Never missing. |
| `total_value_usd` | non-negative number | Never missing. |
| `pack_price_median` | non-negative number | May be missing; median imputation. |
| `unit_price_median` | non-negative number | May be missing; median imputation. |
| `weight_kg` | non-negative number | May be missing; imputation plus missing flag. |
| `freight_cost_usd` | non-negative number | May be missing; imputation plus missing flag. |
| `insurance_usd` | non-negative number | May be missing; median imputation. |
| `source_line_count` | non-negative integer | Never missing. |
| `scheduled_lead_time_days` | non-negative number | May be missing; imputation plus PO-date missing flag. |

## Prohibited columns

Actual delivery date, delivery recorded date, calculated delivery delay, severe-delay label, candidate late-label columns, final status, outcome, and target columns are rejected if present. This fail-closed behavior prevents post-outcome data from entering a prediction batch.

## Output

The scored CSV preserves input columns and adds:

- `predicted_severe_delay_probability`: calibrated probability in `[0, 1]`.
- `risk_rank`: stable descending integer rank, beginning at 1.
- `review_flag`: 1 for the exact top 20% of rows, using ceiling rounding; otherwise 0.
- `model_name`: serialized model name.
- `calibration_method`: serialized calibration method.

## Error behavior

The command exits nonzero with a clear message for missing files, missing columns, duplicate or missing shipment IDs, invalid dates, negative required numeric fields, prohibited outcome fields, missing model artifacts, or non-finite predictions. It writes the output only after validation and scoring succeed.

## Unseen categories

The encoder uses an infrequent-category bucket and does not fail on an unseen supplier, country, mode, product, or site. Unseen-category rates must still be monitored because technical acceptance does not establish reliable model performance.

