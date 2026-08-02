# Data Dictionary

## Raw source fields

| Source column | Type after parsing | Meaning | Processing |
|---|---|---|---|
| `ID` | string | Unique source shipment-line identifier | Retained in `source_line_ids`; not modeled. |
| `Project Code` | category | Program/project code | First stable value per ASN/DN; not modeled because of project-specific granularity. |
| `PQ #` | string/status | Price-quotation identifier or process status | Excluded as a high-cardinality workflow identifier. |
| `PO / SO #` | string | Purchase- or sales-order number | Retained for traceability; excluded from modeling. |
| `ASN/DN #` | string | Advance shipment notice or delivery note | Shipment aggregation and operational identifier; excluded as a predictive feature. |
| `Country` | category | Destination country | Included as `country`. |
| `Managed By` | category | Managing organization | Included as `managed_by`. |
| `Fulfill Via` | category | Direct drop or regional distribution center | Included as `fulfill_via`. |
| `Vendor INCO Term` | category | Vendor Incoterm | Included as `vendor_inco_term`. |
| `Shipment Mode` | category | Air, air charter, truck, ocean, or not recorded | Included as `shipment_mode`. |
| `PQ First Sent to Client Date` | date/status | Quotation workflow milestone | Excluded because process statuses and missingness dominate. |
| `PO Sent to Vendor Date` | date/status | PO transmission date for direct drops; not applicable for regional-center fulfillment | Parsed as `po_sent_date`; used only to calculate scheduled lead time when available. |
| `Scheduled Delivery Date` | date | Promised delivery date | Latest line date per ASN/DN becomes `scheduled_delivery_date`; drives calendar features and temporal splits. |
| `Delivered to Client Date` | date | Actual client delivery date | Outcome only; excluded from features. |
| `Delivery Recorded Date` | date | Date delivery was recorded | Post-outcome; excluded from features. |
| `Product Group` | category | Commodity group | Included as `product_group`. |
| `Sub Classification` | category | Product subclass | Included as `sub_classification`. |
| `Vendor` | category | Supplier/vendor | Included as `supplier`. |
| `Item Description` | text | Detailed product description | Retained as `product` in outputs; excluded from modeling because of cardinality and sparse support. |
| `Molecule/Test Type` | category | Molecule or test category | Included as `molecule_test_type`. |
| `Brand` | category | Product brand | Included as `brand`. |
| `Dosage` | category | Dosage strength | Excluded because dosage form and product fields already capture more stable groupings. |
| `Dosage Form` | category | Tablet, test kit, solution, and similar form | Included as `dosage_form`. |
| `Unit of Measure (Per Pack)` | numeric | Units per pack | Not modeled separately; quantity/value/price already capture scale. |
| `Line Item Quantity` | numeric | Packs or units on a source line | Summed to `total_quantity`; log transformed for modeling. |
| `Line Item Value` | USD numeric | Source-line commodity value | Summed to `total_value_usd`; log transformed for modeling. |
| `Pack Price` | USD numeric | Price per pack | Median per ASN/DN as `pack_price_median`; log transformed. |
| `Unit Price` | USD numeric | Price per unit | Median per ASN/DN as `unit_price_median`; log transformed. |
| `Manufacturing Site` | category | Production site | Included as `manufacturing_site`. |
| `First Line Designation` | category | First-line treatment designation | Included as `first_line_designation`. |
| `Weight (Kilograms)` | numeric/status | Shipment weight or handling note | Single numeric value per ASN/DN as `weight_kg`; status text becomes missing. |
| `Freight Cost (USD)` | numeric/status | Shipment freight charge or cost-handling note | Single numeric value per ASN/DN as `freight_cost_usd`; status text becomes missing. |
| `Line Item Insurance (USD)` | numeric | Line-level insurance | Summed to `insurance_usd`. |

## Derived shipment fields

| Field | Type | Definition |
|---|---|---|
| `shipment_id` | string | ASN/DN identifier; one modeled row per value. |
| `source_line_count` | integer | Number of raw lines aggregated into the shipment. |
| `source_line_ids` | string | Pipe-delimited source IDs for traceability. |
| `scheduled_delivery_date` | date | Latest scheduled line date for complete-shipment delivery. |
| `actual_delivery_date` | date | Observed actual delivery date. |
| `po_sent_date` | date/nullable | Earliest parseable PO-sent date within the shipment. |
| `delivery_delay_days` | integer | Actual delivery minus scheduled delivery in calendar days; negative means early. |
| `severe_delay` | binary | 1 when `delivery_delay_days > 7`; otherwise 0. |
| `late_gt_0_days`, `late_gt_3_days`, `late_gt_7_days`, `late_gt_14_days` | binary | Candidate target thresholds used for prevalence analysis. |
| `scheduled_lead_time_days` | numeric/nullable | Scheduled delivery minus PO-sent date. |
| `actual_lead_time_days` | numeric/nullable | Actual delivery minus PO-sent date; outcome only. |
| `prediction_date` | date | Scheduled delivery date used as the chronological index. |

## Model features

Calendar fields (`scheduled_month`, `scheduled_quarter`, `scheduled_day_of_week`, and `peak_period`) derive only from the scheduled date. Quantity, value, price, weight, freight, and insurance fields use `log1p` transformations. Missing-value indicators are explicit for PO date, weight, and freight.

Target-derived historical supplier, destination, or product rates are implemented for testing but excluded from the final model because the dataset does not record the exact timestamp when the scheduled commitment was entered.

