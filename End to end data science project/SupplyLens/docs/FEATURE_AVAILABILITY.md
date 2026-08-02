# Feature Availability and Leakage Audit

## Intended prediction time

SupplyLens scores a shipment when its scheduled-delivery commitment is available and before actual delivery is known. The source does not record when that schedule was entered, so `scheduled_delivery_date` is a chronological proxy rather than a precise creation timestamp. This limitation drives a conservative feature policy.

## Availability table

| Original column or candidate | Cleaned feature | Business meaning | Availability | Included | Decision and leakage risk |
|---|---|---|---|---|---|
| Country | `country` | Destination | By schedule commitment | Yes | Core routing context; pre-outcome. |
| Managed By | `managed_by` | Managing organization | By schedule commitment | Yes | Pre-outcome operational context. |
| Fulfill Via | `fulfill_via` | Direct drop or regional center | By schedule commitment | Yes | Pre-outcome routing choice; interpreted associationally. |
| Vendor INCO Term | `vendor_inco_term` | Commercial delivery term | By schedule commitment | Yes | Known at order/shipping arrangement. |
| Shipment Mode | `shipment_mode` | Air, truck, ocean, or charter | Assumed selected by schedule commitment | Yes | Included with documented assumption; mode must not be updated after scoring. |
| Product Group | `product_group` | Commodity group | By order creation | Yes | Pre-outcome. |
| Sub Classification | `sub_classification` | Product subclass | By order creation | Yes | Pre-outcome. |
| Vendor | `supplier` | Supplier identity | By order creation | Yes | Pre-outcome; not used for automatic penalties. |
| Molecule/Test Type | `molecule_test_type` | Product family | By order creation | Yes | Pre-outcome. |
| Brand | `brand` | Brand | By order creation | Yes | Pre-outcome. |
| Dosage Form | `dosage_form` | Product form | By order creation | Yes | Pre-outcome. |
| Manufacturing Site | `manufacturing_site` | Production site | Assumed known by schedule commitment | Yes | Pre-outcome assumption; unknown values are handled explicitly. |
| First Line Designation | `first_line_designation` | Treatment designation | By order creation | Yes | Pre-outcome. |
| Line Item Quantity | `log_total_quantity` | Aggregated shipment quantity | By order creation | Yes | Summed within ASN/DN before modeling. |
| Line Item Value | `log_total_value_usd` | Aggregated shipment value | By order creation | Yes | Observed shipment value, not an invented cost. |
| Pack Price | `log_pack_price_median` | Median pack price | By order creation | Yes | Pre-outcome. |
| Unit Price | `log_unit_price_median` | Median unit price | By order creation | Yes | Pre-outcome. |
| Weight | `log_weight_kg`, `weight_missing` | Shipment weight | Assumed available by schedule commitment when numeric | Yes | Textual statuses remain missing; the missing flag is included. |
| Freight Cost | `log_freight_cost_usd`, `freight_missing` | Freight charge | Assumed quoted by schedule commitment when numeric | Yes | If an implementation only knows freight after dispatch, this field must be removed and the model retrained. |
| Insurance | `log_insurance_usd` | Aggregated insured amount | Assumed available by schedule commitment | Yes | Pre-outcome assumption. |
| Source line count | `source_line_count` | Shipment complexity proxy | At aggregation | Yes | Derived only from current shipment lines. |
| PO Sent to Vendor Date | `scheduled_lead_time_days`, `po_date_missing` | Planned lead time | Before schedule for direct drops | Yes | Missing for regional-center records; no future information. |
| Scheduled Delivery Date | Calendar fields | Season and commitment timing | At prediction | Yes | The date is the prediction anchor, not the outcome. |
| ASN/DN, PO/SO, raw ID | — | Unique identifiers | At prediction | No | High cardinality and memorization risk; retained only for traceability. |
| Item Description | `product` | Detailed description | At prediction | No | High cardinality; retained in operational outputs. |
| Actual delivery date | `actual_delivery_date` | Observed outcome date | After outcome | **No** | Direct target leakage. |
| Delivery recorded date | `delivery_recorded_date` | Outcome recording date | After outcome | **No** | Post-outcome leakage. |
| Calculated delay | `delivery_delay_days` | Actual minus scheduled | After outcome | **No** | Direct target source. |
| Severe-delay label | `severe_delay` | More than 7 days late | After outcome | **No** | Target only. |
| Final delivery status | — | Final outcome status | After outcome | **No** | Prohibited post-outcome feature. |
| Historical supplier delay rate | `historical_supplier_delay_rate` | Prior supplier outcomes | Unknown at true schedule-entry time | No in production | A shifted implementation exists and passes same-date leakage tests, but the schedule-entry timestamp is absent. Excluded conservatively. |
| Historical destination delay rate | `historical_country_delay_rate` | Prior destination outcomes | Unknown at true schedule-entry time | No in production | Same timestamp limitation. |
| Historical product delay rate | `historical_product_group_delay_rate` | Prior product outcomes | Unknown at true schedule-entry time | No in production | Same timestamp limitation. |

## Automated protection

`supplylens.features.LEAKAGE_BLOCKLIST` rejects actual delivery, recorded delivery, delay, target, and outcome-status fields. `prepare_model_frame` selects an allowlist rather than dropping a few known bad columns. Tests verify the blocklist, feature schema, historical shifting, and that changing the current row's target cannot alter its historical feature values.

## Deployment requirement

Before using the pipeline with another operational system, each input must be revalidated against that system's actual event timestamps. In particular, shipment mode, manufacturing site, freight cost, weight, and insurance must be known at scoring time; otherwise they must be removed and the model retrained.

