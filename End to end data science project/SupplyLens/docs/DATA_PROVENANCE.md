# Data Provenance

## Identity

- **Dataset:** Supply Chain Shipment Pricing / SCMS Delivery History
- **Original publisher:** U.S. Agency for International Development
- **Original catalog:** [Data.gov — Supply Chain Shipment Pricing](https://catalog.data.gov/dataset/supply-chain-shipment-pricing-data)
- **Historical publisher identifier:** `0162a542-4f2e-4fe2-ad5d-8f6ed2344056`
- **Mirror repository:** [SanjogRam619/FedEx-Logistics-EDA](https://github.com/SanjogRam619/FedEx-Logistics-EDA)
- **Pinned mirror commit:** `27dc0c7d20267ec2627b39f2290994bcf7186f30`
- **Retrieved:** 2026-08-01
- **File:** `SCMS_Delivery_History_Dataset.csv`
- **Bytes:** 3,785,904
- **SHA-256:** `918b992dd3e8d4b64d2a727b2c4ea607603d0c58f19484e73f7b78528c6a8673`
- **Rows / columns:** 10,324 / 33

## Retrieval chain

1. The Data.gov catalog and historical dataset descriptions identify the U.S. Agency for International Development as the publisher.
2. The original publisher download was not directly available during this build.
3. A public GitHub mirror containing the same named CSV was selected as the retrieval endpoint.
4. `scripts/download_data.py` pins the mirror commit and expected SHA-256.
5. The downloader writes to `data/raw/` only after the checksum matches.
6. `scripts/validate_data.py` rechecks checksum, schema, dimensions, identifiers, dates, and numeric constraints before processing.

No other dataset is joined to these records.

## License and redistribution decision

The current mirror does not provide an explicit data license. Although the historical publisher is a U.S. government agency, this repository does not infer a redistribution grant from publisher identity alone. The raw CSV is excluded from version control, and users retrieve it with the checksum-pinned script. This conservative decision separates the MIT-licensed project code from the source data's usage status.

## Raw-data immutability

The raw file is never rewritten. Cleaning and shipment-line aggregation produce `data/processed/shipments.csv`. Generated processed data and model files are reproducible local artifacts and are ignored by version control.

## Grain

The raw unit is a shipment line, not necessarily an independent shipment:

- 10,324 unique line IDs.
- 7,030 unique ASN/DN values.
- 1,450 ASN/DNs contain more than one line.
- Vendor, country, shipment mode, fulfillment path, product group, and actual delivery date are stable within observed ASN/DNs.
- Ten ASN/DNs contain more than one scheduled line date.

The model uses one row per ASN/DN. For the ten mixed-schedule groups, the latest line schedule represents the completion commitment for the full shipment. Values and quantities are summed; weight and freight are taken as the single numeric shipment-level observation; insurance is summed across lines.

## Date coverage

- Scheduled delivery: 2006-05-02 to 2015-12-31.
- Actual delivery: 2006-05-02 to 2015-09-14.
- PO-sent date is parseable for 4,592 raw rows; `N/A - From RDC` and `Date Not Captured` are preserved as missing.

## Known source-quality findings

- Three lines have a delivery-recorded date earlier than the delivered date.
- Four lines arrived more than 365 days before their scheduled date.
- Weight is numeric on 6,372 source lines; nonnumeric entries include `Weight Captured Separately` and cross-references.
- Freight cost is numeric on 6,198 source lines; nonnumeric entries include `Freight Included in Commodity Cost`, `Invoiced Separately`, and cross-references.
- No duplicate raw rows or duplicate line IDs were found.
- No negative required quantities, values, pack prices, or unit prices were found.

These observations are retained and surfaced as warnings. Outcome rows are not deleted to improve model performance.

