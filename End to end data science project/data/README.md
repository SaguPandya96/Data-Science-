# Data

## Selected source

- **Name:** Supply Chain Shipment Pricing / SCMS Delivery History
- **Publisher:** U.S. Agency for International Development
- **Catalog:** <https://catalog.data.gov/dataset/supply-chain-shipment-pricing-data>
- **Pinned public mirror:** <https://raw.githubusercontent.com/SanjogRam619/FedEx-Logistics-EDA/27dc0c7d20267ec2627b39f2290994bcf7186f30/SCMS_Delivery_History_Dataset.csv>
- **Usage status:** the mirror has no explicit license; raw redistribution is therefore not assumed
- **Retrieval date:** 2026-08-01
- **File:** `SCMS_Delivery_History_Dataset.csv`
- **Size:** 3,785,904 bytes
- **SHA-256:** `918b992dd3e8d4b64d2a727b2c4ea607603d0c58f19484e73f7b78528c6a8673`
- **Shape:** 10,324 rows × 33 columns
- **Raw unit:** shipment line
- **Modeled unit:** ASN/DN shipment
- **Scheduled-date coverage:** 2006-05-02 through 2015-12-31

## Redistribution decision

The raw CSV is not committed. The original asset is historically attributable to a U.S. government publisher, but the accessible mirror does not provide an explicit redistribution license. The repository takes the conservative path: users download a pinned copy and verify its checksum locally.

## Download

```bash
python scripts/download_data.py
```

PowerShell from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py
```

The downloader refuses to overwrite a mismatched local file unless `--force` is explicitly supplied. A checksum mismatch always fails.

## Raw handling

`data/raw/` is immutable input. Project code never rewrites the source file. The directory is ignored except for its sentinel file.

## Processed handling

```bash
python scripts/validate_data.py
```

This creates `data/processed/shipments.csv` after checksum, schema, date, numeric, duplicate, sequence, and category checks pass. The file is reproducible and ignored by version control. The processing step aggregates shipment lines to one ASN/DN shipment without combining unrelated entities.

## Sample data

No business records are committed under `data/sample/` because the accessible mirror's redistribution permission is not explicit. Tests use the locally downloaded public file and reproducible subsets of its rows.

