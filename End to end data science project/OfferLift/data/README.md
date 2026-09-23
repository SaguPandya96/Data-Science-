# Data

`python scripts/download_data.py` fetches the Hillstrom MineThatData E-Mail Analytics
challenge file (64,000 rows, 12 columns) from the mirror used by the scikit-uplift
project and checks it against a pinned SHA-256. The raw file is not committed because
the mirror carries no explicit redistribution grant.

Original source: Kevin Hillstrom, *MineThatData E-Mail Analytics and Data Mining
Challenge*, March 2008.

| Column | Timing | Description |
| --- | --- | --- |
| `recency` | before | months since last purchase |
| `history_segment` | before | binned `history` |
| `history` | before | dollars spent in the past year |
| `mens`, `womens` | before | bought men's / women's merchandise in the past year |
| `zip_code` | before | Urban, Surburban (sic), Rural |
| `newbie` | before | new customer in the past twelve months |
| `channel` | before | Phone, Web, Multichannel |
| `segment` | assignment | `Mens E-Mail`, `Womens E-Mail`, `No E-Mail` |
| `visit` | after | visited the site within two weeks |
| `conversion` | after | purchased within two weeks |
| `spend` | after | dollars spent within two weeks |

Only "before" columns may be used as model features.
