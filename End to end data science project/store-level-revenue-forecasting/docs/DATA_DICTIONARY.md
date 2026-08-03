# Data Dictionary

## Daily sales table

| Field | Meaning | Forecast use |
|---|---|---|
| `Store` | store identifier | categorical store effect |
| `DayOfWeek` | source weekday index | rebuilt from `Date` for consistency |
| `Date` | observation date | split key and calendar feature source |
| `Sales` | daily turnover, used as a revenue proxy | target and lag history |
| `Customers` | observed customer count | excluded because it is unknown before the forecast |
| `Open` | planned/observed operating status | numeric feature; must be known for the forecast date |
| `Promo` | standard promotion indicator | feature and scenario lever |
| `StateHoliday` | public-holiday category or `0` | categorical feature and holiday flag |
| `SchoolHoliday` | school-holiday indicator | feature and holiday flag |

## Store table

| Field | Meaning | Forecast use |
|---|---|---|
| `StoreType` | source store-format category | categorical feature |
| `Assortment` | source assortment category | categorical feature |
| `CompetitionDistance` | distance to nearest competitor, in metres | converted to kilometres; missing values imputed inside the fitted pipeline |
| `CompetitionOpenSinceMonth`, `CompetitionOpenSinceYear` | approximate competition opening fields | retained in source but not used in version 1 |
| `Promo2` | participation in an extended promotion | combined with interval metadata |
| `Promo2SinceWeek`, `Promo2SinceYear` | approximate extended-promotion start | retained in source but not used in version 1 |
| `PromoInterval` | comma-separated active months | used to calculate `Promo2Active` |

## Engineered fields

| Field | Definition |
|---|---|
| `Year`, `Month`, `Week`, `DayOfWeek` | calendar components calculated from `Date` |
| `IsWeekend` | 1 for Saturday or Sunday |
| `CompetitionDistance_km` | `CompetitionDistance / 1000` |
| `Promo2Active` | 1 when the store participates in Promo2 and the current month is listed in `PromoInterval` |
| `PromoActive` | 1 when either `Promo` or `Promo2Active` is active |
| `HolidayFlag` | 1 for a state or school holiday |
| `Sales_Lag_7_RollingAvg` | mean of up to seven prior store observations after `shift(1)` |
| `Sales_Lag_30_RollingAvg` | mean of up to thirty prior store observations after `shift(1)` |

Cold-start lag values are zero. A production deployment may replace this policy with store-format or regional priors, provided it is fitted only on information available before the forecast.
