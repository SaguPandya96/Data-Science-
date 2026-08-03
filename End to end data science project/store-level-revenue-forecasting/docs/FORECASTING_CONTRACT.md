# Forecasting Contract

## Decision and prediction timing

The unit of prediction is one store on one calendar day. Version 1 assumes a daily planning process:

1. day `t-1` actual sales are finalized;
2. lag features are refreshed using data through `t-1`;
3. known day-`t` calendar, operating, holiday, and promotion plans are supplied;
4. the pipeline predicts day `t` sales;
5. actual day-`t` sales later become history for the next forecast.

Every rolling demand feature applies `shift(1)` before the rolling mean. The target on a row cannot influence that row's features.

## Holdout interpretation

The final configured weeks are held out chronologically. Their lag features are calculated from prior actuals, including actuals earlier in the holdout. This estimates repeated one-day-ahead operations; it does not estimate a one-shot forecast across the entire holdout horizon.

For a forecast created once and consumed for several future weeks, use one of these alternatives:

- recursive prediction, where each predicted day updates later lag features;
- direct horizon-specific models; or
- lag features frozen at the forecast origin, with a documented accuracy tradeoff.

## Feature availability

`Open`, promotion plans, and holiday fields are treated as known future inputs. Before deployment, their upstream systems need freshness checks and a fallback policy. `Customers` is explicitly excluded because realized traffic is not available at prediction time.

## Output semantics

The target is the source `Sales` field, described by the Rossmann data as turnover. The project uses “forecasted revenue” as planning shorthand, but it does not forecast profit, margin, units, or cash receipts.
