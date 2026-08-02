"""Observed historical replenishment-planning indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def replenishment_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["observed_lead_time_days"] = working["actual_lead_time_days"]
    grouped = working.groupby(["product_group", "country"], dropna=False)
    table = grouped.agg(
        shipment_count=("shipment_id", "nunique"),
        historical_shipped_quantity=("total_quantity", "sum"),
        quantity_variability=("total_quantity", "std"),
        average_lead_time_days=("observed_lead_time_days", "mean"),
        p90_lead_time_days=("observed_lead_time_days", lambda values: values.quantile(0.90)),
        lead_time_variability_days=("observed_lead_time_days", "std"),
        severe_delay_rate=("severe_delay", "mean"),
    ).reset_index()
    daily_demand_proxy = table["historical_shipped_quantity"] / 365.25
    table["historical_safety_stock_scenario_units"] = (
        1.645
        * daily_demand_proxy
        * table["lead_time_variability_days"].clip(lower=0).pow(0.5)
    )
    table["risk_aware_replenishment_trigger"] = np.select(
        [table["severe_delay_rate"] >= 0.15, table["severe_delay_rate"] >= 0.075],
        ["Elevated historical variability", "Watch historical variability"],
        default="Standard historical variability",
    )
    return table.sort_values(
        ["severe_delay_rate", "shipment_count"], ascending=[False, False]
    ).reset_index(drop=True)

