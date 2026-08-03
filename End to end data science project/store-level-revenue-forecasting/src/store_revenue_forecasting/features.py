from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

MONTH_NAMES = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sept",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

CATEGORICAL_FEATURES = ["Store", "StoreType", "Assortment", "StateHoliday"]
TARGET_COLUMN = "Sales"


def lag_column(window: int) -> str:
    return f"Sales_Lag_{window}_RollingAvg"


def numeric_feature_columns(lag_windows: Sequence[int]) -> list[str]:
    return [
        "Open",
        "Promo",
        "SchoolHoliday",
        "Year",
        "Month",
        "Week",
        "DayOfWeek",
        "IsWeekend",
        "CompetitionDistance_km",
        "Promo2Active",
        "PromoActive",
        "HolidayFlag",
        *[lag_column(window) for window in lag_windows],
    ]


def feature_columns(lag_windows: Sequence[int]) -> list[str]:
    return numeric_feature_columns(lag_windows) + CATEGORICAL_FEATURES


def build_features(merged: pd.DataFrame, lag_windows: Sequence[int]) -> pd.DataFrame:
    """Create forecast-time features, including strictly shifted store demand history."""
    windows = [int(window) for window in lag_windows]
    if not windows or any(window <= 0 for window in windows):
        raise ValueError("lag_windows must contain positive integers.")

    frame = merged.sort_values(["Store", "Date"]).reset_index(drop=True).copy()
    frame["Year"] = frame["Date"].dt.year
    frame["Month"] = frame["Date"].dt.month
    frame["Week"] = frame["Date"].dt.isocalendar().week.astype(int)
    frame["DayOfWeek"] = frame["Date"].dt.dayofweek + 1
    frame["IsWeekend"] = frame["DayOfWeek"].isin([6, 7]).astype(int)

    frame["CompetitionDistance_km"] = frame["CompetitionDistance"] / 1000
    frame["StateHoliday"] = frame["StateHoliday"].fillna("0").astype(str)

    month_name = frame["Month"].map(MONTH_NAMES)
    promo_interval = frame["PromoInterval"].fillna("").astype(str)
    promo2 = frame["Promo2"].fillna(0).astype(int)
    interval_matches_month = [
        name in {part.strip() for part in interval.split(",") if part.strip()}
        for name, interval in zip(month_name, promo_interval, strict=True)
    ]
    frame["Promo2Active"] = (promo2.eq(1) & pd.Series(interval_matches_month)).astype(int)
    frame["PromoActive"] = (
        frame["Promo"].fillna(0).astype(int).eq(1) | frame["Promo2Active"].eq(1)
    ).astype(int)

    frame["StateHolidayFlag"] = frame["StateHoliday"].ne("0").astype(int)
    frame["HolidayFlag"] = (
        frame["StateHolidayFlag"].eq(1)
        | frame["SchoolHoliday"].fillna(0).astype(int).eq(1)
    ).astype(int)

    grouped_sales = frame.groupby("Store", sort=False)[TARGET_COLUMN]
    for window in windows:
        column = lag_column(window)
        frame[column] = grouped_sales.transform(
            lambda values, size=window: values.shift(1).rolling(size, min_periods=1).mean()
        )
        frame[column] = frame[column].fillna(0.0)

    frame["Store"] = frame["Store"].astype(str)
    return frame
