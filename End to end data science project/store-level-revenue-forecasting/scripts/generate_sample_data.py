from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42


def build_sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a small Rossmann-shaped dataset for tests, demos, and CI."""
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2024-01-01", periods=140, freq="D")
    store_ids = range(1, 7)

    stores = pd.DataFrame(
        {
            "Store": list(store_ids),
            "StoreType": ["a", "a", "b", "c", "d", "d"],
            "Assortment": ["a", "c", "a", "c", "a", "c"],
            "CompetitionDistance": [500, 1200, 3500, np.nan, 800, 2200],
            "CompetitionOpenSinceMonth": [1, 4, 7, np.nan, 9, 3],
            "CompetitionOpenSinceYear": [2010, 2012, 2008, np.nan, 2014, 2011],
            "Promo2": [1, 0, 1, 0, 1, 1],
            "Promo2SinceWeek": [1, np.nan, 10, np.nan, 5, 12],
            "Promo2SinceYear": [2013, np.nan, 2012, np.nan, 2014, 2011],
            "PromoInterval": [
                "Jan,Apr,Jul,Oct",
                pd.NA,
                "Feb,May,Aug,Nov",
                pd.NA,
                "Mar,Jun,Sept,Dec",
                "Jan,Apr,Jul,Oct",
            ],
        }
    )

    rows: list[dict[str, object]] = []
    holiday_dates = {dates[0], dates[60], dates[120]}
    for date in dates:
        day_of_week = date.dayofweek + 1
        for store in store_ids:
            open_flag = int(day_of_week != 7 or store in {3, 6})
            promo = int(day_of_week in {1, 5} and (date.dayofyear + store) % 2 == 0)
            state_holiday = "a" if date in holiday_dates else "0"
            school_holiday = int(date.dayofyear % 45 in {0, 1, 2})
            weekly_effect = {1: 500, 2: 100, 3: 0, 4: 150, 5: 650, 6: 300, 7: -400}[day_of_week]
            seasonal_effect = 300 * np.sin(2 * np.pi * date.dayofyear / 90)
            expected = (
                4200
                + store * 420
                + weekly_effect
                + seasonal_effect
                + promo * 1600
                - school_holiday * 250
            )
            sales = int(max(0, expected + rng.normal(0, 180))) if open_flag else 0
            rows.append(
                {
                    "Store": store,
                    "DayOfWeek": day_of_week,
                    "Date": date.date().isoformat(),
                    "Sales": sales,
                    "Customers": int(round(sales / 12)) if sales else 0,
                    "Open": open_flag,
                    "Promo": promo,
                    "StateHoliday": state_holiday,
                    "SchoolHoliday": school_holiday,
                }
            )

    return pd.DataFrame(rows), stores


def build_future_plan(sales: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Create a one-day operating plan immediately after the sample history."""
    forecast_date = pd.to_datetime(sales["Date"]).max() + pd.Timedelta(days=1)
    day_of_week = forecast_date.dayofweek + 1
    return pd.DataFrame(
        {
            "Store": stores["Store"],
            "Date": forecast_date.date().isoformat(),
            "Open": [int(day_of_week != 7 or store in {3, 6}) for store in stores["Store"]],
            "Promo": [int(day_of_week in {1, 5} and store % 2 == 0) for store in stores["Store"]],
            "StateHoliday": "0",
            "SchoolHoliday": 0,
        }
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output = project_root / "data" / "sample"
    output.mkdir(parents=True, exist_ok=True)
    sales, stores = build_sample()
    future = build_future_plan(sales, stores)
    sales.to_csv(output / "train.csv", index=False)
    stores.to_csv(output / "store.csv", index=False)
    future.to_csv(output / "future.csv", index=False)
    print(
        f"Wrote {len(sales):,} sales rows, {len(stores):,} stores, "
        f"and a {len(future):,}-row future plan to {output}"
    )


if __name__ == "__main__":
    main()
