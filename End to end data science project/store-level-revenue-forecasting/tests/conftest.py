from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rossmann_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=70, freq="D")
    stores = pd.DataFrame(
        {
            "Store": [1, 2, 3],
            "StoreType": ["a", "b", "d"],
            "Assortment": ["a", "c", "a"],
            "CompetitionDistance": [500.0, np.nan, 2200.0],
            "Promo2": [1, 0, 1],
            "PromoInterval": ["Jan,Apr,Jul,Oct", pd.NA, "Feb,May,Aug,Nov"],
        }
    )

    rows = []
    for date in dates:
        day = date.dayofweek + 1
        for store in stores["Store"]:
            open_flag = int(day != 7)
            promo = int(day in {1, 5} and store != 2)
            sales = int(open_flag * (3500 + 300 * store + 800 * promo + 20 * date.dayofyear))
            rows.append(
                {
                    "Store": store,
                    "DayOfWeek": day,
                    "Date": date,
                    "Sales": sales,
                    "Customers": sales // 12,
                    "Open": open_flag,
                    "Promo": promo,
                    "StateHoliday": "0",
                    "SchoolHoliday": 0,
                }
            )
    return pd.DataFrame(rows), stores
