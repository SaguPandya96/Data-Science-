import pandas as pd

from src.data_cleaning import clean_market
from src.monitoring import data_health


def test_invalid_market_rows_are_removed_and_dates_are_unique():
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "open": [1, 1, -1],
            "high": [2, 2, 2],
            "low": [0.5, 0.5, 0.5],
            "close": [1.5, 1.5, 1],
            "volume": [1, 1, 1],
        }
    )

    cleaned = clean_market(frame)

    assert len(cleaned) == 1
    assert not cleaned["date"].duplicated().any()


def test_health_check_reports_duplicate_dates():
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "close": [10.0, 11.0],
        }
    )

    result = data_health(frame, required_columns=["date", "close"])

    assert result["duplicate_dates"] == 1
    assert result["missing_columns"] == []
