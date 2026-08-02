"""Small monitoring checks that can run after each scheduled data refresh."""

from __future__ import annotations

import pandas as pd


def data_health(
    frame: pd.DataFrame,
    required_columns: list[str],
    date_column: str = "date",
) -> dict[str, object]:
    """Summarize schema, duplication, staleness, and missingness."""
    dates = pd.to_datetime(frame[date_column], utc=True)
    missing_columns = sorted(set(required_columns) - set(frame.columns))
    missing_fraction = (
        frame[required_columns].isna().mean().to_dict()
        if not missing_columns
        else {}
    )
    latest_date = dates.max().floor("D")
    today = pd.Timestamp.now(tz="UTC").floor("D")

    return {
        "missing_columns": missing_columns,
        "duplicate_dates": int(dates.duplicated().sum()),
        "stale_days": int((today - latest_date).days),
        "missing_fraction": missing_fraction,
    }
