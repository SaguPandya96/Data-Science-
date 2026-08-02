"""Conservative validation and cleaning for market and headline data."""

from __future__ import annotations

import pandas as pd


def clean_market(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize daily market rows and remove structurally invalid observations."""
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"], utc=True).dt.floor("D")
    output = output.sort_values("date").drop_duplicates("date", keep="last")

    numeric_columns = ["open", "high", "low", "close", "volume"]
    output[numeric_columns] = output[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    invalid_price = (output[["open", "high", "low", "close"]] <= 0).any(axis=1)
    invalid_volume = output["volume"] < 0
    missing_required = output[numeric_columns].isna().any(axis=1)
    output["market_quality_flag"] = invalid_price | invalid_volume | missing_required
    output["missing_calendar_days_before"] = (
        output["date"].diff().dt.days.fillna(1).sub(1).clip(lower=0)
    )

    # Extreme but structurally valid returns are kept; market shocks are data.
    structurally_invalid = invalid_price | invalid_volume
    return output.loc[~structurally_invalid].reset_index(drop=True)


def clean_news(frame: pd.DataFrame, cutoff_hour: int = 23) -> pd.DataFrame:
    """Normalize, filter, deduplicate, and assign headlines to feature dates."""
    output = frame.copy()
    output["published_at"] = pd.to_datetime(
        output["published_at"], utc=True, errors="coerce"
    )
    output["headline"] = (
        output["headline"]
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    output = output.dropna(subset=["published_at", "headline"])
    output = output[output["headline"].str.len().between(8, 500)]

    relevant = output["headline"].str.contains(
        r"\b(?:bitcoin|btc)\b", case=False, regex=True, na=False
    )
    output = output[relevant].drop_duplicates(
        subset=["published_at", "headline"]
    )

    after_cutoff = (output["published_at"].dt.hour >= cutoff_hour).astype(int)
    output["feature_date"] = (
        output["published_at"] + pd.to_timedelta(after_cutoff, unit="D")
    ).dt.floor("D")
    return output.sort_values("published_at").reset_index(drop=True)
