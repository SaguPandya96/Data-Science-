"""Download and cache the market and headline source data used by the project."""

from __future__ import annotations

import logging
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)


def _save_snapshot(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Write a source snapshot and record when it was collected."""
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output["ingested_at_utc"] = datetime.now(timezone.utc).isoformat()
    output.to_csv(path, index=False)
    return output


def _get_json(url: str) -> dict[str, Any]:
    """Fetch JSON with an explicit timeout and research user agent."""
    response = requests.get(
        url,
        headers={"User-Agent": "crypto-forecasting-research/1.0"},
        timeout=45,
    )
    response.raise_for_status()
    return response.json()


def download_market_data(
    ticker: str,
    start: str,
    end: str,
    path: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download daily Yahoo Finance OHLCV or return the existing snapshot."""
    if path.exists() and path.stat().st_size > 100 and not refresh:
        LOGGER.info("Using cached market data: %s", path)
        return pd.read_csv(path)

    try:
        period_start = int(pd.Timestamp(start, tz="UTC").timestamp())
        period_end = int(pd.Timestamp(end, tz="UTC").timestamp())
        encoded_ticker = urllib.parse.quote(ticker)
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{encoded_ticker}?period1={period_start}&period2={period_end}"
            "&interval=1d&events=history"
        )
        result = _get_json(url)["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]
        market = pd.DataFrame(quote)
        market["date"] = pd.to_datetime(
            result["timestamp"], unit="s", utc=True
        ).floor("D")
        market = market[["date", "open", "high", "low", "close", "volume"]]
        if market.empty:
            raise ValueError("Yahoo Finance returned no market rows")

        market["source"] = "Yahoo Finance chart endpoint"
        LOGGER.info("Downloaded %d market rows for %s", len(market), ticker)
        return _save_snapshot(market, path)
    except Exception:
        if path.exists() and path.stat().st_size > 100:
            LOGGER.exception("Market refresh failed; falling back to the cache")
            return pd.read_csv(path)
        raise


def download_gdelt_news(
    query: str,
    start: str,
    end: str,
    path: Path,
    chunk_days: int = 30,
    max_records: int = 250,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download GDELT article lists in bounded date chunks with retries."""
    if path.exists() and path.stat().st_size > 100 and not refresh:
        LOGGER.info("Using cached headline data: %s", path)
        return pd.read_csv(path)

    session = requests.Session()
    session.headers["User-Agent"] = "crypto-forecasting-research/1.0"
    cursor = pd.Timestamp(start, tz="UTC")
    stop = pd.Timestamp(end, tz="UTC")
    rows: list[dict[str, Any]] = []

    while cursor < stop:
        chunk_end = min(cursor + timedelta(days=chunk_days), stop)
        parameters = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": max_records,
            "sort": "datedesc",
            "startdatetime": cursor.strftime("%Y%m%d%H%M%S"),
            "enddatetime": chunk_end.strftime("%Y%m%d%H%M%S"),
        }
        last_error: Exception | None = None

        for attempt in range(3):
            try:
                response = session.get(
                    "https://api.gdeltproject.org/api/v2/doc/doc",
                    params=parameters,
                    timeout=45,
                )
                if response.status_code == 429:
                    time.sleep(2**attempt)
                    continue
                response.raise_for_status()
                rows.extend(response.json().get("articles", []))
                last_error = None
                break
            except Exception as error:
                last_error = error
                time.sleep(2**attempt)

        if last_error is not None:
            LOGGER.warning("GDELT chunk %s failed: %s", cursor.date(), last_error)
        cursor = chunk_end
        time.sleep(0.15)

    if not rows:
        if path.exists() and path.stat().st_size > 100:
            LOGGER.warning("GDELT returned no rows; using the cached snapshot")
            return pd.read_csv(path)
        raise ValueError("GDELT returned no articles and no cache is available")

    news = pd.DataFrame(rows).rename(
        columns={"seendate": "published_at", "title": "headline"}
    )
    available_columns = [
        column
        for column in [
            "published_at",
            "headline",
            "url",
            "domain",
            "language",
            "sourcecountry",
        ]
        if column in news
    ]
    news = news[available_columns].drop_duplicates(
        subset=["published_at", "headline", "url"]
    )
    news["source"] = "GDELT DOC 2.0"
    LOGGER.info("Downloaded %d unique GDELT headlines", len(news))
    return _save_snapshot(news, path)
