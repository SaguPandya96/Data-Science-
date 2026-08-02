"""Idempotent public-data ingestion for BTC market data and GDELT headlines."""
from __future__ import annotations
import logging, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

def _write_with_metadata(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["ingested_at_utc"] = datetime.now(timezone.utc).isoformat()
    out.to_csv(path, index=False)
    return out

def download_market_data(ticker: str, start: str, end: str, path: Path, refresh: bool = False) -> pd.DataFrame:
    """Download unadjusted daily Yahoo Finance OHLCV, or use the nonempty cache."""
    if path.exists() and path.stat().st_size > 100 and not refresh:
        return pd.read_csv(path)
    try:
        import urllib.parse
        p1=int(pd.Timestamp(start,tz="UTC").timestamp()); p2=int(pd.Timestamp(end,tz="UTC").timestamp())
        url=(f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}"
             f"?period1={p1}&period2={p2}&interval=1d&events=history")
        response=session_get(url); result=response["chart"]["result"][0]
        quote=result["indicators"]["quote"][0]
        data=pd.DataFrame(quote); data["date"]=pd.to_datetime(result["timestamp"],unit="s",utc=True).floor("D")
        data=data[["date","open","high","low","close","volume"]]
        if data.empty: raise ValueError("Yahoo Finance returned no rows")
        data["source"] = "Yahoo Finance via yfinance"
        return _write_with_metadata(data, path)
    except Exception:
        if path.exists() and path.stat().st_size > 100:
            LOGGER.exception("Market refresh failed; using cache")
            return pd.read_csv(path)
        raise

def session_get(url: str) -> dict[str, Any]:
    """GET JSON with a research user-agent and explicit HTTP errors."""
    response=requests.get(url,headers={"User-Agent":"Mozilla/5.0 crypto-forecasting-research"},timeout=45)
    response.raise_for_status(); return response.json()

def download_gdelt_news(query: str, start: str, end: str, path: Path, chunk_days: int = 30,
                        max_records: int = 250, refresh: bool = False) -> pd.DataFrame:
    """Download GDELT DOC 2.0 article lists in bounded chunks with retry and caching."""
    if path.exists() and path.stat().st_size > 100 and not refresh:
        return pd.read_csv(path)
    session = requests.Session(); session.headers["User-Agent"] = "crypto-forecasting-research/1.0"
    cursor, stop, rows = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), []
    while cursor < stop:
        chunk_end = min(cursor + timedelta(days=chunk_days), stop)
        params: dict[str, Any] = {"query": query, "mode": "artlist", "format": "json",
            "maxrecords": max_records, "sort": "datedesc",
            "startdatetime": cursor.strftime("%Y%m%d%H%M%S"),
            "enddatetime": chunk_end.strftime("%Y%m%d%H%M%S")}
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = session.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=45)
                if response.status_code == 429: time.sleep(2 ** attempt); continue
                response.raise_for_status()
                articles = response.json().get("articles", [])
                rows.extend(articles); last_error = None; break
            except Exception as exc:
                last_error = exc; time.sleep(2 ** attempt)
        if last_error: LOGGER.warning("GDELT chunk %s failed: %s", cursor.date(), last_error)
        cursor = chunk_end; time.sleep(0.15)
    if not rows:
        if path.exists() and path.stat().st_size > 100: return pd.read_csv(path)
        raise ValueError("GDELT returned no articles and no cache exists")
    news = pd.DataFrame(rows).rename(columns={"seendate": "published_at", "title": "headline"})
    keep = [c for c in ["published_at","headline","url","domain","language","sourcecountry"] if c in news]
    news = news[keep].drop_duplicates(subset=["published_at","headline","url"])
    news["source"] = "GDELT DOC 2.0"
    return _write_with_metadata(news, path)
