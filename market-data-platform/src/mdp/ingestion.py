"""Ingestion layer: pull OHLCV data, with retries/backoff, and persist an immutable raw copy."""
from __future__ import annotations

import functools
import hashlib
import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
RAW_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume"]


def retry(attempts: int = 3, base_delay: float = 1.0) -> Callable:
    """Retry with exponential backoff (handles flaky networks and rate limits)."""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for i in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    if i == attempts:
                        raise
                    delay = base_delay * 2 ** (i - 1)
                    log.warning("%s failed (%s). retry %d/%d in %.1fs", fn.__name__, exc, i, attempts, delay)
                    time.sleep(delay)

        return wrapper

    return deco


@retry(attempts=3, base_delay=2.0)
def fetch_yfinance(ticker: str, start: date, end: date) -> pd.DataFrame:
    import yfinance as yf  # lazy import keeps the package usable offline

    df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(),
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"no data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns=str.lower)
    df["ticker"] = ticker
    return df[RAW_COLUMNS]


def fetch_synthetic(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Deterministic geometric random walk. Lets the whole platform run offline / in CI."""
    seed = int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    if len(dates) == 0:
        return pd.DataFrame(columns=RAW_COLUMNS)
    drift, vol = rng.uniform(0.0001, 0.0008), rng.uniform(0.01, 0.025)
    rets = rng.normal(drift, vol, len(dates))
    close = rng.uniform(40, 400) * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, 0.003, len(dates)))
    spread = np.abs(rng.normal(0.006, 0.003, len(dates)))
    return pd.DataFrame({
        "ticker": ticker,
        "date": dates,
        "open": open_,
        "high": np.maximum(open_, close) * (1 + spread),
        "low": np.minimum(open_, close) * (1 - spread),
        "close": close,
        "volume": rng.integers(1_000_000, 50_000_000, len(dates)),
    })


def fetch_one(ticker: str, start: date, end: date, source: str = "auto") -> pd.DataFrame:
    if source == "synthetic":
        return fetch_synthetic(ticker, start, end)
    if source == "yfinance":
        return fetch_yfinance(ticker, start, end)
    try:  # auto
        return fetch_yfinance(ticker, start, end)
    except Exception as exc:  # noqa: BLE001
        log.warning("yfinance unavailable for %s (%s); using synthetic data", ticker, exc)
        return fetch_synthetic(ticker, start, end)


def save_raw(df: pd.DataFrame, raw_dir: Path, ticker: str) -> Path:
    """Write an immutable, timestamped raw snapshot (Parquet, falling back to CSV)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    folder = raw_dir / ticker
    folder.mkdir(parents=True, exist_ok=True)
    try:
        path = folder / f"{stamp}.parquet"
        df.to_parquet(path, index=False)
    except ImportError:
        path = folder / f"{stamp}.csv"
        df.to_csv(path, index=False)
    return path


def ingest(
    tickers: list[str] | tuple[str, ...],
    start_by_ticker: dict[str, date],
    end: date,
    raw_dir: Path,
    source: str = "auto",
) -> pd.DataFrame:
    """Fetch every ticker; one failure never kills the batch, but total failure raises."""
    frames, failed = [], []
    for t in tickers:
        try:
            df = fetch_one(t, start_by_ticker[t], end, source)
            if df.empty:
                log.info("%s: nothing new", t)
                continue
            save_raw(df, raw_dir, t)
            frames.append(df)
            log.info("%s: ingested %d rows", t, len(df))
        except Exception as exc:  # noqa: BLE001
            failed.append(t)
            log.error("%s: ingestion failed: %s", t, exc)
    if failed and not frames:
        raise RuntimeError(f"all tickers failed: {failed}")
    if failed:
        log.warning("partial ingestion, failed tickers: %s", failed)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RAW_COLUMNS)
