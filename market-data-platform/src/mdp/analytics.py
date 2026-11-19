"""Analyst layer: KPIs such as returns, moving averages, volatility, movers, correlations."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def add_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Expects one ticker, sorted by date. Adds return, SMA, volatility and drawdown columns."""
    out = df.sort_values("date").copy()
    out["daily_return"] = out["close"].pct_change()
    out["sma_20"] = out["close"].rolling(20).mean()
    out["sma_50"] = out["close"].rolling(50).mean()
    out["volatility_20d"] = out["daily_return"].rolling(20).std() * np.sqrt(TRADING_DAYS)
    out["cumulative_return"] = out["close"] / out["close"].iloc[0] - 1
    out["drawdown"] = out["close"] / out["close"].cummax() - 1
    return out


def summary(df: pd.DataFrame) -> dict:
    m = add_metrics(df)
    r = m["daily_return"].dropna()
    ann_ret = float(r.mean() * TRADING_DAYS) if len(r) else 0.0
    ann_vol = float(r.std() * np.sqrt(TRADING_DAYS)) if len(r) > 1 else 0.0
    last = m.iloc[-1]
    return {
        "ticker": str(df["ticker"].iloc[0]),
        "as_of": last["date"].strftime("%Y-%m-%d"),
        "last_close": round(float(last["close"]), 4),
        "total_return": round(float(last["cumulative_return"]), 4),
        "annualised_return": round(ann_ret, 4),
        "annualised_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(ann_ret / ann_vol, 3) if ann_vol else None,
        "max_drawdown": round(float(m["drawdown"].min()), 4),
        "sma_20": None if pd.isna(last["sma_20"]) else round(float(last["sma_20"]), 4),
        "sma_50": None if pd.isna(last["sma_50"]) else round(float(last["sma_50"]), 4),
    }


def top_movers(close_matrix: pd.DataFrame, n: int = 5) -> dict[str, pd.Series]:
    """Best/worst tickers by latest 1-day return."""
    rets = close_matrix.pct_change().iloc[-1].dropna().sort_values()
    return {"losers": rets.head(n), "gainers": rets.tail(n)[::-1]}


def return_correlation(close_matrix: pd.DataFrame) -> pd.DataFrame:
    return close_matrix.pct_change().dropna(how="all").corr()


def volatility_heatmap(close_matrix: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Month-end annualised rolling volatility, ticker x month."""
    vol = close_matrix.pct_change().rolling(window).std() * np.sqrt(TRADING_DAYS)
    return vol.resample("ME").last().dropna(how="all").T
