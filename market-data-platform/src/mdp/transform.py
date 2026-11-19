"""Transform layer: standardise types, dedupe, repair small gaps, drop invalid rows."""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)
PRICE_COLS = ["open", "high", "low", "close"]


def clean(raw: pd.DataFrame, max_gap_fill: int = 3) -> pd.DataFrame:
    if raw.empty:
        return raw.copy()
    df = raw.copy()
    df.columns = [c.lower() for c in df.columns]
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    for c in PRICE_COLS + ["volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    log.info("removed %d duplicate rows", before - len(df))

    # Forward-fill short gaps per ticker only (never invent long histories).
    df[PRICE_COLS] = df.groupby("ticker")[PRICE_COLS].transform(lambda s: s.ffill(limit=max_gap_fill))
    df["volume"] = df["volume"].fillna(0)

    before = len(df)
    df = df.dropna(subset=PRICE_COLS)
    df = df[(df[PRICE_COLS] > 0).all(axis=1)]
    log.info("dropped %d invalid rows", before - len(df))
    return df.reset_index(drop=True)[["ticker", "date", *PRICE_COLS, "volume"]]
