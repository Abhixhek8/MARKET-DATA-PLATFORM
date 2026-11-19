"""Warehouse: star schema (fact_prices + dim_ticker + dim_date) with idempotent upserts.

SQLite keeps this dependency-free. The SQL is standard, so moving to DuckDB or Postgres
only means swapping the connection (and `?` placeholders for `%s` on Postgres).
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS dim_ticker (
    ticker_id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol    TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS dim_date (
    date_id     INTEGER PRIMARY KEY,   -- YYYYMMDD
    date        TEXT NOT NULL UNIQUE,
    year        INTEGER, quarter INTEGER, month INTEGER, day_of_week INTEGER
);
CREATE TABLE IF NOT EXISTS fact_prices (
    ticker_id INTEGER NOT NULL REFERENCES dim_ticker(ticker_id),
    date_id   INTEGER NOT NULL REFERENCES dim_date(date_id),
    open REAL, high REAL, low REAL, close REAL, volume INTEGER,
    PRIMARY KEY (ticker_id, date_id)
);
CREATE INDEX IF NOT EXISTS ix_fact_date ON fact_prices(date_id);
"""

PRICES_SQL = """
SELECT t.symbol AS ticker, d.date AS date, f.open, f.high, f.low, f.close, f.volume
FROM fact_prices f
JOIN dim_ticker t USING (ticker_id)
JOIN dim_date d USING (date_id)
"""


class Warehouse:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ---------- load ----------
    def load(self, df: pd.DataFrame) -> int:
        """Idempotent: re-running with the same data never creates duplicates."""
        if df.empty:
            return 0
        with self.tx() as c:
            c.executemany("INSERT OR IGNORE INTO dim_ticker(symbol) VALUES (?)",
                          [(s,) for s in df["ticker"].unique()])
            dates = pd.to_datetime(df["date"]).drop_duplicates()
            c.executemany(
                "INSERT OR IGNORE INTO dim_date VALUES (?,?,?,?,?,?)",
                [(int(d.strftime("%Y%m%d")), d.strftime("%Y-%m-%d"), d.year,
                  (d.month - 1) // 3 + 1, d.month, d.dayofweek) for d in dates],
            )
            ids = dict(c.execute("SELECT symbol, ticker_id FROM dim_ticker").fetchall())
            rows = [
                (ids[r.ticker], int(r.date.strftime("%Y%m%d")),
                 r.open, r.high, r.low, r.close, int(r.volume))
                for r in df.itertuples(index=False)
            ]
            c.executemany(
                """INSERT INTO fact_prices VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(ticker_id, date_id) DO UPDATE SET
                     open=excluded.open, high=excluded.high, low=excluded.low,
                     close=excluded.close, volume=excluded.volume""",
                rows,
            )
        return len(rows)

    # ---------- read ----------
    def tickers(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT symbol FROM dim_ticker ORDER BY symbol")]

    def last_date(self, symbol: str) -> date | None:
        row = self.conn.execute(
            "SELECT MAX(d.date) FROM fact_prices f JOIN dim_ticker t USING(ticker_id) "
            "JOIN dim_date d USING(date_id) WHERE t.symbol=?", (symbol,)).fetchone()
        return date.fromisoformat(row[0]) if row and row[0] else None

    def prices(self, symbol: str | None = None, start: str | None = None, end: str | None = None,
               limit: int | None = None, offset: int = 0) -> pd.DataFrame:
        sql, params = PRICES_SQL + " WHERE 1=1", []
        if symbol:
            sql += " AND t.symbol = ?"; params.append(symbol)
        if start:
            sql += " AND d.date >= ?"; params.append(start)
        if end:
            sql += " AND d.date <= ?"; params.append(end)
        sql += " ORDER BY t.symbol, d.date"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"; params += [limit, offset]
        df = pd.read_sql_query(sql, self.conn, params=params)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def count(self, symbol: str, start: str | None = None, end: str | None = None) -> int:
        sql = ("SELECT COUNT(*) FROM fact_prices f JOIN dim_ticker t USING(ticker_id) "
               "JOIN dim_date d USING(date_id) WHERE t.symbol=?")
        params: list = [symbol]
        if start:
            sql += " AND d.date >= ?"; params.append(start)
        if end:
            sql += " AND d.date <= ?"; params.append(end)
        return self.conn.execute(sql, params).fetchone()[0]

    def close_matrix(self) -> pd.DataFrame:
        """Wide date x ticker matrix of close prices (for correlations, movers)."""
        df = self.prices()
        return df.pivot(index="date", columns="ticker", values="close") if not df.empty else df


def default_start(wh: Warehouse, symbol: str, lookback_days: int, overlap_days: int = 5) -> date:
    """Incremental-load watermark with a small overlap to catch late revisions."""
    last = wh.last_date(symbol)
    if last is None:
        return date.today() - timedelta(days=lookback_days)
    return last - timedelta(days=overlap_days)
