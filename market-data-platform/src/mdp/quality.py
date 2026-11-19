"""Data quality gate. Fails the pipeline loudly so bad data never reaches the warehouse."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)


class DataQualityError(RuntimeError):
    pass


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def run_checks(df: pd.DataFrame, max_daily_move: float = 0.5) -> list[CheckResult]:
    results: list[CheckResult] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        results.append(CheckResult(name, bool(ok), detail))

    add("not_empty", len(df) > 0, f"{len(df)} rows")
    if df.empty:
        return results

    nulls = int(df[["ticker", "date", "open", "high", "low", "close"]].isna().sum().sum())
    add("no_nulls", nulls == 0, f"{nulls} null cells")

    dups = int(df.duplicated(["ticker", "date"]).sum())
    add("no_duplicates", dups == 0, f"{dups} duplicate keys")

    nonpos = int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    add("positive_prices", nonpos == 0, f"{nonpos} rows")

    bad_hl = int((df["high"] < df["low"]).sum())
    add("high_gte_low", bad_hl == 0, f"{bad_hl} rows")

    rets = df.sort_values(["ticker", "date"]).groupby("ticker")["close"].pct_change().abs()
    outliers = int((rets > max_daily_move).sum())
    add("no_extreme_moves", outliers == 0, f"{outliers} moves > {max_daily_move:.0%}")
    return results


def enforce(df: pd.DataFrame, max_daily_move: float = 0.5) -> list[CheckResult]:
    results = run_checks(df, max_daily_move)
    for r in results:
        log.log(logging.INFO if r.passed else logging.ERROR, "DQ %-18s %s %s",
                r.name, "PASS" if r.passed else "FAIL", r.detail)
    failed = [r for r in results if not r.passed]
    if failed:
        raise DataQualityError("; ".join(f"{r.name}: {r.detail}" for r in failed))
    return results
