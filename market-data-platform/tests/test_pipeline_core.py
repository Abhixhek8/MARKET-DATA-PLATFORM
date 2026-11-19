import numpy as np
import pandas as pd
import pytest

from mdp import analytics, ml
from mdp.quality import DataQualityError, enforce, run_checks
from mdp.transform import clean
from mdp.warehouse import Warehouse


# ---------- transform ----------
def test_clean_removes_duplicates_and_bad_rows(sample):
    dirty = pd.concat([sample, sample.head(5)], ignore_index=True)
    dirty.loc[0, "close"] = -1
    out = clean(dirty)
    assert not out.duplicated(["ticker", "date"]).any()
    assert (out["close"] > 0).all()


def test_clean_fills_short_gaps_only():
    df = pd.DataFrame({"ticker": "X", "date": pd.bdate_range("2024-01-01", periods=8),
                       "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 10})
    df.loc[2:3, ["open", "high", "low", "close"]] = np.nan
    assert len(clean(df)) == 8


# ---------- quality ----------
def test_quality_passes_on_clean_data(sample):
    assert all(r.passed for r in run_checks(sample))


def test_quality_fails_loudly_on_outlier(sample):
    bad = sample.copy()
    bad.loc[10, "close"] = bad.loc[10, "close"] * 10
    with pytest.raises(DataQualityError):
        enforce(bad)


def test_quality_fails_on_empty():
    with pytest.raises(DataQualityError):
        enforce(pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"]))


# ---------- warehouse ----------
def test_load_is_idempotent(sample):
    wh = Warehouse(":memory:")
    wh.load(sample)
    first = wh.conn.execute("SELECT COUNT(*) FROM fact_prices").fetchone()[0]
    wh.load(sample)
    assert wh.conn.execute("SELECT COUNT(*) FROM fact_prices").fetchone()[0] == first
    assert first == len(sample)


def test_upsert_updates_changed_values(sample):
    wh = Warehouse(":memory:")
    wh.load(sample)
    changed = sample.head(1).copy()
    changed["close"] = 123.45
    wh.load(changed)
    row = changed.iloc[0]
    got = wh.prices(row["ticker"], start=row["date"].strftime("%Y-%m-%d"), limit=1)
    assert got["close"].iloc[0] == pytest.approx(123.45)


def test_pagination_and_matrix(sample):
    wh = Warehouse(":memory:")
    wh.load(sample)
    assert len(wh.prices("AAA", limit=10)) == 10
    assert wh.count("AAA") == len(sample[sample.ticker == "AAA"])
    assert set(wh.close_matrix().columns) == {"AAA", "BBB", "CCC"}


# ---------- analytics ----------
def test_summary_fields(sample):
    s = analytics.summary(sample[sample.ticker == "AAA"])
    assert s["ticker"] == "AAA" and s["max_drawdown"] <= 0 and s["annualised_volatility"] > 0


def test_movers_and_correlation(sample):
    wide = sample.pivot(index="date", columns="ticker", values="close")
    assert len(analytics.top_movers(wide, 2)["gainers"]) == 2
    corr = analytics.return_correlation(wide)
    assert corr.shape == (3, 3) and np.allclose(np.diag(corr), 1)


# ---------- ML ----------
def test_forecast_shape_and_band(sample):
    s = sample[sample.ticker == "AAA"].set_index("date")["close"]
    out = ml.forecast(s, horizon=7)
    assert len(out["forecast"]) == 7
    assert all(f["lower"] < f["predicted_close"] < f["upper"] for f in out["forecast"])
    assert "model_mape" in out["backtest"] and "naive_mape" in out["backtest"]


def test_anomaly_detector_finds_injected_spike(sample):
    s = sample[sample.ticker == "AAA"].set_index("date")["close"].copy()
    s.iloc[200:] *= 1.6  # +60% jump
    assert s.index[200] in ml.detect_anomalies(s).index
