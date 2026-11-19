"""ML layer: honest baseline forecasting (vs. naive) and anomaly detection.

Forecast model: Ridge regression on lagged log-returns, rolled forward recursively.
It is deliberately simple; the backtest compares it with a naive 'tomorrow = today'
baseline so the number you report is meaningful. Swap in Prophet / ARIMA / LightGBM here.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

log = logging.getLogger(__name__)
N_LAGS = 10


def _lag_matrix(x: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([x[i : len(x) - n_lags + i] for i in range(n_lags)])
    return X, x[n_lags:]


def _fit(log_ret: np.ndarray) -> Ridge:
    X, y = _lag_matrix(log_ret, N_LAGS)
    return Ridge(alpha=1.0).fit(X, y)


def _roll(model: Ridge, history: np.ndarray, horizon: int) -> np.ndarray:
    window = list(history[-N_LAGS:])
    preds = []
    for _ in range(horizon):
        p = float(model.predict(np.array(window[-N_LAGS:]).reshape(1, -1))[0])
        preds.append(p)
        window.append(p)
    return np.array(preds)


def backtest(close: pd.Series, horizon: int = 5, folds: int = 5) -> dict:
    """Rolling-origin backtest of the model vs the naive baseline (MAPE on price level)."""
    prices = close.dropna().to_numpy(dtype=float)
    min_train = 120
    if len(prices) < min_train + horizon * folds:
        raise ValueError("not enough history to backtest (need ~150+ rows)")
    model_err, naive_err = [], []
    for k in range(folds):
        cut = len(prices) - horizon * (folds - k)
        train, actual = prices[:cut], prices[cut : cut + horizon]
        lr = np.diff(np.log(train))
        pred_ret = _roll(_fit(lr), lr, horizon)
        pred = train[-1] * np.exp(np.cumsum(pred_ret))
        model_err.append(np.mean(np.abs((actual - pred) / actual)))
        naive_err.append(np.mean(np.abs((actual - train[-1]) / actual)))
    return {"model_mape": float(np.mean(model_err)), "naive_mape": float(np.mean(naive_err)),
            "folds": folds, "horizon": horizon}


def forecast(close: pd.Series, horizon: int = 5) -> dict:
    close = close.dropna().sort_index()
    prices = close.to_numpy(dtype=float)
    lr = np.diff(np.log(prices))
    pred = prices[-1] * np.exp(np.cumsum(_roll(_fit(lr), lr, horizon)))
    # Uncertainty band from historical vol, widening with sqrt(t).
    sigma = float(np.std(lr[-60:]))
    steps = np.arange(1, horizon + 1)
    dates = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=horizon)
    result = {
        "forecast": [
            {"date": d.strftime("%Y-%m-%d"), "predicted_close": round(float(p), 4),
             "lower": round(float(p * np.exp(-1.96 * sigma * np.sqrt(s))), 4),
             "upper": round(float(p * np.exp(1.96 * sigma * np.sqrt(s))), 4)}
            for d, p, s in zip(dates, pred, steps)
        ],
    }
    try:
        result["backtest"] = backtest(close, horizon)
    except ValueError as exc:
        result["backtest"] = {"error": str(exc)}
    return result


def detect_anomalies(close: pd.Series, window: int = 30, z_threshold: float = 3.0) -> pd.DataFrame:
    """Flag days whose return is > z_threshold std-devs from its trailing mean."""
    rets = close.sort_index().pct_change()
    mean = rets.rolling(window).mean().shift(1)
    std = rets.rolling(window).std().shift(1)
    z = (rets - mean) / std
    out = pd.DataFrame({"close": close, "return": rets, "z_score": z})
    return out[out["z_score"].abs() > z_threshold].dropna()


def log_to_mlflow(ticker: str, metrics: dict) -> None:
    """Optional experiment tracking. No-op if MLflow isn't installed."""
    try:
        import mlflow
    except ImportError:
        log.info("mlflow not installed; skipping tracking")
        return
    mlflow.set_experiment("market-forecast")
    with mlflow.start_run(run_name=ticker):
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("n_lags", N_LAGS)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
