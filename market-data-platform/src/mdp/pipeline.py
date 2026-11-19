"""Pipeline orchestration: ingest -> transform -> quality gate -> load (+ model tracking).

Usage:
    python -m mdp.pipeline run [--source synthetic] [--tickers AAPL,MSFT]
    python -m mdp.pipeline train
"""
from __future__ import annotations

import argparse
import logging
from datetime import date

from . import ingestion, ml, quality, transform
from .config import get_settings
from .logging_setup import setup_logging
from .warehouse import Warehouse, default_start

log = logging.getLogger(__name__)


def run(tickers=None, source=None, settings=None) -> int:
    s = settings or get_settings()
    tickers = list(tickers or s.tickers)
    wh = Warehouse(s.db_path)
    try:
        starts = {t: default_start(wh, t, s.lookback_days) for t in tickers}
        raw = ingestion.ingest(tickers, starts, date.today(), s.raw_dir, source or s.source)
        clean = transform.clean(raw)
        quality.enforce(clean, s.max_daily_move)  # raises -> pipeline fails loudly
        n = wh.load(clean)
        log.info("pipeline complete: %d rows upserted for %d tickers", n, len(tickers))
        return n
    finally:
        wh.close()


def train(settings=None) -> None:
    s = settings or get_settings()
    wh = Warehouse(s.db_path)
    try:
        for t in wh.tickers():
            df = wh.prices(t)
            try:
                res = ml.backtest(df.set_index("date")["close"])
            except ValueError as exc:
                log.warning("%s: %s", t, exc)
                continue
            log.info("%s backtest: model MAPE %.4f vs naive %.4f", t, res["model_mape"], res["naive_mape"])
            ml.log_to_mlflow(t, res)
    finally:
        wh.close()


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(prog="mdp")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the ETL pipeline")
    r.add_argument("--source", choices=["auto", "yfinance", "synthetic"])
    r.add_argument("--tickers", help="comma separated, overrides MDP_TICKERS")
    sub.add_parser("train", help="backtest models and log to MLflow")
    a = p.parse_args()
    if a.cmd == "run":
        run(a.tickers.split(",") if a.tickers else None, a.source)
    else:
        train()


if __name__ == "__main__":
    main()
