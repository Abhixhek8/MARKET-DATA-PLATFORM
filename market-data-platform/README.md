# Market Data Platform

An end-to-end data platform: **ingest → clean → quality-gate → star-schema warehouse → REST API → dashboard → forecasting & anomaly detection.**
One codebase, built so a Python developer, data engineer, data analyst or ML engineer can each point to "their" part.

```
yfinance / synthetic fallback
        │  retries + backoff, immutable raw snapshots (Parquet)
        ▼
 ingestion.py ─► transform.py ─► quality.py ─► warehouse.py
                 (clean, dedupe)  (fail loudly)  (star schema, idempotent upserts)
                                                      │
                  ┌───────────────────────────────────┼──────────────────────┐
                  ▼                                   ▼                      ▼
           api/main.py (FastAPI)            dashboard/app.py          ml.py (forecast,
           auth, pagination                 (Streamlit + Plotly)      anomalies, MLflow)
```

## Quick start (works offline)

```bash
python -m venv .venv && source .venv/bin/activate
make install
make demo          # loads ~2 years of synthetic data for 20 tickers
make api           # http://localhost:8000/docs   (header: X-API-Key: dev-key)
make dashboard     # http://localhost:8501
make test
```

Live data: `make run` (uses yfinance, silently falls back to synthetic data per ticker if the network or ticker fails).
Everything in Docker: `make docker`.

## Who owns what

| Role | Look at | Shows |
|---|---|---|
| Python developer | `src/mdp/api/`, `tests/`, `Dockerfile`, `.github/workflows/ci.yml`, `pyproject.toml` | FastAPI, typing, dependency injection, API-key auth, pagination, pytest, CI, packaging, CLI entry point |
| Data engineer | `ingestion.py`, `transform.py`, `quality.py`, `warehouse.py`, `dags/` | retries/backoff, incremental loads with watermark, idempotent upserts, star schema, data-quality gate, Airflow DAG |
| Data analyst | `analytics.py`, `dashboard/app.py` | returns, SMA, annualised volatility, drawdown, Sharpe, correlations, volatility heatmap, top movers |
| ML engineer | `ml.py`, `pipeline.py train` | lag-feature Ridge forecast, **rolling-origin backtest vs naive baseline**, z-score anomaly detection, MLflow tracking |
| DevOps | `Dockerfile`, `docker-compose.yml`, CI | containers, service dependencies, shared volume, CI matrix |

## API

| Endpoint | Description |
|---|---|
| `GET /health` | liveness (public) |
| `GET /tickers` | available symbols |
| `GET /prices/{ticker}?start=&end=&limit=&offset=` | paginated OHLCV |
| `GET /metrics/{ticker}` | return, volatility, Sharpe, max drawdown, SMAs |
| `GET /forecast/{ticker}?horizon=5` | forecast with 95% band and backtest metrics |
| `GET /anomalies/{ticker}?z=3` | unusual-return days |

```bash
curl -H "X-API-Key: dev-key" localhost:8000/metrics/AAPL
```

## Design decisions (good interview material)

- **Idempotent loads.** `INSERT ... ON CONFLICT DO UPDATE` on `(ticker_id, date_id)`; re-running never duplicates. The pipeline is safe to retry.
- **Incremental ingestion.** Start date = last loaded date minus a 5-day overlap, which catches late revisions.
- **Quality gate before load.** Nulls, duplicates, non-positive prices, `high < low` and >50% daily moves abort the run with a clear error.
- **Immutable raw layer.** Every fetch is stored timestamped, so you can reprocess without re-downloading.
- **Honest ML.** The model is compared with a naive baseline. On random-walk-like prices it will roughly tie, and that is the correct, reportable result. Don't claim you can predict stocks.
- **SQLite as the warehouse** keeps setup at zero. The SQL is portable; swap `warehouse.py`'s connection for DuckDB or Postgres to scale.

## Next steps to level it up

1. Swap SQLite for Postgres/DuckDB and add dbt models on top of `fact_prices`.
2. Replace the Ridge model with Prophet/LightGBM and register models in MLflow.
3. Add Great Expectations or Soda for richer data checks.
4. Deploy API + dashboard to Render/Railway; schedule the pipeline with Airflow or cron.
5. Add streaming (Kafka) or an LLM "ask your data" endpoint.

*Not financial advice. Demo project.*
