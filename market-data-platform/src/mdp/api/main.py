"""REST API over the warehouse: prices, metrics, forecasts, anomalies. API-key protected."""
from __future__ import annotations

from typing import Iterator

from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from .. import analytics, ml
from ..config import get_settings
from ..warehouse import Warehouse

app = FastAPI(title="Market Data Platform API", version="0.1.0")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_key(key: str | None = Security(api_key_header)) -> None:
    if key != get_settings().api_key:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def get_wh() -> Iterator[Warehouse]:
    wh = Warehouse(get_settings().db_path)
    try:
        yield wh
    finally:
        wh.close()


class PriceRow(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class PricePage(BaseModel):
    ticker: str
    total: int
    limit: int
    offset: int
    items: list[PriceRow]


def _history(wh: Warehouse, ticker: str):
    ticker = ticker.upper()
    df = wh.prices(ticker)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"unknown ticker {ticker}")
    return df


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/tickers", dependencies=[Depends(require_key)])
def tickers(wh: Warehouse = Depends(get_wh)) -> list[str]:
    return wh.tickers()


@app.get("/prices/{ticker}", response_model=PricePage, dependencies=[Depends(require_key)])
def prices(ticker: str, start: str | None = None, end: str | None = None,
           limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
           wh: Warehouse = Depends(get_wh)) -> PricePage:
    ticker = ticker.upper()
    total = wh.count(ticker, start, end)
    if total == 0:
        raise HTTPException(status_code=404, detail=f"no data for {ticker}")
    df = wh.prices(ticker, start, end, limit, offset)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    items = df.drop(columns="ticker").to_dict("records")
    return PricePage(ticker=ticker, total=total, limit=limit, offset=offset, items=items)


@app.get("/metrics/{ticker}", dependencies=[Depends(require_key)])
def metrics(ticker: str, wh: Warehouse = Depends(get_wh)) -> dict:
    return analytics.summary(_history(wh, ticker))


@app.get("/forecast/{ticker}", dependencies=[Depends(require_key)])
def forecast(ticker: str, horizon: int = Query(5, ge=1, le=30),
             wh: Warehouse = Depends(get_wh)) -> dict:
    df = _history(wh, ticker)
    try:
        result = ml.forecast(df.set_index("date")["close"], horizon)
    except (ValueError, IndexError) as exc:
        raise HTTPException(status_code=422, detail=f"cannot forecast: {exc}") from exc
    return {"ticker": ticker.upper(), "horizon": horizon, **result}


@app.get("/anomalies/{ticker}", dependencies=[Depends(require_key)])
def anomalies(ticker: str, z: float = Query(3.0, gt=0), wh: Warehouse = Depends(get_wh)) -> dict:
    df = _history(wh, ticker)
    out = ml.detect_anomalies(df.set_index("date")["close"], z_threshold=z)
    return {"ticker": ticker.upper(), "count": len(out), "items": [
        {"date": d.strftime("%Y-%m-%d"), "close": round(r["close"], 4),
         "return": round(r["return"], 4), "z_score": round(r["z_score"], 2)}
        for d, r in out.iterrows()]}
