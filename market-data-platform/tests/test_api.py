import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from mdp.pipeline import run  # noqa: E402

H = {"X-API-Key": "test-key"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MDP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MDP_DB_PATH", str(tmp_path / "wh.db"))
    monkeypatch.setenv("MDP_API_KEY", "test-key")
    run(["AAPL", "MSFT"], source="synthetic")
    from mdp.api.main import app
    return TestClient(app)


def test_health_is_public(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_auth_required(client):
    assert client.get("/tickers").status_code == 401


def test_prices_paginated(client):
    r = client.get("/prices/aapl?limit=5&offset=2", headers=H).json()
    assert len(r["items"]) == 5 and r["total"] > 5


def test_metrics_forecast_anomalies(client):
    assert client.get("/metrics/AAPL", headers=H).status_code == 200
    assert len(client.get("/forecast/AAPL?horizon=3", headers=H).json()["forecast"]) == 3
    assert "count" in client.get("/anomalies/AAPL", headers=H).json()


def test_unknown_ticker_404(client):
    assert client.get("/metrics/ZZZZ", headers=H).status_code == 404
