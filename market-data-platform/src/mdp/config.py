"""Central configuration, driven by environment variables (12-factor style)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "WMT",
    "JNJ", "PG", "XOM", "UNH", "HD", "BAC", "KO", "PEP", "NFLX", "DIS",
]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("MDP_DATA_DIR", "data")))
    db_path: Path = field(default_factory=lambda: Path(os.getenv("MDP_DB_PATH", "data/warehouse.db")))
    api_key: str = field(default_factory=lambda: os.getenv("MDP_API_KEY", "dev-key"))
    source: str = field(default_factory=lambda: os.getenv("MDP_SOURCE", "auto"))  # auto|yfinance|synthetic
    tickers: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            t.strip().upper()
            for t in os.getenv("MDP_TICKERS", ",".join(DEFAULT_TICKERS)).split(",")
            if t.strip()
        )
    )
    lookback_days: int = int(os.getenv("MDP_LOOKBACK_DAYS", "730"))
    max_daily_move: float = float(os.getenv("MDP_MAX_DAILY_MOVE", "0.5"))

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


def get_settings() -> Settings:
    return Settings()
