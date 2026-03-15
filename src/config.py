"""
Application configuration loaded from environment variables.

All API keys and tunable parameters are centralized here.
Never hardcode secrets — use .env or container environment.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Alpha Sniper configuration.  Reads from env vars / .env file."""

    # ── API Keys ──
    fmp_api_key: str = Field(default="", description="Financial Modeling Prep API key")
    finnhub_api_key: str = Field(default="", description="Finnhub API key")

    # ── App Metadata ──
    app_name: str = "Alpha Sniper"
    app_version: str = "2.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # ── Stock Universe ──
    universe: str = Field(default="discovery", description="sp500 | sp500+midcap | discovery | custom")
    custom_tickers: list[str] = Field(default_factory=lambda: ["AAPL", "MSFT", "NVDA"])
    test_mode: bool = False
    test_tickers: list[str] = Field(default_factory=lambda: ["DPZ", "BRO", "CART", "DOC", "AAPL"])

    # ── Lookback / Data ──
    lookback_days: int = 400

    # ── Technical Reversal ──
    max_pct_from_52w_low: float = 15.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_recovery: float = 35.0
    volume_spike_mult: float = 1.5

    # ── Earnings Drift ──
    earnings_lookback_days: int = 60
    earnings_surprise_min: float = 5.0
    earnings_upcoming_danger: int = 14

    # ── Insider Buying ──
    insider_lookback_days: int = 90
    insider_min_buyers: int = 2

    # ── Trade Management ──
    hold_days: int = 20
    stop_loss_pct: float = -5.0
    take_profit_pct: float = 15.0

    # ── Signal Thresholds ──
    min_signal_score: int = 45
    min_layers: int = 3

    # ── Rate Limiting ──
    finnhub_calls_per_min: int = 55
    yahoo_batch_pause: float = 0.3
    yahoo_big_pause: float = 2.0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — call this everywhere instead of constructing Settings()."""
    return Settings()
