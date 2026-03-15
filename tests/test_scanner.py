"""
Tests for Alpha Sniper core scanner and API endpoints.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import Settings
from src.features.engineering import compute_macd, compute_rsi
from src.models.scanner import scan_stock


# ── Fixtures ──


def _make_price_series(n: int = 300, seed: int = 42) -> pd.Series:
    """Generate a synthetic price series with realistic-ish noise."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.02, n)
    prices = 100 * np.cumprod(1 + returns)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    return pd.Series(prices, index=idx, name="Close")


def _make_volume_series(n: int = 300, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    volumes = rng.integers(500_000, 5_000_000, size=n).astype(float)
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    return pd.Series(volumes, index=idx, name="Volume")


# ── Feature Engineering Tests ──


class TestIndicators:
    def test_rsi_range(self):
        prices = _make_price_series()
        rsi = compute_rsi(prices)
        valid = rsi.dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_rsi_length(self):
        prices = _make_price_series(50)
        rsi = compute_rsi(prices, period=14)
        assert len(rsi) == len(prices)

    def test_macd_shapes(self):
        prices = _make_price_series()
        macd_line, signal_line, hist = compute_macd(prices)
        assert len(macd_line) == len(prices)
        assert len(signal_line) == len(prices)
        assert len(hist) == len(prices)


# ── Scanner Tests ──


class TestScanner:
    def test_returns_none_for_short_series(self):
        short = _make_price_series(100)
        result = scan_stock(
            "TEST", short, None, None, None,
            {"TEST": "Technology"}, {}, {}, {},
            cfg=Settings(test_mode=True),
        )
        assert result is None

    def test_scan_returns_signal_or_none(self):
        close = _make_price_series(300)
        vol = _make_volume_series(300)
        result = scan_stock(
            "TEST", close, vol, close * 1.01, close * 0.99,
            {"TEST": "Technology"},
            {"TEST": {"surprise_pct": 10, "recent_beat": True, "earnings_danger": False}},
            {"TEST": {"cluster": True, "num_buyers": 3}},
            {"Technology": "up"},
            cfg=Settings(test_mode=True),
        )
        if result is not None:
            assert result.signal in ("STRONG BUY", "BUY", "WATCH")
            assert 0 <= result.score <= 130
            assert 0.0 <= result.confidence <= 1.0

    def test_signal_dict_serialisation(self):
        close = _make_price_series(300, seed=7)
        vol = _make_volume_series(300, seed=7)
        result = scan_stock(
            "AAPL", close, vol, close * 1.01, close * 0.99,
            {"AAPL": "Technology"},
            {"AAPL": {"surprise_pct": 20, "recent_beat": True, "earnings_danger": False}},
            {"AAPL": {"cluster": True, "num_buyers": 4}},
            {"Technology": "up"},
            cfg=Settings(test_mode=True),
        )
        if result is not None:
            d = result.to_dict()
            assert "ticker" in d
            assert "timestamp" in d
            assert isinstance(d["key_metrics"], dict)


# ── API Tests ──


class TestAPI:
    def test_health(self):
        from src.main import app
        client = TestClient(app)
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data
