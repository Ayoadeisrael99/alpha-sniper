"""
Feature engineering — technical indicators and sector regime detection.

Pure functions operating on pandas Series / DataFrames.
No API calls or side effects.
"""

import numpy as np
import pandas as pd


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def compute_sector_trends(
    close_prices: pd.DataFrame,
    ticker_sector: dict[str, str],
    lookback: int = 63,
    up_threshold: float = 2.0,
    down_threshold: float = -2.0,
) -> dict[str, str]:
    """Classify each sector as 'up', 'down', or 'flat' based on recent returns."""
    sector_tickers: dict[str, list[str]] = {}
    for t in close_prices.columns:
        s = ticker_sector.get(t, "Unknown")
        sector_tickers.setdefault(s, []).append(t)

    sector_trend: dict[str, str] = {}
    for sector, tickers in sector_tickers.items():
        rets = []
        for t in tickers:
            p = close_prices[t].dropna()
            if len(p) > lookback:
                rets.append((p.iloc[-1] / p.iloc[-lookback] - 1) * 100)
        avg = float(np.mean(rets)) if rets else 0.0
        if avg > up_threshold:
            sector_trend[sector] = "up"
        elif avg < down_threshold:
            sector_trend[sector] = "down"
        else:
            sector_trend[sector] = "flat"
    return sector_trend
