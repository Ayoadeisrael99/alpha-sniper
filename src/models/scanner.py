"""
Alpha Sniper v2 — 10-Layer Signal Scanner.

Layers:
  1. Proximity to 52-week low       (0-20 pts)
  2. RSI reversal                    (0-20 pts)
  3. Higher low formation            (0-15 pts)
  4. Volume spike / accumulation     (0-15 pts)
  5. Moving-average reclaim          (0-15 pts)
  6. MACD crossover / momentum       (0-15 pts)
  7. Earnings surprise / PEAD        (0-15 pts)
  8. Insider buying cluster          (0-10 pts)
  9. Upcoming-earnings penalty       (0 or -10)
  10. Sector regime                  (−5 to +5)

Max composite: 130 pts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import Settings, get_settings
from src.features.engineering import compute_macd, compute_rsi

logger = logging.getLogger(__name__)


@dataclass
class StockSignal:
    """Structured output for a single stock signal."""

    ticker: str
    company_name: str
    sector: str
    price: float
    low_52w: float
    high_52w: float
    pct_from_low: float
    pct_from_high: float
    signal: str          # STRONG BUY | BUY | WATCH
    score: int
    layers: int
    layers_max: int = 10
    rsi: float = 0.0
    days_since_bottom: int = 0
    confidence: float = 0.0
    cluster_label: int = 0
    key_metrics: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "sector": self.sector,
            "price": self.price,
            "low_52w": self.low_52w,
            "high_52w": self.high_52w,
            "pct_from_low": self.pct_from_low,
            "pct_from_high": self.pct_from_high,
            "signal": self.signal,
            "score": self.score,
            "layers": f"{self.layers}/{self.layers_max}",
            "confidence": self.confidence,
            "rsi": self.rsi,
            "days_since_bottom": self.days_since_bottom,
            "key_metrics": self.key_metrics,
            "timestamp": self.timestamp.isoformat(),
        }


def scan_stock(
    ticker: str,
    close: pd.Series,
    volume: pd.Series | None,
    high: pd.Series | None,
    low: pd.Series | None,
    ticker_sector: dict[str, str],
    earnings_info: dict[str, dict],
    insider_info: dict[str, dict],
    sector_trend: dict[str, str],
    cfg: Settings | None = None,
) -> StockSignal | None:
    """Run the 10-layer scan on a single stock. Returns None if no signal."""
    cfg = cfg or get_settings()

    if len(close) < 252:
        return None

    current_price = float(close.iloc[-1])
    sector = ticker_sector.get(ticker, "Unknown")

    # ═══ LAYER 1: Proximity to 52-week low (0-20) ═══
    low_52w = float(close.tail(252).min())
    high_52w = float(close.tail(252).max())
    pct_from_low = (current_price / low_52w - 1) * 100
    pct_from_high = (1 - current_price / high_52w) * 100
    near_low = pct_from_low <= cfg.max_pct_from_52w_low

    if pct_from_low <= 3:
        low_score = 20
    elif pct_from_low <= 7:
        low_score = 15
    elif pct_from_low <= cfg.max_pct_from_52w_low:
        low_score = 10
    else:
        low_score = 0

    # ═══ LAYER 2: RSI Reversal (0-20) ═══
    rsi = compute_rsi(close, cfg.rsi_period)
    current_rsi = float(rsi.iloc[-1])
    was_oversold = bool((rsi.tail(10) < cfg.rsi_oversold).any())
    was_deeply_oversold = bool((rsi.tail(20) < 25).any())
    rsi_recovering = current_rsi >= cfg.rsi_recovery
    rsi_reversal = was_oversold and rsi_recovering

    if rsi_reversal and was_deeply_oversold:
        rsi_score = 20
    elif rsi_reversal:
        rsi_score = 15
    elif was_oversold:
        rsi_score = 5
    else:
        rsi_score = 0

    # ═══ LAYER 3: Higher Low (0-15) ═══
    ref = low if low is not None and len(low) >= 20 else close
    recent_low_5d = float(ref.tail(5).min())
    prior_low = float(ref.tail(20).head(15).min())
    higher_low = recent_low_5d > prior_low
    bottom_252 = close.tail(252)
    days_since_low = len(bottom_252) - int(bottom_252.values.argmin())
    bottom_behind = days_since_low > 3

    if higher_low and bottom_behind and days_since_low > 10:
        hl_score = 15
    elif higher_low and bottom_behind:
        hl_score = 10
    elif bottom_behind:
        hl_score = 5
    else:
        hl_score = 0

    # ═══ LAYER 4: Volume Spike (0-15) ═══
    vol_score = 5
    vol_spike = False
    if volume is not None and len(volume) > 50:
        avg_vol = float(volume.tail(50).mean())
        recent_vol = float(volume.tail(5).mean())
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0
        rets = close.pct_change().tail(10)
        vols = volume.tail(10)
        up_vol = float(vols[rets > 0].mean()) if (rets > 0).any() else 0.0
        dn_vol = float(vols[rets < 0].mean()) if (rets < 0).any() else 1.0
        updn = up_vol / dn_vol if dn_vol > 0 else 1.0

        vol_spike = vol_ratio >= cfg.volume_spike_mult
        if vol_spike and updn > 1.2:
            vol_score = 15
        elif vol_spike:
            vol_score = 10
        elif updn > 1.3:
            vol_score = 8
        else:
            vol_score = 2

    # ═══ LAYER 5: MA Reclaim (0-15) ═══
    ma10 = float(close.tail(10).mean())
    ma20 = float(close.tail(20).mean())
    above_10 = current_price > ma10
    above_20 = current_price > ma20
    was_below = bool((close.tail(10).head(5) < close.tail(15).head(5).mean()).any())
    ma_reclaim = above_10 and was_below

    if above_20 and ma_reclaim:
        ma_score = 15
    elif above_10 and ma_reclaim:
        ma_score = 12
    elif above_10:
        ma_score = 7
    else:
        ma_score = 0

    # ═══ LAYER 6: MACD Crossover (0-15) ═══
    _, _, hist = compute_macd(close)
    h0 = float(hist.iloc[-1]) if len(hist) > 0 else 0.0
    h1 = float(hist.iloc[-2]) if len(hist) > 1 else 0.0
    h2 = float(hist.iloc[-3]) if len(hist) > 2 else 0.0
    macd_cross = (h0 > 0 and h1 <= 0) or (h0 > 0 and h2 <= 0)
    macd_improving = h0 > h1 > h2

    if macd_cross:
        macd_score = 15
    elif macd_improving and h0 > 0:
        macd_score = 10
    elif macd_improving:
        macd_score = 5
    else:
        macd_score = 0

    # ═══ LAYER 7: Earnings Surprise / PEAD (0-15) ═══
    earn = earnings_info.get(ticker, {})
    recent_beat = earn.get("recent_beat", False)
    surprise = earn.get("surprise_pct")

    if recent_beat and surprise is not None and surprise >= 15:
        earn_score = 15
    elif recent_beat and surprise is not None and surprise >= cfg.earnings_surprise_min:
        earn_score = 10
    elif surprise is not None and surprise > 0:
        earn_score = 3
    else:
        earn_score = 0

    # ═══ LAYER 8: Insider Buying Cluster (0-10) ═══
    ins = insider_info.get(ticker, {})
    cluster = ins.get("cluster", False)
    n_buyers = ins.get("num_buyers", 0)

    if cluster and n_buyers >= 3:
        insider_score = 10
    elif cluster:
        insider_score = 7
    elif n_buyers >= 1:
        insider_score = 3
    else:
        insider_score = 0

    # ═══ LAYER 9: Upcoming Earnings Penalty (0 / -10) ═══
    earnings_danger = earn.get("earnings_danger", False)
    danger_penalty = -10 if earnings_danger else 0

    # ═══ LAYER 10: Sector Regime (-5 to +5) ═══
    sec_trend = sector_trend.get(sector, "flat")
    if sec_trend == "up":
        sector_score = 5
    elif sec_trend == "down":
        sector_score = -5
    else:
        sector_score = 0

    # ═══ COMPOSITE ═══
    total = max(
        low_score + rsi_score + hl_score + vol_score + ma_score
        + macd_score + earn_score + insider_score + danger_penalty + sector_score,
        0,
    )

    layers_confirmed = sum([
        near_low,
        rsi_reversal,
        higher_low and bottom_behind,
        vol_score >= 10,
        ma_reclaim or above_20,
        macd_cross or (macd_improving and h0 > 0),
        recent_beat,
        cluster,
        not earnings_danger,
        sec_trend == "up",
    ])

    if total >= 85 and layers_confirmed >= 7:
        signal = "STRONG BUY"
    elif total >= 65 and layers_confirmed >= 5:
        signal = "BUY"
    elif total >= cfg.min_signal_score and layers_confirmed >= cfg.min_layers:
        signal = "WATCH"
    else:
        return None

    confidence = round(min(total / 130.0, 1.0), 3)

    return StockSignal(
        ticker=ticker,
        company_name=ticker,  # enriched later if needed
        sector=sector,
        price=round(current_price, 2),
        low_52w=round(low_52w, 2),
        high_52w=round(high_52w, 2),
        pct_from_low=round(pct_from_low, 1),
        pct_from_high=round(pct_from_high, 1),
        signal=signal,
        score=total,
        layers=layers_confirmed,
        rsi=round(current_rsi, 1),
        days_since_bottom=days_since_low,
        confidence=confidence,
        key_metrics={
            "low_score": low_score, "rsi_score": rsi_score, "hl_score": hl_score,
            "vol_score": vol_score, "ma_score": ma_score, "macd_score": macd_score,
            "earn_score": earn_score, "insider_score": insider_score,
            "danger_penalty": danger_penalty, "sector_score": sector_score,
            "rsi_reversal": rsi_reversal, "higher_low": higher_low and bottom_behind,
            "vol_spike": vol_spike, "ma_reclaim": ma_reclaim or above_20,
            "macd_cross": macd_cross, "earnings_beat": recent_beat,
            "insider_cluster": cluster, "earnings_danger": earnings_danger,
            "sector_trend": sec_trend,
        },
    )


def run_full_scan(
    cfg: Settings | None = None,
) -> list[StockSignal]:
    """
    Orchestrate the complete pipeline:
    Fetch universe → Download prices → Finnhub data → Engineer features → Scan → Rank.
    """
    from src.data.fetcher import (
        build_universe,
        fetch_earnings,
        fetch_insider_sentiment,
        fetch_prices,
    )
    from src.features.engineering import compute_sector_trends

    cfg = cfg or get_settings()

    # 1. Build universe
    logger.info("Building universe: %s", cfg.universe)
    ticker_sector = build_universe(cfg)
    tickers = list(ticker_sector.keys())
    logger.info("Universe: %d tickers", len(tickers))

    # 2. Fetch prices
    logger.info("Fetching price data (%d-day lookback)", cfg.lookback_days)
    price_data = fetch_prices(
        tickers,
        lookback_days=cfg.lookback_days,
        batch_pause=cfg.yahoo_batch_pause,
        big_pause=cfg.yahoo_big_pause,
    )
    close_prices = price_data["close"]
    volume_data = price_data["volume"]
    high_data = price_data["high"]
    low_data = price_data["low"]

    active_tickers = list(close_prices.columns)
    logger.info("Prices loaded for %d tickers", len(active_tickers))

    # 3. Finnhub: earnings
    logger.info("Fetching earnings data from Finnhub")
    earnings_info = fetch_earnings(
        active_tickers,
        finnhub_key=cfg.finnhub_api_key,
        surprise_min=cfg.earnings_surprise_min,
        lookback_days=cfg.earnings_lookback_days,
        danger_days=cfg.earnings_upcoming_danger,
        calls_per_min=cfg.finnhub_calls_per_min,
    )

    # 4. Finnhub: insider sentiment
    logger.info("Fetching insider sentiment from Finnhub")
    insider_info = fetch_insider_sentiment(
        active_tickers,
        finnhub_key=cfg.finnhub_api_key,
        lookback_days=cfg.insider_lookback_days,
        min_buyers=cfg.insider_min_buyers,
        calls_per_min=cfg.finnhub_calls_per_min,
    )

    # 5. Sector trends
    sector_trend = compute_sector_trends(close_prices, ticker_sector)

    # 6. Scan
    logger.info("Running 10-layer scan on %d tickers", len(active_tickers))
    signals: list[StockSignal] = []
    for ticker in active_tickers:
        close = close_prices[ticker].dropna()
        vol = volume_data[ticker].dropna() if ticker in volume_data.columns else None
        hi = high_data[ticker].dropna() if ticker in high_data.columns else None
        lo = low_data[ticker].dropna() if ticker in low_data.columns else None

        result = scan_stock(
            ticker, close, vol, hi, lo,
            ticker_sector, earnings_info, insider_info, sector_trend, cfg,
        )
        if result:
            signals.append(result)

    signals.sort(key=lambda s: s.score, reverse=True)
    logger.info("Scan complete: %d signals", len(signals))
    return signals
