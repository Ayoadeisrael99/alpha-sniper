"""
API routes — /scan, /stock/{ticker}, /health.
"""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import get_settings
from src.models.scanner import StockSignal, run_full_scan, scan_stock

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


# ── Request / Response schemas ──


class ScanRequest(BaseModel):
    min_market_cap: int | None = Field(None, description="Minimum market cap filter")
    sectors: list[str] | None = Field(None, description="Sector filter (e.g. ['Technology'])")
    limit: int = Field(20, ge=1, le=100, description="Max results")
    universe: str | None = Field(None, description="Override universe: sp500 | sp500+midcap | discovery")


class StockSignalResponse(BaseModel):
    ticker: str
    company_name: str
    sector: str
    signal: str
    confidence: float
    score: int
    layers: str
    price: float
    pct_from_low: float
    pct_from_high: float
    rsi: float
    key_metrics: dict
    timestamp: str


class ScanResponse(BaseModel):
    total_signals: int
    scan_universe: str
    signals: list[StockSignalResponse]
    generated_at: str


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


# ── Endpoints ──


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness / readiness probe."""
    cfg = get_settings()
    return HealthResponse(
        status="healthy",
        version=cfg.app_version,
        timestamp=datetime.utcnow().isoformat(),
    )


@router.post("/scan", response_model=ScanResponse)
async def run_scan(body: ScanRequest | None = None):
    """
    Execute a full 10-layer Alpha Sniper scan.

    This is a long-running operation (~15-30 min on discovery universe
    due to Finnhub rate limits).  For production, run as a background
    task and poll /scan/status.
    """
    cfg = get_settings()

    # Override universe if provided
    if body and body.universe:
        cfg = cfg.model_copy(update={"universe": body.universe})

    try:
        signals = run_full_scan(cfg)
    except Exception as exc:
        logger.exception("Scan failed")
        raise HTTPException(status_code=500, detail=f"Scan error: {exc}") from exc

    # Apply filters
    if body and body.sectors:
        allowed = {s.lower() for s in body.sectors}
        signals = [s for s in signals if s.sector.lower() in allowed]

    limit = body.limit if body else 20
    signals = signals[:limit]

    return ScanResponse(
        total_signals=len(signals),
        scan_universe=cfg.universe,
        signals=[
            StockSignalResponse(
                ticker=s.ticker,
                company_name=s.company_name,
                sector=s.sector,
                signal=s.signal,
                confidence=s.confidence,
                score=s.score,
                layers=f"{s.layers}/{s.layers_max}",
                price=s.price,
                pct_from_low=s.pct_from_low,
                pct_from_high=s.pct_from_high,
                rsi=s.rsi,
                key_metrics=s.key_metrics,
                timestamp=s.timestamp.isoformat(),
            )
            for s in signals
        ],
        generated_at=datetime.utcnow().isoformat(),
    )


@router.get("/stock/{ticker}", response_model=StockSignalResponse | dict)
async def analyze_single_stock(ticker: str):
    """Analyze a single ticker through the 10-layer engine."""
    from src.data.fetcher import (
        build_universe,
        fetch_earnings,
        fetch_insider_sentiment,
        fetch_prices,
    )
    from src.features.engineering import compute_sector_trends

    cfg = get_settings()
    ticker = ticker.upper()

    try:
        # Minimal universe for sector info
        universe = build_universe(cfg)
        sector_map = {ticker: universe.get(ticker, "Unknown")}

        # Fetch just this ticker
        price_data = fetch_prices([ticker], lookback_days=cfg.lookback_days)
        close_prices = price_data["close"]

        if ticker not in close_prices.columns:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")

        earnings = fetch_earnings(
            [ticker], cfg.finnhub_api_key,
            surprise_min=cfg.earnings_surprise_min,
            lookback_days=cfg.earnings_lookback_days,
        )
        insiders = fetch_insider_sentiment(
            [ticker], cfg.finnhub_api_key,
            lookback_days=cfg.insider_lookback_days,
        )
        sector_trend = compute_sector_trends(close_prices, sector_map)

        close = close_prices[ticker].dropna()
        vol = price_data["volume"][ticker].dropna() if ticker in price_data["volume"].columns else None
        hi = price_data["high"][ticker].dropna() if ticker in price_data["high"].columns else None
        lo = price_data["low"][ticker].dropna() if ticker in price_data["low"].columns else None

        result = scan_stock(
            ticker, close, vol, hi, lo,
            sector_map, earnings, insiders, sector_trend, cfg,
        )

        if result is None:
            return {
                "ticker": ticker,
                "signal": "NEUTRAL",
                "message": "No actionable signal — stock does not meet minimum layer/score thresholds.",
                "timestamp": datetime.utcnow().isoformat(),
            }

        return StockSignalResponse(
            ticker=result.ticker,
            company_name=result.company_name,
            sector=result.sector,
            signal=result.signal,
            confidence=result.confidence,
            score=result.score,
            layers=f"{result.layers}/{result.layers_max}",
            price=result.price,
            pct_from_low=result.pct_from_low,
            pct_from_high=result.pct_from_high,
            rsi=result.rsi,
            key_metrics=result.key_metrics,
            timestamp=result.timestamp.isoformat(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Single-stock analysis failed for %s", ticker)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
