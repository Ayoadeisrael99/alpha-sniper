"""
Data fetching layer — Yahoo Finance, FMP, and Finnhub.

Each function is rate-limit-aware and returns clean DataFrames / dicts.
All API keys come from config, never hardcoded.
"""

import io
import logging
import time
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36"
    )
}


# ───────────────────────────────────────────────────
#  STOCK UNIVERSE BUILDERS
# ───────────────────────────────────────────────────


def scrape_sp500() -> dict[str, str]:
    """Return {ticker: sector} for S&P 500 from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    df = pd.read_html(io.StringIO(resp.text))[0]
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return dict(zip(df["Symbol"], df["GICS Sector"]))


def scrape_midcap() -> dict[str, str]:
    """Return {ticker: sector} for S&P 400 MidCap."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        df = pd.read_html(io.StringIO(resp.text))[0]
        sym_col = next(c for c in df.columns if "ymbol" in c or "icker" in c)
        sec_cols = [c for c in df.columns if "ector" in c.lower()]
        df[sym_col] = df[sym_col].astype(str).str.replace(".", "-", regex=False).str.strip()
        if sec_cols:
            return dict(zip(df[sym_col], df[sec_cols[0]]))
        return {t: "MidCap" for t in df[sym_col]}
    except Exception as exc:
        logger.warning("MidCap scrape failed: %s", exc)
        return {}


def discover_small_caps(fmp_key: str) -> dict[str, str]:
    """Find active, high-volume, and small-cap stocks via FMP."""
    discovered: dict[str, str] = {}

    endpoints = [
        ("actives", f"https://financialmodelingprep.com/api/v3/stock_market/actives?apikey={fmp_key}"),
        ("losers", f"https://financialmodelingprep.com/api/v3/stock_market/losers?apikey={fmp_key}"),
        ("gainers", f"https://financialmodelingprep.com/api/v3/stock_market/gainers?apikey={fmp_key}"),
        (
            "screener",
            f"https://financialmodelingprep.com/api/v3/stock-screener"
            f"?marketCapMoreThan=100000000&marketCapLowerThan=5000000000"
            f"&priceMoreThan=2&priceLowerThan=50"
            f"&volumeMoreThan=500000&exchange=NYSE,NASDAQ"
            f"&limit=200&apikey={fmp_key}",
        ),
    ]

    for label, url in endpoints:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                continue
            for stock in resp.json():
                sym = stock.get("symbol", "")
                if sym and "." not in sym and len(sym) <= 5:
                    if label == "screener":
                        discovered[sym] = stock.get("sector", "SmallCap") or "SmallCap"
                    else:
                        discovered[sym] = f"{label.title()} - {stock.get('name', '')[:30]}"
            logger.info("FMP %s: found %d stocks", label, len(discovered))
        except Exception as exc:
            logger.warning("FMP %s failed: %s", label, exc)

    # Hand-picked popular retail stocks
    popular_retail = {
        "ACHR": "eVTOL/Aviation", "PLTR": "AI/Defense", "SOFI": "Fintech",
        "RIVN": "EV", "LCID": "EV", "NIO": "EV", "JOBY": "eVTOL",
        "RKLB": "Space", "IONQ": "Quantum", "RGTI": "Quantum",
        "HOOD": "Fintech", "AFRM": "Fintech", "UPST": "AI/Fintech",
        "SOUN": "AI/Voice", "BBAI": "AI/Gov", "AI": "AI/Enterprise",
        "SMCI": "AI/Hardware", "ARM": "Semiconductor",
        "MARA": "Crypto/Mining", "COIN": "Crypto",
        "DNA": "Biotech", "BEAM": "Biotech",
        "LUNR": "Space", "ASTS": "Space/Telecom",
        "HIMS": "Telehealth", "DOCS": "Telehealth",
        "RDDT": "Social Media", "GRAB": "Ride-hailing", "SE": "E-commerce/Gaming",
    }
    for sym, sector in popular_retail.items():
        discovered.setdefault(sym, sector)

    return discovered


def build_universe(cfg: Settings | None = None) -> dict[str, str]:
    """Build the full ticker→sector mapping based on config."""
    cfg = cfg or get_settings()

    if cfg.test_mode:
        return {t: "Test" for t in cfg.test_tickers}

    universe = cfg.universe.lower()
    if universe == "sp500":
        return scrape_sp500()
    elif universe == "sp500+midcap":
        return {**scrape_sp500(), **scrape_midcap()}
    elif universe == "discovery":
        base = {**scrape_sp500(), **scrape_midcap()}
        extra = discover_small_caps(cfg.fmp_api_key)
        for sym, sector in extra.items():
            base.setdefault(sym, sector)
        return base
    elif universe == "custom":
        return {t: "Custom" for t in cfg.custom_tickers}
    else:
        raise ValueError(f"Unknown universe: {cfg.universe}")


# ───────────────────────────────────────────────────
#  PRICE DATA (Yahoo Finance)
# ───────────────────────────────────────────────────


def _download_yahoo_csv(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Fallback CSV download when yfinance is rate-limited."""
    p1 = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
    p2 = int(datetime.strptime(end, "%Y-%m-%d").timestamp())
    for base in ("query1", "query2"):
        try:
            url = (
                f"https://{base}.finance.yahoo.com/v7/finance/download/{ticker}"
                f"?period1={p1}&period2={p2}&interval=1d&events=history&includeAdjustedClose=true"
            )
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code == 200 and "Date" in resp.text[:50]:
                return pd.read_csv(io.StringIO(resp.text), parse_dates=["Date"], index_col="Date")
        except Exception:
            pass
    return None


def fetch_prices(
    tickers: list[str],
    lookback_days: int = 400,
    batch_pause: float = 0.3,
    big_pause: float = 2.0,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV for *tickers*. Returns {col: DataFrame} with Close, Volume, High, Low."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    start_str, end_str = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    all_close, all_volume, all_high, all_low = {}, {}, {}, {}
    failed: list[str] = []

    # probe yfinance
    yf_works = True
    try:
        probe = yf.download("SPY", period="5d", progress=False, timeout=10)
        if probe is None or probe.empty:
            yf_works = False
    except Exception:
        yf_works = False

    t0 = time.time()
    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed else 1
            eta = (len(tickers) - i) / rate
            logger.info("Prices %d/%d  loaded=%d  eta=%.0fs", i + 1, len(tickers), len(all_close), eta)

        df = None
        if yf_works:
            try:
                raw = yf.download(ticker, start=start_str, end=end_str, progress=False, timeout=10)
                if raw is not None and not raw.empty:
                    if isinstance(raw.columns, pd.MultiIndex):
                        lvl0 = raw.columns.get_level_values(0)
                        df = pd.DataFrame(
                            {c: raw[c].iloc[:, 0] for c in ("Close", "Volume", "High", "Low") if c in lvl0}
                        )
                    else:
                        need = [c for c in ("Close", "Volume", "High", "Low") if c in raw.columns]
                        df = raw[need].copy() if need else None
            except Exception as exc:
                if "Rate" in str(exc) or "429" in str(exc):
                    yf_works = False
                    logger.warning("Rate-limited at %d, switching to CSV fallback", i + 1)

        if df is None:
            try:
                raw = _download_yahoo_csv(ticker, start_str, end_str)
                if raw is not None:
                    cols: dict[str, pd.Series] = {}
                    for c in ("Close", "Adj Close"):
                        if c in raw.columns:
                            cols["Close"] = raw[c]
                            break
                    for c in ("Volume", "High", "Low"):
                        if c in raw.columns:
                            cols[c] = raw[c]
                    if "Close" in cols:
                        df = pd.DataFrame(cols)
            except Exception:
                pass

        if df is not None and "Close" in df.columns and len(df) > 100:
            all_close[ticker] = df["Close"].dropna()
            if "Volume" in df:
                all_volume[ticker] = df["Volume"].dropna()
            if "High" in df:
                all_high[ticker] = df["High"].dropna()
            if "Low" in df:
                all_low[ticker] = df["Low"].dropna()
        else:
            failed.append(ticker)

        if (i + 1) % 5 == 0:
            time.sleep(batch_pause)
        if (i + 1) % 100 == 0:
            time.sleep(big_pause)

    logger.info("Loaded %d tickers in %.0fs  (failed=%d)", len(all_close), time.time() - t0, len(failed))
    return {
        "close": pd.DataFrame(all_close),
        "volume": pd.DataFrame(all_volume),
        "high": pd.DataFrame(all_high),
        "low": pd.DataFrame(all_low),
        "failed": failed,
    }


# ───────────────────────────────────────────────────
#  FINNHUB — EARNINGS + INSIDER SENTIMENT
# ───────────────────────────────────────────────────


def fetch_earnings(
    tickers: list[str],
    finnhub_key: str,
    surprise_min: float = 5.0,
    lookback_days: int = 60,
    danger_days: int = 14,
    calls_per_min: int = 55,
) -> dict[str, dict]:
    """Fetch earnings surprises and upcoming-earnings flags from Finnhub."""
    base = "https://finnhub.io/api/v1"
    today = datetime.now()
    earnings_info: dict[str, dict] = {}

    # Upcoming earnings calendar
    upcoming_set: set[str] = set()
    try:
        from_d = today.strftime("%Y-%m-%d")
        to_d = (today + timedelta(days=danger_days)).strftime("%Y-%m-%d")
        url = f"{base}/calendar/earnings?from={from_d}&to={to_d}&token={finnhub_key}"
        data = requests.get(url, timeout=10).json()
        if isinstance(data, dict) and "earningsCalendar" in data:
            upcoming_set = {e["symbol"] for e in data["earningsCalendar"] if "symbol" in e}
    except Exception as exc:
        logger.warning("Earnings calendar failed: %s", exc)

    call_count = 0
    for i, ticker in enumerate(tickers):
        if call_count > 0 and call_count % calls_per_min == 0:
            logger.debug("Finnhub rate-limit pause")
            time.sleep(10)

        try:
            url = f"{base}/stock/earnings?symbol={ticker}&limit=4&token={finnhub_key}"
            resp = requests.get(url, timeout=10)
            call_count += 1

            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    latest = data[0]
                    actual = latest.get("actual")
                    estimate = latest.get("estimate")
                    surprise_pct = latest.get("surprisePercent")
                    period = latest.get("period", "")

                    if surprise_pct is None and actual is not None and estimate and estimate != 0:
                        surprise_pct = ((actual - estimate) / abs(estimate)) * 100

                    days_since = None
                    if period:
                        try:
                            rd = datetime.strptime(str(period)[:10], "%Y-%m-%d")
                            days_since = (today - rd).days
                        except ValueError:
                            pass

                    beat = surprise_pct is not None and surprise_pct >= surprise_min
                    recent = beat and days_since is not None and days_since <= lookback_days
                    danger = ticker in upcoming_set

                    earnings_info[ticker] = {
                        "surprise_pct": round(surprise_pct, 2) if surprise_pct is not None else None,
                        "actual_eps": actual,
                        "estimated_eps": estimate,
                        "beat": beat,
                        "days_since_earnings": days_since,
                        "recent_beat": recent,
                        "earnings_danger": danger,
                        "next_earnings_days": danger_days if danger else None,
                    }
                    continue

            elif resp.status_code == 429:
                logger.warning("Finnhub 429 at ticker %d, pausing 60s", i)
                time.sleep(60)
        except Exception:
            pass

        # Default for missing
        earnings_info.setdefault(ticker, {
            "surprise_pct": None, "beat": False, "days_since_earnings": None,
            "recent_beat": False, "earnings_danger": ticker in upcoming_set,
            "next_earnings_days": danger_days if ticker in upcoming_set else None,
        })
        time.sleep(1.1)

    # Fill any still missing
    for ticker in tickers:
        earnings_info.setdefault(ticker, {
            "surprise_pct": None, "beat": False, "days_since_earnings": None,
            "recent_beat": False, "earnings_danger": ticker in upcoming_set,
            "next_earnings_days": danger_days if ticker in upcoming_set else None,
        })

    return earnings_info


def fetch_insider_sentiment(
    tickers: list[str],
    finnhub_key: str,
    lookback_days: int = 90,
    min_buyers: int = 2,
    calls_per_min: int = 55,
) -> dict[str, dict]:
    """Fetch monthly insider-sentiment (MSPR) data from Finnhub."""
    base = "https://finnhub.io/api/v1"
    today = datetime.now()
    from_date = (today - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    insider_info: dict[str, dict] = {}

    call_count = 0
    for i, ticker in enumerate(tickers):
        if call_count > 0 and call_count % calls_per_min == 0:
            time.sleep(10)

        try:
            url = (
                f"{base}/stock/insider-sentiment?symbol={ticker}"
                f"&from={from_date}&token={finnhub_key}"
            )
            resp = requests.get(url, timeout=10)
            call_count += 1

            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and "data" in data and len(data["data"]) > 0:
                    months = data["data"]
                    avg_mspr = float(np.mean([m.get("mspr", 0) for m in months]))
                    total_change = sum(m.get("change", 0) for m in months)
                    buy_months = sum(1 for m in months if m.get("mspr", 0) > 0)

                    insider_info[ticker] = {
                        "num_buyers": buy_months,
                        "num_transactions": len(months),
                        "total_value": round(total_change, 2),
                        "avg_mspr": round(avg_mspr, 2),
                        "cluster": buy_months >= min_buyers,
                    }
                    continue

            elif resp.status_code == 429:
                logger.warning("Finnhub 429 at ticker %d, pausing 60s", i)
                time.sleep(60)
        except Exception:
            pass

        insider_info.setdefault(ticker, {
            "num_buyers": 0, "num_transactions": 0,
            "total_value": 0, "avg_mspr": 0, "cluster": False,
        })
        time.sleep(1.1)

    for ticker in tickers:
        insider_info.setdefault(ticker, {
            "num_buyers": 0, "num_transactions": 0,
            "total_value": 0, "avg_mspr": 0, "cluster": False,
        })

    return insider_info
