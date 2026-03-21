# Alpha Sniper v2

[![Watch the Demo](https://img.shields.io/badge/YouTube-Watch%20the%20Demo-red?style=for-the-badge&logo=youtube)](https://youtu.be/LGFK91Fcq1w)

**ML-driven stock scanner** that identifies high-probability reversal setups using a 10-layer signal engine across 900+ equities.

Built with **FastAPI · Docker · Python 3.11 · scikit-learn** — designed as a production-grade demonstration of end-to-end ML pipeline engineering.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI (uvicorn)                       │
│  POST /api/v1/scan     GET /api/v1/stock/{ticker}           │
│  GET  /api/v1/health                                        │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   ┌────────────┐ ┌────────────┐ ┌────────────┐
   │  Universe  │ │   Price    │ │  Finnhub   │
   │  Builder   │ │  Fetcher   │ │  Client    │
   │  (FMP/Wiki)│ │  (Yahoo)   │ │ (Earn/Ins) │
   └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
         └───────────────┼──────────────┘
                         ▼
              ┌─────────────────────┐
              │  Feature Engineering │
              │  RSI · MACD · MA    │
              │  Sector Regimes     │
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  10-Layer Scanner   │
              │  Score · Rank · Tag │
              │  (STRONG BUY / BUY  │
              │   / WATCH / NEUTRAL)│
              └─────────────────────┘
```

## Signal Layers

| # | Layer | Max Pts | Source |
|---|-------|---------|--------|
| 1 | Proximity to 52-week low | 20 | Yahoo Finance |
| 2 | RSI reversal (oversold → recovery) | 20 | Computed |
| 3 | Higher low formation | 15 | Computed |
| 4 | Volume spike / accumulation | 15 | Yahoo Finance |
| 5 | Moving-average reclaim (10/20 MA) | 15 | Computed |
| 6 | MACD crossover / momentum | 15 | Computed |
| 7 | Earnings surprise / PEAD | 15 | Finnhub |
| 8 | Insider buying cluster (MSPR) | 10 | Finnhub |
| 9 | Upcoming earnings penalty | −10 | Finnhub |
| 10 | Sector regime | ±5 | Computed |

**Max composite score: 130.** Signals require ≥45 pts and ≥3 confirmed layers.

---

## Quick Start

### 1. Clone & Configure

```bash
git clone https://github.com/YOUR_USERNAME/alpha-sniper.git
cd alpha-sniper
cp .env.example .env
# Edit .env → add your FMP_API_KEY and FINNHUB_API_KEY
```

### 2. Run with Docker

```bash
docker compose up --build
```

The API will be live at **http://localhost:8000**.

### 3. Run Locally (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8000
```

### 4. Run Tests

```bash
pytest tests/ -v
```

---

## API Usage

### Health Check
```bash
curl http://localhost:8000/api/v1/health
```

### Full Scan
```bash
curl -X POST http://localhost:8000/api/v1/scan \
  -H "Content-Type: application/json" \
  -d '{"limit": 10, "universe": "sp500"}'
```

### Single Stock Analysis
```bash
curl http://localhost:8000/api/v1/stock/AAPL
```

### Interactive Docs
Open **http://localhost:8000/docs** for the auto-generated Swagger UI.

---

## Project Structure

```
alpha-sniper/
├── Dockerfile              # Multi-stage build, non-root, health check
├── docker-compose.yml      # One-command deployment
├── requirements.txt
├── .env.example            # API key template
├── src/
│   ├── main.py             # FastAPI app + lifespan hooks
│   ├── config.py           # Pydantic settings from env vars
│   ├── api/
│   │   └── routes.py       # REST endpoints (/scan, /stock, /health)
│   ├── models/
│   │   └── scanner.py      # 10-layer signal engine
│   ├── data/
│   │   └── fetcher.py      # Yahoo, FMP, Finnhub data fetching
│   └── features/
│       └── engineering.py   # RSI, MACD, sector trend computation
├── tests/
│   └── test_scanner.py     # Unit + integration tests
└── notebooks/
    └── exploration.ipynb   # Original Jupyter notebook (reference)
```

---

## Key Design Decisions

- **No hardcoded API keys** — everything via env vars / `.env`
- **Rate-limit awareness** — Finnhub calls auto-pause at 55/min; Yahoo falls back to CSV scraping on 429
- **Structured logging** — no `print()` calls; all output goes through `logging`
- **Type-hinted throughout** — every function signature has full annotations
- **Pydantic v2** for config and API schemas — validated, serialisable, documented

---

## Roadmap

- [ ] Background task queue (Celery / ARQ) for async scans
- [ ] Redis caching for price data (avoid redundant Yahoo calls)
- [ ] WebSocket endpoint for real-time scan progress
- [ ] GMM / K-Means clustering layer (scikit-learn) for regime detection
- [ ] Backtest endpoint (`POST /api/v1/backtest`)
- [ ] GitHub Actions CI pipeline

---

## License

MIT
