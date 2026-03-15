"""
Alpha Sniper v2 — FastAPI application entry point.

Run locally:
    uvicorn src.main:app --reload --port 8000

Run via Docker:
    docker compose up --build
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config import get_settings


def _configure_logging() -> None:
    cfg = get_settings()
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    _configure_logging()
    logger = logging.getLogger(__name__)
    cfg = get_settings()
    logger.info(
        "%s v%s starting  (universe=%s, debug=%s)",
        cfg.app_name, cfg.app_version, cfg.universe, cfg.debug,
    )
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Alpha Sniper",
    description=(
        "ML-driven stock scanner using a 10-layer signal engine "
        "(RSI reversal, MACD crossover, insider sentiment, earnings drift, "
        "volume accumulation, and more)."
    ),
    version=get_settings().app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
