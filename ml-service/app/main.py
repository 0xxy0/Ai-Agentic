"""
FastAPI application entry point.

Responsibilities:
- Create and configure the FastAPI app.
- Register routers.
- Configure CORS for Node.js / browser clients.
- Load the ML model at startup (singleton).
- Expose lifespan events for clean startup/shutdown.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import health, predict, train
from app.api import report, forecast
from app.core.config import settings
from app.core.logger import get_logger
from app.models.loader import load_model

logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI lifespan context manager.

    - On startup: pre-load the MLflow model into the in-process cache.
    - On shutdown: log graceful exit.
    """
    logger.info(
        "Starting up",
        extra={"service": settings.APP_NAME, "version": settings.APP_VERSION},
    )

    try:
        load_model()
        logger.info("Model loaded and cached at startup")
    except RuntimeError as exc:
        # Allow app to start even if model isn't available yet (useful in CI)
        logger.warning(
            "Model not loaded at startup — inference will fail until model is available",
            extra={"error": str(exc)},
        )

    yield  # App is running

    logger.info("Shutting down", extra={"service": settings.APP_NAME})


# ── App factory ───────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Production-ready Churn Prediction ML Microservice with Multi-Agent Pipeline. "
            "Supports single inference, batch scoring, model training, PDF reports, and dashboard JSON via REST."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(train.router)
    app.include_router(report.router)
    app.include_router(forecast.router)

    # ── Global exception handler ──────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled exception",
            extra={"path": str(request.url), "error": str(exc)},
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected error occurred. Check server logs."},
        )

    logger.info("FastAPI app created", extra={"routes": [r.path for r in app.routes]})
    return app


# ── Entry point ───────────────────────────────────────────────────────────────

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.PYTHON_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
