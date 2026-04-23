"""
Health check endpoint.

GET /health — Returns service status, model load state, and version info.
Used by load balancers, k8s liveness/readiness probes, and Node.js consumers.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Health"])

# Track startup time for uptime reporting
_START_TIME: datetime = datetime.now(tz=timezone.utc)


@router.get("/health", summary="Service health check")
def health_check() -> dict:
    """
    Liveness probe endpoint.

    Returns:
        JSON with status, version, uptime, and model registry URI.
    """
    uptime_seconds = (datetime.now(tz=timezone.utc) - _START_TIME).total_seconds()

    payload = {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENV,
        "uptime_seconds": round(uptime_seconds, 2),
        "mlflow_tracking_uri": settings.MLFLOW_TRACKING_URI,
        "model_registry": f"models:/{settings.MLFLOW_MODEL_NAME}/{settings.MLFLOW_MODEL_STAGE}",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }

    logger.debug("Health check called", extra=payload)
    return payload
