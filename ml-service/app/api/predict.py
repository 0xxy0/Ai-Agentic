"""
Prediction API endpoints.

POST /predict      — Single-user churn prediction.
POST /predict/batch — Batch churn prediction (up to 1000 users per call).
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logger import get_logger
from app.features.validators import BatchScoringRequest, RawUserActivity
from app.services.inference_service import inference_service

logger = get_logger(__name__)

router = APIRouter(prefix="/predict", tags=["Prediction"])


def _to_predict_response(result: dict[str, Any], raw: dict[str, Any]) -> PredictResponse:
    """Normalize inference output into the public API schema."""
    return PredictResponse(
        customer_id=(
            result.get("customer_id")
            or raw.get("customer_id")
            or raw.get("user_id")
            or "unknown"
        ),
        name=raw.get("name"),
        segment=result.get("segment") or raw.get("segment"),
        subscription_plan=raw.get("subscription_plan"),
        current_status=raw.get("current_status"),
        churn_probability=float(result.get("churn_score", 0.0)),
        risk_level=result.get("risk_level", raw.get("risk_level") or "LOW"),
        predicted_revenue_loss=raw.get("predicted_revenue_loss"),
        last_active_date=raw.get("last_active_date"),
        forecast_month=raw.get("forecast_month"),
        recommended_action=result.get("recommended_action") or raw.get("recommended_action"),
        source=result.get("source", "ML-Engine-v2"),
    )

# ── Response schemas ──────────────────────────────────────────────────────────


class PredictResponse(BaseModel):
    """Churn prediction output with enterprise metadata."""

    customer_id: str
    name: Optional[str] = None
    segment: Optional[str] = None
    subscription_plan: Optional[str] = None
    current_status: Optional[str] = None
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")
    predicted_revenue_loss: Optional[str] = None
    last_active_date: Optional[str] = None
    forecast_month: Optional[str] = None
    recommended_action: Optional[str] = None
    source: str = "ML-Engine-v2"

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": "C-001",
                "churn_probability": 0.78,
                "risk_level": "HIGH",
                "predicted_revenue_loss": "$1,200",
                "recommended_action": "Incentivize renewal with 20% discount"
            }
        }
    }


class BatchPredictResponse(BaseModel):
    """Batch prediction response envelope."""

    predictions: list[PredictResponse]
    total: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post("", response_model=PredictResponse, summary="Single-user churn prediction")
def predict_single(body: RawUserActivity) -> PredictResponse:
    """
    Predict churn probability for a single user.

    **Request body**: A `RawUserActivity` JSON object.
    **Response**: `churn_score` (float) + `risk_level` (LOW/MEDIUM/HIGH).

    Node.js compatible — pure JSON in/out.
    """
    try:
        raw = body.model_dump()
        result = inference_service.predict_one(raw)
        return _to_predict_response(result, raw)
    except Exception as exc:
        logger.error("Prediction endpoint error", extra={"error": str(exc)})
        raw = body.model_dump()
        return _to_predict_response(
            {
                "customer_id": raw.get("customer_id") or raw.get("user_id") or "unknown",
                "churn_score": 0.0,
                "risk_level": "LOW",
                "recommended_action": "Manual review required.",
                "source": "Prediction-API-Fallback",
            },
            raw,
        )


@router.post(
    "/batch",
    response_model=BatchPredictResponse,
    summary="Batch churn prediction",
)
def predict_batch_endpoint(body: BatchScoringRequest) -> BatchPredictResponse:
    """
    Score a batch of users in a single API call.

    Accepts up to **1 000 users** per request.  For larger datasets use the
    standalone batch scoring pipeline (`pipelines/batch_scoring.py`).
    """
    if len(body.users) > 1000:
        raise HTTPException(
            status_code=422,
            detail="Batch size exceeds 1000. Use the batch scoring pipeline for larger datasets.",
        )

    try:
        records = [u.model_dump() for u in body.users]
        results = inference_service.predict_many(records)

        predictions = [_to_predict_response(r, raw) for r, raw in zip(results, records)]
        high   = sum(1 for p in predictions if p.risk_level == "HIGH")
        medium = sum(1 for p in predictions if p.risk_level == "MEDIUM")
        low    = sum(1 for p in predictions if p.risk_level == "LOW")

        return BatchPredictResponse(
            predictions=predictions,
            total=len(predictions),
            high_risk_count=high,
            medium_risk_count=medium,
            low_risk_count=low,
        )
    except Exception as exc:
        logger.error("Batch prediction endpoint error", extra={"error": str(exc)})
        fallback_predictions = [
            _to_predict_response(
                {
                    "customer_id": raw.get("customer_id") or raw.get("user_id") or "unknown",
                    "churn_score": 0.0,
                    "risk_level": "LOW",
                    "recommended_action": "Manual review required.",
                    "source": "Prediction-API-Fallback",
                },
                raw,
            )
            for raw in [u.model_dump() for u in body.users]
        ]
        return BatchPredictResponse(
            predictions=fallback_predictions,
            total=len(fallback_predictions),
            high_risk_count=0,
            medium_risk_count=0,
            low_risk_count=len(fallback_predictions),
        )
