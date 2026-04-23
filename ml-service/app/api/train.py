"""
Training API endpoint.

POST /train — Triggers model training from a CSV file path or uploaded data.
This endpoint is intended for MLOps operators, not end-users; protect it
behind authentication middleware in production.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from app.core.logger import get_logger
from app.services.training_service import training_service

logger = get_logger(__name__)

router = APIRouter(prefix="/train", tags=["Training"])


class TrainRequest(BaseModel):
    """Training trigger request body."""

    csv_path: Optional[str] = Field(
        None,
        description="Absolute path to training CSV. Defaults to settings.BATCH_INPUT_PATH.",
    )
    churn_window_days: int = Field(
        30,
        description="Churn definition window: 30 | 60 | 90 days.",
    )
    xgb_params: Optional[dict[str, Any]] = Field(
        None,
        description="XGBoost hyperparameter overrides.",
    )
    register_model: bool = Field(
        True,
        description="If true, register trained model in MLflow Model Registry.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "churn_window_days": 30,
                "register_model": True,
            }
        }
    }


class TrainResponse(BaseModel):
    """Training job response."""

    status: str
    run_id: Optional[str] = None
    n_samples: Optional[int] = None
    churn_window_days: Optional[int] = None
    message: Optional[str] = None


@router.post("", response_model=TrainResponse, summary="Trigger model training")
def trigger_training(body: TrainRequest) -> TrainResponse:
    """
    Train a new churn model from CSV data.

    Synchronous execution — the response is returned after training completes.
    For long-running jobs consider the dedicated training pipeline script.
    """
    logger.info(
        "Training request received",
        extra={
            "csv_path": body.csv_path,
            "churn_window_days": body.churn_window_days,
            "register_model": body.register_model,
        },
    )

    try:
        result = training_service.train_from_csv(
            csv_path=body.csv_path,
            window_days=body.churn_window_days,
            xgb_params=body.xgb_params,
        )
        return TrainResponse(**result)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Training failed", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail=f"Training error: {exc}") from exc
