"""
Forecast API endpoints.
"""

from __future__ import annotations
from typing import Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.forecast_service import forecast_service
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/forecast", tags=["Quarterly Forecast"])

class ForecastItem(BaseModel):
    customer_id: str
    name: str
    segment: str
    subscription_plan: str
    current_status: str
    predicted_churn_probability: str
    risk_level: str
    predicted_revenue_loss: str
    last_active_date: str
    forecast_month: str
    recommended_action: str

@router.get("", response_model=list[ForecastItem], summary="Get quarterly forecast predictions")
def get_forecasts(month: Optional[str] = Query(None, description="Filter by forecast month (e.g., 'Q3 2025')")) -> Any:
    """
    Retrieve quarterly forecast predictions from MongoDB.
    """
    try:
        results = forecast_service.get_all_forecasts(month)
        if not results:
            return []
        return results
    except Exception as exc:
        logger.error("Forecast endpoint error", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail="Internal forecast retrieval error")

@router.post("/refresh", summary="Manually refresh forecast data from default CSV")
def refresh_forecasts() -> dict[str, str]:
    """
    Reload the forecast data from the raw predictions CSV into MongoDB.
    """
    try:
        import os
        from pathlib import Path
        csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "quarterly_forecast_raw_predictions.csv"
        
        if not csv_path.exists():
            raise HTTPException(status_code=404, detail="Raw predictions CSV not found")
            
        forecast_service.upload_forecast_csv(str(csv_path))
        return {"status": "success", "message": "Forecast data refreshed from CSV"}
    except Exception as exc:
        logger.error("Forecast refresh error", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail=str(exc))
