"""
Forecast service — Churn Intelligence v2.0
==========================================
Handles retrieval and management of quarterly forecast data from MongoDB.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any
import pandas as pd

# Add ml_pipeline to path for core modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ml_pipeline"))

from db.mongo import load_forecasts, save_forecasts
from app.core.logger import get_logger

logger = get_logger(__name__)

class ForecastService:
    def get_all_forecasts(self, month: str = None) -> list[dict[str, Any]]:
        """Retrieve forecast predictions from MongoDB."""
        try:
            df = load_forecasts(month)
            if df.empty:
                return []
            
            # Remove MongoDB internal _id
            if "_id" in df.columns:
                df = df.drop(columns=["_id"])
            
            # Convert timestamp to string if it exists
            if "timestamp" in df.columns:
                df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
                
            return df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Failed to load forecasts: {str(e)}")
            return []

    def upload_forecast_csv(self, file_path: str):
        """Upload a CSV file containing forecast predictions."""
        try:
            df = pd.read_csv(file_path)
            save_forecasts(df)
            logger.info(f"Successfully uploaded forecast data from {file_path}")
        except Exception as e:
            logger.error(f"Failed to upload forecast CSV: {str(e)}")
            raise

forecast_service = ForecastService()
