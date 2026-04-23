"""
MLflow model loader.

Loads the registered ChurnModel from the MLflow Model Registry.
The model is loaded ONCE at application startup and cached.
"""

from __future__ import annotations

from typing import Any, Optional

import mlflow
import mlflow.pyfunc

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# Module-level singleton — populated on first call to load_model()
_model_cache: Optional[mlflow.pyfunc.PyFuncModel] = None


def load_model(force_reload: bool = False) -> Any:
    """
    Load the upgraded churn model (v2.0) or fallback to MLflow.
    """
    global _model_cache

    if _model_cache is not None and not force_reload:
        return _model_cache

    # 1. Try Loading Upgraded v2.0 Model (Joblib)
    import joblib
    from pathlib import Path
    v2_path = Path("ml_pipeline/outputs/xgb_model.pkl")
    
    if v2_path.exists():
        try:
            logger.info(f"Loading upgraded Churn v2.0 model from {v2_path}")
            _model_cache = joblib.load(v2_path)
            return _model_cache
        except Exception as e:
            logger.warning(f"Failed to load v2.0 pkl model: {e}")

    # 2. Fallback to MLflow logic
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    model_uri = settings.MODEL_URI or f"models:/{settings.MLFLOW_MODEL_NAME}/{settings.MLFLOW_MODEL_STAGE}"

    try:
        _model_cache = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded successfully from MLflow")
    except Exception as exc:
        logger.warning(f"MLflow load failed. Checking model.pkl fallback...")
        local_path = "ml_pipeline/outputs/xgb_model.pkl" # Double check
        if Path(local_path).exists():
            _model_cache = joblib.load(local_path)
        else:
            raise RuntimeError(f"No model found at {v2_path}") from exc

    return _model_cache


def get_model_metadata() -> dict[str, Any]:
    """
    Return metadata about the currently loaded model.

    Returns:
        A dictionary with model URI, run ID, and flavors.
    """
    model = load_model()
    meta = model.metadata
    return {
        "model_uri": meta.model_uri if hasattr(meta, "model_uri") else "unknown",
        "run_id": meta.run_id if hasattr(meta, "run_id") else "unknown",
        "flavors": list(meta.flavors.keys()) if meta.flavors else [],
        "mlflow_version": meta.mlflow_version if hasattr(meta, "mlflow_version") else "unknown",
    }
