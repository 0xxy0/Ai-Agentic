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


def load_model(force_reload: bool = False) -> mlflow.pyfunc.PyFuncModel:
    """
    Load the churn model from the MLflow Model Registry.

    The model is cached in-process after the first successful load.
    Set ``force_reload=True`` to bypass the cache (e.g., after promotion).

    Args:
        force_reload: If True, discard the cached model and reload.

    Returns:
        A loaded :class:`mlflow.pyfunc.PyFuncModel` instance.

    Raises:
        RuntimeError: If the model cannot be loaded from MLflow.
    """
    global _model_cache

    if _model_cache is not None and not force_reload:
        logger.debug("Returning cached model")
        return _model_cache

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    model_uri = (
        settings.MODEL_URI
        or f"models:/{settings.MLFLOW_MODEL_NAME}/{settings.MLFLOW_MODEL_STAGE}"
    )

    logger.info("Loading model from MLflow", extra={"model_uri": model_uri})

    try:
        # Fix for Windows paths in MLflow
        if ":" in model_uri and not model_uri.startswith(("models:/", "runs:/", "file://", "http://", "https://")):
            clean_path = model_uri.replace("\\", "/")
            model_uri = f"file:///{clean_path}"
            
        _model_cache = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded successfully from MLflow", extra={"model_uri": model_uri})
    except Exception as exc:
        logger.warning(
            "Failed to load model from MLflow. Checking local fallback...",
            extra={"model_uri": model_uri, "error": str(exc)},
        )
        
        # Local fallback using native XGBoost
        import os
        import xgboost as xgb
        local_path = "data/model.json"
        
        if os.path.exists(local_path):
            logger.info("Loading model from local fallback", extra={"path": local_path})
            _model_cache = xgb.XGBClassifier()
            _model_cache.load_model(local_path)
        else:
            raise RuntimeError(
                f"Could not load model from '{model_uri}' and no local fallback found at '{local_path}'."
            ) from exc

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
