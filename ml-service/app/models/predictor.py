"""
Predictor module.

Bridges the raw feature dict → model inference → structured output.
Keeps all ML logic isolated from API/business layers.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.core.config import settings
from app.core.logger import get_logger
from app.features.builder import FEATURE_COLUMNS, build_single_feature_dict
from app.models.loader import load_model

logger = get_logger(__name__)


def _risk_level(score: float) -> str:
    """Map a churn probability score to a risk label."""
    if score >= settings.RISK_HIGH_THRESHOLD:
        return "HIGH"
    if score >= settings.RISK_LOW_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def predict_single(raw_features: dict[str, Any]) -> dict[str, Any]:
    """
    Run inference on a single user's raw activity record.

    Steps:
    1. Engineer features from raw dict.
    2. Wrap in a one-row DataFrame (preserves column order).
    3. Call the MLflow model's ``predict()`` method.
    4. Return churn_score + risk_level.

    Args:
        raw_features: Validated raw activity dict.

    Returns:
        ``{"churn_score": float, "risk_level": str}``

    Raises:
        RuntimeError: If inference fails.
    """
    try:
        feature_dict = build_single_feature_dict(raw_features)
        df = pd.DataFrame([feature_dict])[FEATURE_COLUMNS]

        model = load_model()
        
        # Use predict_proba for granular scores if available (native XGB or pyfunc)
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(df)
        else:
            preds = model.predict(df)

        # Handle different response shapes from MLflow pyfunc vs native XGB
        if hasattr(preds, "shape") and len(preds.shape) == 2:
            churn_score = float(preds[0][1])
        else:
            churn_score = float(preds[0])

        # Clamp to [0, 1] defensively
        churn_score = max(0.0, min(1.0, churn_score))
        risk = _risk_level(churn_score)

        logger.info(
            "Prediction complete",
            extra={
                "user_id": raw_features.get("user_id", "unknown"),
                "churn_score": churn_score,
                "risk_level": risk,
            },
        )

        return {
            "churn_score": round(churn_score, 6),
            "risk_level": risk,
        }

    except Exception as exc:
        import traceback
        logger.error(
            "Prediction failed",
            extra={
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "input_features": list(feature_dict.keys()) if 'feature_dict' in locals() else "none"
            }
        )
        raise RuntimeError(f"Inference error: {exc}") from exc


def predict_batch(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Run inference on a pre-engineered feature DataFrame.

    Args:
        df: DataFrame with columns == :data:`~app.features.builder.FEATURE_COLUMNS`.

    Returns:
        List of ``{"churn_score": float, "risk_level": str}`` dicts
        (same order as input rows).
    """
    try:
        model = load_model()
        preds = model.predict(df[FEATURE_COLUMNS])

        results: list[dict[str, Any]] = []
        for i in range(len(preds)):
            if hasattr(preds, "shape") and len(preds.shape) == 2:
                score = float(preds[i][1])
            else:
                score = float(preds[i])
            score = max(0.0, min(1.0, score))
            results.append({
                "churn_score": round(score, 6),
                "risk_level": _risk_level(score),
            })

        logger.info(
            "Batch prediction complete",
            extra={"n_predictions": len(results)},
        )
        return results

    except Exception as exc:
        logger.error("Batch prediction failed", extra={"error": str(exc)})
        raise RuntimeError(f"Batch inference error: {exc}") from exc
