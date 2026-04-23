"""
SHAP-based model explainability module.

Provides feature-level explanations for individual predictions.
Kept optional — if SHAP is unavailable, explanations are skipped gracefully.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.core.logger import get_logger

logger = get_logger(__name__)


def explain_prediction(
    model_unwrapped: Any, feature_df: pd.DataFrame
) -> dict[str, float]:
    """
    Compute SHAP values for a single prediction row.

    Args:
        model_unwrapped: The underlying XGBoost booster (not the pyfunc wrapper).
        feature_df: Single-row DataFrame with engineered features.

    Returns:
        Dict mapping feature name → SHAP value. Empty dict if SHAP unavailable.
    """
    try:
        import shap  # Lazy import — SHAP is an optional dependency

        explainer = shap.TreeExplainer(model_unwrapped)
        shap_values = explainer.shap_values(feature_df)

        # For binary classification, shap_values is (n_rows, n_features)
        values = shap_values[0] if len(shap_values) > 1 else shap_values[0]
        explanation = {
            col: round(float(v), 6)
            for col, v in zip(feature_df.columns, values)
        }

        logger.debug("SHAP explanation computed", extra={"explanation": explanation})
        return explanation

    except ImportError:
        logger.warning("SHAP not installed — skipping explanation")
        return {}
    except Exception as exc:
        logger.error("SHAP explanation failed", extra={"error": str(exc)})
        return {}
