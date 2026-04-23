"""
Prediction Agent — Phase 4 of the multi-agent pipeline.

Responsibility:
    Run batch inference using the trained (or MLflow-loaded) model,
    assign risk levels, and enrich the context with a scored DataFrame.

Input context keys consumed:
    ``model``           — trained XGBoost model (or None → load from MLflow)
    ``feature_df``      — flat feature matrix (index = user_id)
    ``feature_columns`` — ordered list of model input column names

Output context keys added:
    ``scored_df``       — DataFrame: user_id | churn_score | risk_level
    ``high_risk_users`` — list of user_ids with risk_level == HIGH
    ``summary_stats``   — dict of aggregate prediction statistics

Contract:
    The agent ONLY produces predictions. It does NOT train or persist models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.logger import get_logger
from app.models.loader import load_model  # reuse existing singleton loader

logger = get_logger(__name__)


def _assign_risk(score: float) -> str:
    """Map churn probability to a risk label."""
    if score >= 0.75:
        return "HIGH"
    if score >= 0.50:
        return "MEDIUM"
    return "LOW"


class PredictionAgent:
    """
    Agent 4/5: Scores all users and assigns risk levels.

    If ``context["model"]`` is present (e.g., just trained), it is used
    directly.  Otherwise the agent loads the Production model from the
    MLflow Model Registry.

    Usage::

        agent = PredictionAgent()
        ctx   = agent.run(ctx)
        # ctx["scored_df"]  → DataFrame with churn_score + risk_level

    """

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute batch scoring.

        Args:
            context: Pipeline context containing ``feature_df``
                     (and optionally ``model``).

        Returns:
            Enriched context with ``scored_df``, ``high_risk_users``,
            and ``summary_stats``.

        Raises:
            KeyError: If ``feature_df`` is absent.
            RuntimeError: If model loading fails.
        """
        logger.info("PredictionAgent started")

        feature_df: pd.DataFrame = context.get("feature_df")
        feat_cols: list[str]     = context.get("feature_columns", [])

        if feature_df is None:
            raise KeyError("PredictionAgent requires 'feature_df' in context.")

        # ── Resolve model ────────────────────────────────────────────────────
        model = context.get("model")
        if model is None:
            logger.info("No in-context model — loading from MLflow registry")
            mlflow_model = load_model()
            # Extract the raw XGBoost classifier from the pyfunc wrapper
            model = mlflow_model

        X = feature_df[feat_cols] if feat_cols else feature_df

        # ── Predict ───────────────────────────────────────────────────────────
        if hasattr(model, "predict_proba"):
            # Raw XGBClassifier (from ModelingAgent)
            proba = model.predict_proba(X)[:, 1]
        else:
            # MLflow pyfunc wrapper
            raw_preds = model.predict(X)
            if hasattr(raw_preds, "shape") and len(raw_preds.shape) == 2:
                proba = raw_preds[:, 1]
            else:
                proba = raw_preds

        proba = np.clip(proba, 0.0, 1.0)

        # ── Build scored DataFrame ────────────────────────────────────────────
        scored_df = pd.DataFrame({
            "user_id":     feature_df.index,
            "churn_score": np.round(proba, 6),
            "risk_level":  [_assign_risk(float(s)) for s in proba],
        }).reset_index(drop=True)

        # ── Summary stats ─────────────────────────────────────────────────────
        risk_counts = scored_df["risk_level"].value_counts().to_dict()
        summary: dict[str, Any] = {
            "total_users":       len(scored_df),
            "high_risk_count":   risk_counts.get("HIGH",   0),
            "medium_risk_count": risk_counts.get("MEDIUM", 0),
            "low_risk_count":    risk_counts.get("LOW",    0),
            "avg_churn_score":   round(float(proba.mean()), 4),
            "max_churn_score":   round(float(proba.max()),  4),
            "min_churn_score":   round(float(proba.min()),  4),
        }

        context["scored_df"]       = scored_df
        context["high_risk_users"] = scored_df.loc[
            scored_df["risk_level"] == "HIGH", "user_id"
        ].tolist()
        context["summary_stats"]   = summary

        logger.info("PredictionAgent complete", extra=summary)
        return context
