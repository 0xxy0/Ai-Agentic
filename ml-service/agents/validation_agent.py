"""
Validation Agent — Phase 5 of the multi-agent pipeline.

Responsibility:
    Evaluate model quality using held-out labels or proxy metrics,
    generate a full evaluation report, and flag quality gates.

Input context keys consumed:
    ``model``          — trained XGBoost model
    ``feature_df``     — flat feature matrix
    ``feature_columns``— ordered feature names
    ``ts_df``          — time-series DataFrame (for label re-derivation)
    ``train_metrics``  — metrics already computed by ModelingAgent (optional)

Output context keys added:
    ``validation_report`` — dict with all evaluation metrics
    ``quality_gate_pass`` — bool: True if model meets minimum thresholds
    ``quality_gate_notes``— list of human-readable gate messages

Contract:
    The agent ONLY evaluates.  It does NOT modify the model or write files.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.core.config import settings
from app.core.logger import get_logger
from tools.model_tools import evaluate_model

logger = get_logger(__name__)

# ── Quality gate thresholds (override via environment) ────────────────────────
_MIN_ROC_AUC   = float(getattr(settings, "MIN_ROC_AUC",   0.65))
_MIN_PRECISION = float(getattr(settings, "MIN_PRECISION",  0.40))
_MIN_RECALL    = float(getattr(settings, "MIN_RECALL",     0.40))


def _derive_labels_from_gap(feature_df: pd.DataFrame, window_days: int) -> pd.Series:
    """Re-derive binary churn labels from recency_days for validation."""
    return (feature_df["recency_days"] >= window_days).astype(int)


class ValidationAgent:
    """
    Agent 5/5: Validates model quality and generates an evaluation report.

    The agent re-derives labels from the feature matrix so it can run
    independently even if ``train_metrics`` are absent from context.

    Quality gates:
        * ROC-AUC  ≥ ``MIN_ROC_AUC``   (default 0.65)
        * Precision ≥ ``MIN_PRECISION`` (default 0.40)
        * Recall   ≥ ``MIN_RECALL``    (default 0.40)

    Usage::

        agent = ValidationAgent()
        ctx   = agent.run(ctx)
        # ctx["validation_report"]  →  full metrics dict
        # ctx["quality_gate_pass"]  →  True / False

    """

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute model validation.

        Args:
            context: Pipeline context with ``model``, ``feature_df``,
                     ``feature_columns``, and optionally ``train_metrics``.

        Returns:
            Enriched context with ``validation_report``,
            ``quality_gate_pass``, and ``quality_gate_notes``.
        """
        logger.info("ValidationAgent started")

        model       = context.get("model")
        feature_df: pd.DataFrame = context.get("feature_df")
        feat_cols   = context.get("feature_columns", [])
        window      = context.get("churn_window_days", settings.CHURN_WINDOW_DAYS)

        if model is None or feature_df is None:
            raise KeyError(
                "ValidationAgent requires 'model' and 'feature_df' in context."
            )

        X = feature_df[feat_cols] if feat_cols else feature_df
        y = _derive_labels_from_gap(feature_df, window_days=window)

        # ── Full evaluation ───────────────────────────────────────────────────
        if hasattr(model, "predict_proba"):
            metrics = evaluate_model(model, X, y)
        else:
            # MLflow pyfunc — use predict and treat as probabilities
            import numpy as np
            preds = model.predict(X)
            proba = preds[:, 1] if hasattr(preds, "shape") and len(preds.shape) == 2 else preds
            y_pred = (np.array(proba) >= 0.5).astype(int)
            from sklearn.metrics import (accuracy_score, confusion_matrix,
                                         precision_score, recall_score, roc_auc_score)
            metrics = {
                "roc_auc":          round(float(roc_auc_score(y, proba)), 4),
                "accuracy":         round(float(accuracy_score(y, y_pred)), 4),
                "precision":        round(float(precision_score(y, y_pred, zero_division=0)), 4),
                "recall":           round(float(recall_score(y, y_pred, zero_division=0)), 4),
                "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            }

        # Merge ModelingAgent metrics if available
        train_m = context.get("train_metrics", {})
        validation_report: dict[str, Any] = {
            "evaluation_metrics":  metrics,
            "training_metrics":    train_m,
            "total_users_scored":  len(feature_df),
            "churn_window_days":   window,
            "run_id":              context.get("run_id"),
        }

        # ── Quality gate ──────────────────────────────────────────────────────
        notes: list[str] = []
        passed = True

        if metrics["roc_auc"] < _MIN_ROC_AUC:
            notes.append(f"ROC-AUC {metrics['roc_auc']} < threshold {_MIN_ROC_AUC}")
            passed = False
        if metrics["precision"] < _MIN_PRECISION:
            notes.append(f"Precision {metrics['precision']} < threshold {_MIN_PRECISION}")
            passed = False
        if metrics["recall"] < _MIN_RECALL:
            notes.append(f"Recall {metrics['recall']} < threshold {_MIN_RECALL}")
            passed = False

        if passed:
            notes.append("All quality gates passed ✅")

        context["validation_report"] = validation_report
        context["quality_gate_pass"] = passed
        context["quality_gate_notes"] = notes

        logger.info(
            "ValidationAgent complete",
            extra={
                "roc_auc":     metrics["roc_auc"],
                "precision":   metrics["precision"],
                "recall":      metrics["recall"],
                "gate_passed": passed,
            },
        )
        return context
