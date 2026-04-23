"""
Model tools — MLflow helpers, XGBoost utilities, and metric computation.

All functions are stateless wrappers; no mutable state is stored here.
"""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


# ── MLflow helpers ────────────────────────────────────────────────────────────


def setup_mlflow(experiment_name: str | None = None) -> None:
    """Configure MLflow tracking URI and set the active experiment."""
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name or settings.MLFLOW_EXPERIMENT_NAME)
    logger.info(
        "MLflow configured",
        extra={
            "tracking_uri": settings.MLFLOW_TRACKING_URI,
            "experiment": experiment_name or settings.MLFLOW_EXPERIMENT_NAME,
        },
    )


def log_model_to_mlflow(
    model: xgb.XGBClassifier,
    params: dict[str, Any],
    metrics: dict[str, float],
    register: bool = True,
) -> str:
    """
    Log parameters, metrics, and the model artifact to MLflow.

    Args:
        model: Trained XGBoost classifier.
        params: Hyperparameters dict to log.
        metrics: Evaluation metrics dict to log.
        register: If True, register model in the MLflow Model Registry.

    Returns:
        MLflow ``run_id``.
    """
    with mlflow.start_run() as run:
        mlflow.log_params({k: v for k, v in params.items()
                           if isinstance(v, (str, int, float, bool))})
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=settings.MLFLOW_MODEL_NAME if register else None,
        )
        run_id = run.info.run_id
        logger.info("Logged to MLflow", extra={"run_id": run_id, "metrics": metrics})

        if register:
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                mv = client.create_model_version(
                    name=settings.MLFLOW_MODEL_NAME,
                    source=f"runs:/{run_id}/model",
                    run_id=run_id
                )
                if settings.MLFLOW_MODEL_STAGE:
                    client.transition_model_version_stage(
                        name=settings.MLFLOW_MODEL_NAME,
                        version=mv.version,
                        stage=settings.MLFLOW_MODEL_STAGE,
                        archive_existing_versions=True
                    )
                    logger.info("Model transitioned to stage", extra={"stage": settings.MLFLOW_MODEL_STAGE, "version": mv.version})
            except Exception as e:
                logger.warning("Failed to transition model stage", extra={"error": str(e)})

    return run_id


# ── Class-imbalance helper ────────────────────────────────────────────────────


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Return XGBoost ``scale_pos_weight`` from label series."""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    return float(n_neg / n_pos) if n_pos > 0 else 1.0


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Compute a full evaluation report for a binary classifier.

    Args:
        model: Trained XGBoost model.
        X_test: Feature matrix.
        y_test: Ground-truth labels.
        threshold: Classification threshold.

    Returns:
        Dict containing ``roc_auc``, ``accuracy``, ``precision``,
        ``recall``, and ``confusion_matrix`` (as a nested list).
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred).tolist()
    metrics: dict[str, Any] = {
        "roc_auc":   round(float(roc_auc_score(y_test, y_proba)), 4),
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": cm,
    }
    logger.info("Evaluation complete", extra={k: v for k, v in metrics.items() if k != "confusion_matrix"})
    return metrics


# ── Default XGBoost hyperparameters ──────────────────────────────────────────


def default_xgb_params(scale_pos_weight: float = 1.0) -> dict[str, Any]:
    """Return production-ready XGBoost defaults."""
    return {
        "objective":         "binary:logistic",
        "eval_metric":       "auc",
        "n_estimators":      400,
        "max_depth":         6,
        "learning_rate":     0.04,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "min_child_weight":  5,
        "gamma":             0.1,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "scale_pos_weight":  scale_pos_weight,
        "random_state":      42,
        "use_label_encoder": False,
    }
