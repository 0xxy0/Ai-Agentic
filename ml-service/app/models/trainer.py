"""
Model trainer module.

Encapsulates XGBoost training logic, MLflow logging, and model registration.
This module is called by the training service — it must NOT contain any
API or business logic.
"""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from app.core.config import settings
from app.core.logger import get_logger
from app.features.builder import FEATURE_COLUMNS

logger = get_logger(__name__)


def _compute_scale_pos_weight(y: pd.Series) -> float:
    """Calculate XGBoost scale_pos_weight to handle class imbalance."""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    if n_pos == 0:
        return 1.0
    weight = n_neg / n_pos
    logger.info(
        "Class imbalance weight",
        extra={"n_neg": int(n_neg), "n_pos": int(n_pos), "weight": weight},
    )
    return float(weight)


def train_churn_model(
    df: pd.DataFrame,
    target_col: str = "churn_label",
    xgb_params: dict[str, Any] | None = None,
    register_model: bool = True,
) -> str:
    """
    Train an XGBoost churn classifier and log everything to MLflow.

    Uses a **time-based split** (first 80% of rows = train, last 20% = test)
    rather than a random split, which prevents data leakage in temporal data.

    Args:
        df: DataFrame containing engineered features + target column.
        target_col: Name of the binary churn label column (0/1).
        xgb_params: Override default XGBoost hyperparameters.
        register_model: If True, register the trained model in the MLflow
                        Model Registry under ``settings.MLFLOW_MODEL_NAME``.

    Returns:
        The MLflow ``run_id`` of the training run.

    Raises:
        ValueError: If ``target_col`` is not in ``df``.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    # ── Time-based split (no shuffle) ─────────────────────────────────────
    split_idx = int(len(df) * 0.8)
    X = df[FEATURE_COLUMNS]
    y = df[target_col]

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # ── XGBoost hyperparameters ───────────────────────────────────────────
    default_params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": _compute_scale_pos_weight(y_train),
        "random_state": 42,
        "use_label_encoder": False,
    }
    if xgb_params:
        default_params.update(xgb_params)

    # ── MLflow experiment tracking ────────────────────────────────────────
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run started", extra={"run_id": run_id})

        # Log params
        mlflow.log_params({k: v for k, v in default_params.items()
                           if isinstance(v, (str, int, float, bool))})
        mlflow.log_param("churn_window_days", settings.CHURN_WINDOW_DAYS)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))

        # Train
        model = xgb.XGBClassifier(**default_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred_class = (y_pred_proba >= 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred_class)
        roc_auc  = roc_auc_score(y_test, y_pred_proba)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", roc_auc)

        logger.info(
            "Training metrics",
            extra={"accuracy": accuracy, "roc_auc": roc_auc, "run_id": run_id},
        )

        # Log model artifact
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=settings.MLFLOW_MODEL_NAME if register_model else None,
        )

        logger.info(
            "Model logged to MLflow",
            extra={"registered": register_model, "model_name": settings.MLFLOW_MODEL_NAME},
        )

    return run_id
