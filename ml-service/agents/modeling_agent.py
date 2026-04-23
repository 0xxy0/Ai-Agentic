"""
Modeling Agent — Phase 3 of the multi-agent pipeline.

Responsibility:
    Train an XGBoost churn classifier using a time-based train/test split,
    log everything to MLflow, and persist the run_id and model object.

Input context keys consumed:
    ``feature_df``      — flat feature matrix (index = user_id)
    ``ts_df``           — time-series DataFrame (for churn label derivation)
    ``feature_columns`` — ordered list of model input column names

Output context keys added:
    ``model``           — trained XGBoost classifier
    ``run_id``          — MLflow run ID
    ``train_metrics``   — dict of training evaluation metrics
    ``train_end_month`` — month used as the train/test cutoff
    ``train_end_year``  — year  used as the train/test cutoff

Contract:
    The agent ONLY trains and logs. It does NOT load data or produce predictions.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import xgboost as xgb

from app.core.config import settings
from app.core.logger import get_logger
from tools.model_tools import (
    compute_scale_pos_weight,
    default_xgb_params,
    evaluate_model,
    log_model_to_mlflow,
    setup_mlflow,
)

logger = get_logger(__name__)

# Default cutoff: train on first 80% of periods
_TRAIN_SPLIT_RATIO = 0.8


def _derive_churn_label(
    feature_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    window_days: int,
) -> pd.Series:
    """
    Label a user as churned (1) if ``activity_gap`` exceeds the churn window.

    ``activity_gap`` is expressed in months; we treat 30 days ≈ 1 month.

    Args:
        feature_df: Feature matrix indexed by ``user_id``.
        ts_df: Time-series DataFrame (unused directly but validated).
        window_days: Churn definition window (30 | 60 | 90).

    Returns:
        Binary :class:`pandas.Series` aligned to ``feature_df`` index.
    """
    gap_months = window_days / 30
    # In the new 12-column set, recency_days is already scaled to days (months * 30)
    label = (feature_df["recency_days"] >= window_days).astype(int)
    churn_rate = label.mean()
    logger.info(
        "Churn labels derived",
        extra={"window_days": window_days, "churn_rate": round(float(churn_rate), 4)},
    )
    return label


class ModelingAgent:
    """
    Agent 3/5: Trains the XGBoost churn model with a time-based split.

    The train/test cut is determined by selecting the earliest 80% of
    unique periods in the dataset, ensuring no future data leaks into
    training.

    Usage::

        agent = ModelingAgent()
        ctx   = agent.run(ctx)
        # ctx["model"]        → fitted XGBClassifier
        # ctx["run_id"]       → MLflow run ID
        # ctx["train_metrics"] → {"roc_auc": ..., "precision": ..., ...}
    """

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute model training.

        Args:
            context: Pipeline context containing ``feature_df``, ``ts_df``,
                     and ``feature_columns``.

        Returns:
            Enriched context with ``model``, ``run_id``, ``train_metrics``.

        Raises:
            KeyError: If required context keys are absent.
        """
        logger.info("ModelingAgent started")

        feature_df: pd.DataFrame = context.get("feature_df")
        ts_df: pd.DataFrame      = context.get("ts_df")
        feat_cols: list[str]     = context.get("feature_columns", [])

        if feature_df is None or ts_df is None:
            raise KeyError(
                "ModelingAgent requires 'feature_df' and 'ts_df' in context."
            )

        window = context.get("churn_window_days", settings.CHURN_WINDOW_DAYS)

        # ── Derive labels ────────────────────────────────────────────────────
        y = _derive_churn_label(feature_df, ts_df, window_days=window)

        # ── Time-based train/test split ──────────────────────────────────────
        all_periods = sorted(ts_df["period"].unique())
        cutoff_idx  = max(1, int(len(all_periods) * _TRAIN_SPLIT_RATIO) - 1)
        cutoff      = all_periods[cutoff_idx]

        # Map periods back to users
        last_period_per_user = ts_df.groupby("user_id")["period"].max()
        train_users = last_period_per_user[last_period_per_user <= cutoff].index
        test_users  = last_period_per_user[last_period_per_user >  cutoff].index

        logger.info(
            "Train/test split attempted",
            extra={
                "cutoff": str(cutoff),
                "train_users_potential": len(train_users),
                "test_users_potential": len(test_users),
            },
        )

        X = feature_df[feat_cols]
        X_train, y_train = X.loc[X.index.isin(train_users)], y.loc[y.index.isin(train_users)]
        X_test,  y_test  = X.loc[X.index.isin(test_users)],  y.loc[y.index.isin(test_users)]

        # ── Fallback for single-class or empty splits (common in small sample data) ────
        if len(y_train.unique()) < 2:
            logger.warning("Time-based split resulted in single class. Falling back to random split.")
            from sklearn.model_selection import train_test_split
            train_idx, test_idx = train_test_split(feature_df.index, test_size=0.2, random_state=42)
            X_train, y_train = X.loc[train_idx], y.loc[train_idx]
            X_test, y_test = X.loc[test_idx], y.loc[test_idx]

        logger.info(
            "Final split summary",
            extra={
                "train_size": len(X_train),
                "test_size": len(X_test),
                "churn_rate_train": round(float(y_train.mean()), 4)
            }
        )

        if len(X_test) == 0:
            logger.warning("Test set is empty — using full data for evaluation")
            X_test, y_test = X_train, y_train

        # ── Train ────────────────────────────────────────────────────────────
        spw    = compute_scale_pos_weight(y_train)
        params = default_xgb_params(scale_pos_weight=spw)
        if context.get("xgb_params"):
            params.update(context["xgb_params"])

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # ── Evaluate ─────────────────────────────────────────────────────────
        metrics = evaluate_model(model, X_test, y_test)

        # ── Log to MLflow ─────────────────────────────────────────────────────
        setup_mlflow()
        run_id = log_model_to_mlflow(
            model=model,
            params=params,
            metrics={k: v for k, v in metrics.items() if isinstance(v, float)},
            register=context.get("register_model", True),
        )

        context["model"]          = model
        context["run_id"]         = run_id
        context["train_metrics"]  = metrics
        context["train_end_month"] = cutoff.month
        context["train_end_year"]  = cutoff.year

        # ── Local Fallback ───────────────────────────────────────────────────
        import os
        os.makedirs("data", exist_ok=True)
        model.save_model("data/model.json")
        logger.info("Model saved to local fallback: data/model.json")

        logger.info(
            "ModelingAgent complete",
            extra={"run_id": run_id, "roc_auc": metrics["roc_auc"]},
        )
        return context
