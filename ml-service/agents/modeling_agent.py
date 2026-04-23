"""
Modeling Agent — Phase 3 of the telecom churn multi-agent pipeline.

Responsibility:
    Train an XGBoost churn classifier on the engineered feature matrix
    using stratified train/test splitting, compute comprehensive metrics,
    derive SHAP-based feature importance, calculate business impact
    metrics, and log everything to MLflow.

Input context keys consumed:
    ``feature_df``      — feature matrix indexed by ``customer_id``
    ``churn_labels``    — binary target Series aligned to ``feature_df``
    ``feature_columns`` — ordered list of model input column names
    ``customers_df``    — optional: original customer DataFrame for
                          business metrics (monthly charges per customer)

Output context keys added:
    ``model``                — trained XGBoost classifier
    ``run_id``               — MLflow run ID
    ``train_metrics``        — dict of evaluation metrics on the test set
    ``feature_importance``   — dict of feature → SHAP mean absolute value
    ``business_metrics``     — estimated revenue-at-risk and precision/cost
    ``optimal_threshold``    — threshold that maximises F1 on test set
    ``X_test``               — held-out feature matrix (for ValidationAgent)
    ``y_test``               — held-out labels (for ValidationAgent)

Contract:
    The agent ONLY trains and logs. It does NOT produce per-customer
    predictions (that is PredictionAgent's job).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedShuffleSplit

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

_TEST_SIZE         = 0.20   # 80/20 stratified split
_EARLY_STOP_ROUNDS = 30
_MIN_CHURN_SAMPLES = 5      # guard for degenerate datasets


class ModelingAgent:
    """
    Agent 3/5: Trains the XGBoost telecom churn model.

    Key capabilities:
        * **Stratified split** — preserves churn rate in both train/test sets.
        * **Class-imbalance handling** — ``scale_pos_weight`` auto-computed.
        * **Early stopping** — prevents over-fitting on small datasets.
        * **SHAP importance** — uses TreeExplainer for feature attribution.
        * **Threshold optimisation** — finds F1-optimal decision threshold.
        * **Business metrics** — estimated monthly revenue at risk.
        * **MLflow logging** — params, metrics, SHAP values, model artifact.
        * **Local fallback** — saves ``data/model.json`` if MLflow is down.

    Usage::

        agent = ModelingAgent()
        ctx   = agent.run(ctx)
        # ctx["model"]              → fitted XGBClassifier
        # ctx["run_id"]             → MLflow run ID
        # ctx["train_metrics"]      → {"roc_auc": ..., "f1": ..., ...}
        # ctx["feature_importance"] → {"risk_composite": 0.14, ...}
        # ctx["business_metrics"]   → {"revenue_at_risk_monthly": 42_000, ...}

    """

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute model training.

        Args:
            context: Pipeline context containing ``feature_df``,
                     ``churn_labels``, and ``feature_columns``.

        Returns:
            Enriched context with ``model``, ``run_id``, ``train_metrics``,
            ``feature_importance``, ``business_metrics``, and
            ``optimal_threshold``.

        Raises:
            KeyError: If required context keys are absent.
            ValueError: If the dataset is too small to train reliably.
        """
        logger.info("ModelingAgent started")

        feature_df: pd.DataFrame = context.get("feature_df")
        churn_labels: pd.Series  = context.get("churn_labels")
        feat_cols: list[str]     = context.get("feature_columns", [])

        if feature_df is None or churn_labels is None:
            raise KeyError(
                "ModelingAgent requires 'feature_df' and 'churn_labels' in "
                "context.  Run IngestionAgent and FeatureAgent first."
            )

        X = feature_df[feat_cols] if feat_cols else feature_df
        y = churn_labels.reindex(X.index).fillna(0).astype(int)

        if len(y.unique()) < 2:
            raise ValueError(
                "Cannot train: churn labels are single-class. "
                f"Unique values: {y.unique().tolist()}"
            )
        if y.sum() < _MIN_CHURN_SAMPLES:
            raise ValueError(
                f"Too few churned samples ({int(y.sum())}) for reliable training. "
                f"Need at least {_MIN_CHURN_SAMPLES}."
            )

        # ── 1. Stratified train/test split ────────────────────────────────────
        sss = StratifiedShuffleSplit(n_splits=1, test_size=_TEST_SIZE, random_state=42)
        train_idx, test_idx = next(sss.split(X, y))

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_test  = y.iloc[test_idx]

        logger.info(
            "Stratified split",
            extra={
                "train_size":        len(X_train),
                "test_size":         len(X_test),
                "train_churn_rate":  round(float(y_train.mean()), 4),
                "test_churn_rate":   round(float(y_test.mean()),  4),
            },
        )

        # ── 2. Hyperparameters ────────────────────────────────────────────────
        spw    = compute_scale_pos_weight(y_train)
        params = default_xgb_params(scale_pos_weight=spw)
        # Allow caller to override any parameter
        if context.get("xgb_params"):
            params.update(context["xgb_params"])

        logger.info("XGBoost hyperparameters", extra={"params": params})

        # ── 3. Train with early stopping ──────────────────────────────────────
        model = xgb.XGBClassifier(
            **params,
            early_stopping_rounds=_EARLY_STOP_ROUNDS,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        logger.info(
            "Model trained",
            extra={"best_iteration": getattr(model, "best_iteration", "N/A")},
        )

        # ── 4. Evaluate (full metric suite) ───────────────────────────────────
        metrics = _evaluate_full(model, X_test, y_test)
        logger.info("Evaluation complete", extra={k: v for k, v in metrics.items()
                                                   if isinstance(v, float)})

        # ── 5. Threshold optimisation ─────────────────────────────────────────
        optimal_threshold = _find_optimal_threshold(model, X_test, y_test)
        metrics["optimal_threshold"] = round(optimal_threshold, 4)

        # ── 6. SHAP feature importance ────────────────────────────────────────
        feature_importance = _compute_shap_importance(model, X_train, feat_cols)

        # ── 7. Business metrics ───────────────────────────────────────────────
        customers_df: pd.DataFrame = context.get("customers_df")
        business_metrics = _compute_business_metrics(
            model, X_test, y_test, customers_df
        )

        # ── 8. MLflow logging ─────────────────────────────────────────────────
        setup_mlflow()
        mlflow_metrics = {k: v for k, v in metrics.items() if isinstance(v, float)}
        mlflow_metrics.update({
            "scale_pos_weight": spw,
            "n_train":          len(X_train),
            "n_test":           len(X_test),
            "churn_rate_train": round(float(y_train.mean()), 4),
        })
        run_id = log_model_to_mlflow(
            model=model,
            params=params,
            metrics=mlflow_metrics,
            register=context.get("register_model", True),
        )

        # ── 9. Local model fallback ───────────────────────────────────────────
        import os
        os.makedirs("data", exist_ok=True)
        model.save_model("data/model.json")
        logger.info("Model saved to local fallback: data/model.json")

        # ── 10. Populate context ──────────────────────────────────────────────
        context["model"]              = model
        context["run_id"]             = run_id
        context["train_metrics"]      = metrics
        context["feature_importance"] = feature_importance
        context["business_metrics"]   = business_metrics
        context["optimal_threshold"]  = optimal_threshold
        context["X_test"]             = X_test
        context["y_test"]             = y_test

        logger.info(
            "ModelingAgent complete",
            extra={
                "run_id":             run_id,
                "roc_auc":            metrics.get("roc_auc"),
                "f1":                 metrics.get("f1"),
                "optimal_threshold":  optimal_threshold,
                "top_feature":        list(feature_importance.keys())[:3] if feature_importance else [],
                "revenue_at_risk":    business_metrics.get("revenue_at_risk_monthly", 0),
            },
        )
        return context


# ── Private helpers ───────────────────────────────────────────────────────────


def _evaluate_full(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Extended evaluation: ROC-AUC, PR-AUC, F1, Brier, KS statistic."""
    from sklearn.metrics import (
        accuracy_score, average_precision_score, brier_score_loss,
        confusion_matrix, f1_score, precision_score, recall_score,
        roc_auc_score,
    )
    from scipy.stats import ks_2samp

    base = evaluate_model(model, X_test, y_test)   # roc_auc, accuracy, precision, recall, cm

    y_proba  = model.predict_proba(X_test)[:, 1]
    y_pred   = (y_proba >= 0.5).astype(int)

    pos_scores = y_proba[y_test == 1]
    neg_scores = y_proba[y_test == 0]
    ks_stat, _ = ks_2samp(pos_scores, neg_scores)

    pr_auc  = float(average_precision_score(y_test, y_proba))
    brier   = float(brier_score_loss(y_test, y_proba))
    f1      = float(f1_score(y_test, y_pred, zero_division=0))

    base.update({
        "pr_auc":    round(pr_auc,  4),
        "brier_score": round(brier, 4),
        "f1":        round(f1,      4),
        "ks_stat":   round(float(ks_stat), 4),
    })
    return base


def _find_optimal_threshold(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_steps: int = 100,
) -> float:
    """Return the probability threshold that maximises F1 on X_test / y_test."""
    from sklearn.metrics import f1_score

    y_proba = model.predict_proba(X_test)[:, 1]
    best_t  = 0.5
    best_f1 = 0.0

    for t in np.linspace(0.01, 0.99, n_steps):
        y_pred = (y_proba >= t).astype(int)
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t

    logger.info(
        "Optimal threshold found",
        extra={"threshold": round(best_t, 4), "f1": round(best_f1, 4)},
    )
    return float(best_t)


def _compute_shap_importance(
    model: xgb.XGBClassifier,
    X_train: pd.DataFrame,
    feat_cols: list[str],
) -> dict[str, float]:
    """
    Compute mean-absolute SHAP values as feature importance.

    Falls back to XGBoost's built-in ``feature_importances_`` if SHAP is
    unavailable or the dataset is too large.
    """
    try:
        import shap
        # Use a sample of up to 2 000 rows for speed
        sample = X_train.sample(min(2_000, len(X_train)), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        mean_abs = np.abs(shap_values).mean(axis=0)
        importance = dict(zip(feat_cols, mean_abs.tolist()))
        importance = dict(
            sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
        )
        logger.info("SHAP importance computed", extra={"top_3": list(importance.keys())[:3]})
        return importance
    except Exception as exc:
        logger.warning("SHAP unavailable, using XGBoost gain importance",
                       extra={"error": str(exc)})
        scores = model.get_booster().get_score(importance_type="gain")
        total  = sum(scores.values()) or 1.0
        return {k: round(v / total, 6) for k, v in
                sorted(scores.items(), key=lambda kv: kv[1], reverse=True)}


def _compute_business_metrics(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    customers_df: pd.DataFrame | None,
) -> dict[str, Any]:
    """
    Estimate revenue-at-risk and cost metrics from test-set predictions.

    If ``customers_df`` is available, monthly charge data is used;
    otherwise a flat $65/month assumption is applied.
    """
    y_proba  = model.predict_proba(X_test)[:, 1]
    y_pred   = (y_proba >= 0.5).astype(int)

    # Average monthly charge per customer
    avg_monthly_charge = 65.0
    if customers_df is not None and "monthlycharges" in customers_df.columns:
        avg_monthly_charge = float(
            pd.to_numeric(customers_df["monthlycharges"], errors="coerce").median()
        )

    n_predicted_churners = int(y_pred.sum())
    revenue_at_risk      = round(n_predicted_churners * avg_monthly_charge, 2)

    # True positives (correctly identified churners) → can be retained
    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())

    retention_opportunity = round(tp * avg_monthly_charge, 2)
    false_alarm_cost      = fp   # wasted retention outreach contacts

    bm = {
        "avg_monthly_charge_usd":     round(avg_monthly_charge, 2),
        "n_test_customers":           len(y_test),
        "n_actual_churners_test":     int(y_test.sum()),
        "n_predicted_churners_test":  n_predicted_churners,
        "revenue_at_risk_monthly":    revenue_at_risk,
        "retention_opportunity_usd":  retention_opportunity,
        "false_alarms":               false_alarm_cost,
        "true_positives":             tp,
        "false_negatives":            fn,
    }
    logger.info("Business metrics computed", extra=bm)
    return bm

