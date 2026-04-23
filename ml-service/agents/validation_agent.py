"""
Validation Agent — Phase 5 of the telecom churn multi-agent pipeline.

Responsibility:
    Perform comprehensive model evaluation covering statistical metrics,
    calibration quality, decile-based lift/gain analysis, demographic
    fairness checks, business impact quantification, and quality gate
    enforcement.

Input context keys consumed:
    ``model``            — trained XGBoost model
    ``feature_df``       — feature matrix indexed by ``customer_id``
    ``feature_columns``  — ordered feature names
    ``churn_labels``     — binary target Series aligned to ``feature_df``
    ``train_metrics``    — metrics from ModelingAgent (optional)
    ``customers_df``     — 32-column customer DataFrame (optional, used
                           for fairness analysis by demographic segment)
    ``optimal_threshold``— F1-optimal threshold (optional, default 0.5)

Output context keys added:
    ``validation_report``   — comprehensive evaluation dict
    ``quality_gate_pass``   — bool: True if all gates pass
    ``quality_gate_notes``  — list of human-readable gate messages
    ``optimal_threshold``   — best F1 threshold (echoed for downstream)

Contract:
    The agent ONLY evaluates. It does NOT modify the model or write files.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.logger import get_logger
from tools.model_tools import evaluate_model

logger = get_logger(__name__)

# ── Quality gate thresholds (configurable via environment) ────────────────────
_MIN_ROC_AUC    = float(getattr(settings, "MIN_ROC_AUC",    0.70))
_MIN_PRECISION  = float(getattr(settings, "MIN_PRECISION",   0.45))
_MIN_RECALL     = float(getattr(settings, "MIN_RECALL",      0.45))
_MIN_PR_AUC     = 0.40
_MAX_BRIER      = 0.20
_MIN_KS_STAT    = 0.25
_MIN_F1         = 0.45

# Fairness: max allowed delta in average churn score across demographic groups
_MAX_FAIRNESS_DELTA = 0.15


class ValidationAgent:
    """
    Agent 5/5: Comprehensive model validation for the telecom churn model.

    Evaluation suite:
        **Statistical metrics**
            ROC-AUC, PR-AUC (Average Precision), F1, Brier score, KS statistic,
            accuracy, precision, recall, confusion matrix

        **Calibration analysis**
            Calibration curve (10 bins), Expected Calibration Error (ECE),
            over-/under-confidence detection

        **Decile lift & gain analysis**
            Lift table (10 deciles), cumulative gain curve, top-decile lift

        **Fairness checks**
            Demographic parity gap (by gender, senior_citizen, education)
            Equalized odds proxy (TPR difference between groups)

        **Business impact metrics**
            Revenue at risk, estimated retention savings, cost of misclassification

        **Threshold optimisation**
            Precision-recall tradeoff at 10 operating points

    Quality gates (all must pass):
        * ROC-AUC  ≥ {_MIN_ROC_AUC}
        * PR-AUC   ≥ {_MIN_PR_AUC}
        * F1       ≥ {_MIN_F1}
        * Precision ≥ {_MIN_PRECISION}
        * Recall   ≥ {_MIN_RECALL}
        * Brier    ≤ {_MAX_BRIER}
        * KS stat  ≥ {_MIN_KS_STAT}

    Usage::

        agent = ValidationAgent()
        ctx   = agent.run(ctx)
        # ctx["validation_report"] → full evaluation dict
        # ctx["quality_gate_pass"] → True / False

    """

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute comprehensive model validation.

        Args:
            context: Pipeline context with ``model``, ``feature_df``,
                     ``churn_labels``, ``feature_columns``, and optionally
                     ``train_metrics``, ``customers_df``.

        Returns:
            Enriched context with ``validation_report``,
            ``quality_gate_pass``, and ``quality_gate_notes``.

        Raises:
            KeyError: If ``model``, ``feature_df``, or ``churn_labels``
                      are absent.
        """
        logger.info("ValidationAgent started")

        model                    = context.get("model")
        feature_df: pd.DataFrame = context.get("feature_df")
        feat_cols: list[str]     = context.get("feature_columns", [])
        churn_labels             = context.get("churn_labels")
        customers_df             = context.get("customers_df")
        threshold: float         = context.get("optimal_threshold", 0.5)

        if model is None:
            raise KeyError("ValidationAgent requires 'model' in context.")
        if feature_df is None:
            raise KeyError("ValidationAgent requires 'feature_df' in context.")
        if churn_labels is None:
            raise KeyError("ValidationAgent requires 'churn_labels' in context.")

        X = feature_df[feat_cols] if feat_cols else feature_df
        y = churn_labels.reindex(X.index).fillna(0).astype(int)

        # ── 1. Core metrics ───────────────────────────────────────────────────
        core_metrics = _compute_core_metrics(model, X, y, threshold)
        logger.info("Core metrics", extra={k: v for k, v in core_metrics.items()
                                            if isinstance(v, float)})

        # ── 2. Calibration analysis ───────────────────────────────────────────
        calibration = _compute_calibration(model, X, y)

        # ── 3. Decile lift analysis ───────────────────────────────────────────
        decile_table = _compute_decile_lift(model, X, y)

        # ── 4. Threshold analysis ─────────────────────────────────────────────
        threshold_analysis = _compute_threshold_analysis(model, X, y)

        # ── 5. Fairness analysis ──────────────────────────────────────────────
        fairness_report = _compute_fairness(model, X, y, feature_df, customers_df, feat_cols)

        # ── 6. Business impact ────────────────────────────────────────────────
        business_impact = _compute_business_impact(model, X, y, customers_df, threshold)

        # ── 7. Assemble full report ───────────────────────────────────────────
        train_m = context.get("train_metrics", {})
        validation_report: dict[str, Any] = {
            "evaluation_metrics":    core_metrics,
            "training_metrics":      train_m,
            "calibration_analysis":  calibration,
            "decile_lift_table":     decile_table,
            "threshold_analysis":    threshold_analysis,
            "fairness_report":       fairness_report,
            "business_impact":       business_impact,
            "total_customers_scored": len(y),
            "churn_prevalence":      round(float(y.mean()), 4),
            "decision_threshold":    round(threshold, 4),
            "run_id":                context.get("run_id"),
        }

        # ── 8. Quality gates ──────────────────────────────────────────────────
        passed, notes = _evaluate_quality_gates(core_metrics, fairness_report)

        context["validation_report"] = validation_report
        context["quality_gate_pass"] = passed
        context["quality_gate_notes"] = notes

        logger.info(
            "ValidationAgent complete",
            extra={
                "roc_auc":    core_metrics.get("roc_auc"),
                "pr_auc":     core_metrics.get("pr_auc"),
                "f1":         core_metrics.get("f1"),
                "ks_stat":    core_metrics.get("ks_stat"),
                "brier":      core_metrics.get("brier_score"),
                "gate_passed": passed,
            },
        )
        return context


# ── Private helpers ───────────────────────────────────────────────────────────


def _compute_core_metrics(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
) -> dict[str, Any]:
    """Compute full statistical metric suite."""
    from sklearn.metrics import (
        accuracy_score, average_precision_score, brier_score_loss,
        confusion_matrix, f1_score, precision_score, recall_score,
        roc_auc_score,
    )
    from scipy.stats import ks_2samp

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        raw = model.predict(X)
        y_proba = raw[:, 1] if hasattr(raw, "shape") and raw.ndim == 2 else raw

    y_proba = np.clip(y_proba, 0.0, 1.0)
    y_pred  = (y_proba >= threshold).astype(int)

    pos_scores = y_proba[y == 1]
    neg_scores = y_proba[y == 0]
    ks_stat = 0.0
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        ks_stat, _ = ks_2samp(pos_scores, neg_scores)

    metrics: dict[str, Any] = {
        "roc_auc":          round(float(roc_auc_score(y, y_proba)), 4),
        "pr_auc":           round(float(average_precision_score(y, y_proba)), 4),
        "accuracy":         round(float(accuracy_score(y, y_pred)), 4),
        "precision":        round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "recall":           round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "f1":               round(float(f1_score(y, y_pred, zero_division=0)), 4),
        "brier_score":      round(float(brier_score_loss(y, y_proba)), 4),
        "ks_stat":          round(float(ks_stat), 4),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "n_positive":       int(y.sum()),
        "n_negative":       int((y == 0).sum()),
    }
    return metrics


def _compute_calibration(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_bins: int = 10,
) -> dict[str, Any]:
    """
    Compute calibration curve and Expected Calibration Error (ECE).

    Returns a dict with ``bins``, ``mean_predicted_prob``,
    ``fraction_of_positives``, and ``ece``.
    """
    try:
        from sklearn.calibration import calibration_curve
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
        else:
            raw = model.predict(X)
            y_proba = raw[:, 1] if hasattr(raw, "shape") and raw.ndim == 2 else raw

        frac_pos, mean_pred = calibration_curve(y, y_proba, n_bins=n_bins, strategy="quantile")

        # Expected Calibration Error
        bin_counts = np.histogram(y_proba, bins=n_bins)[0]
        bin_totals = bin_counts.sum() or 1
        ece = float(np.sum(np.abs(frac_pos - mean_pred) * (bin_counts[:len(frac_pos)] / bin_totals)))

        over_confidence  = bool(mean_pred.mean() > frac_pos.mean() + 0.05)
        under_confidence = bool(mean_pred.mean() < frac_pos.mean() - 0.05)

        return {
            "mean_predicted_prob":   [round(float(x), 4) for x in mean_pred],
            "fraction_of_positives": [round(float(x), 4) for x in frac_pos],
            "ece":                   round(ece, 4),
            "over_confidence":       over_confidence,
            "under_confidence":      under_confidence,
        }
    except Exception as exc:
        logger.warning("Calibration analysis failed", extra={"error": str(exc)})
        return {"error": str(exc)}


def _compute_decile_lift(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
) -> list[dict[str, Any]]:
    """
    Build a 10-decile lift table.

    Each row contains: decile number, score range, n_customers, n_actual_churners,
    churn_rate_in_decile, lift (relative to overall churn rate), cumulative_gain.
    """
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
        else:
            raw = model.predict(X)
            y_proba = raw[:, 1] if hasattr(raw, "shape") and raw.ndim == 2 else raw

        df = pd.DataFrame({"score": y_proba, "label": y.values})
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df["decile"] = pd.qcut(df.index, q=10, labels=False) + 1

        overall_churn_rate = float(y.mean()) or 1e-9
        table = []
        cumulative_gain = 0.0
        total_pos = int(y.sum()) or 1

        for dec in range(1, 11):
            grp = df[df["decile"] == dec]
            n   = len(grp)
            pos = int(grp["label"].sum())
            rate = pos / n if n > 0 else 0.0
            lift = rate / overall_churn_rate
            cumulative_gain += pos / total_pos

            table.append({
                "decile":             dec,
                "min_score":          round(float(grp["score"].min()), 4),
                "max_score":          round(float(grp["score"].max()), 4),
                "n_customers":        n,
                "n_churned":          pos,
                "churn_rate":         round(rate,              4),
                "lift":               round(float(lift),       4),
                "cumulative_gain":    round(cumulative_gain,   4),
            })

        return table
    except Exception as exc:
        logger.warning("Decile lift computation failed", extra={"error": str(exc)})
        return []


def _compute_threshold_analysis(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_points: int = 10,
) -> list[dict[str, Any]]:
    """
    Evaluate precision, recall, and F1 at ``n_points`` evenly spaced
    thresholds between 0.1 and 0.9.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        raw = model.predict(X)
        y_proba = raw[:, 1] if hasattr(raw, "shape") and raw.ndim == 2 else raw

    results = []
    for t in np.linspace(0.1, 0.9, n_points):
        y_pred = (y_proba >= t).astype(int)
        results.append({
            "threshold": round(float(t), 2),
            "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
            "recall":    round(float(recall_score(y, y_pred, zero_division=0)),    4),
            "f1":        round(float(f1_score(y, y_pred, zero_division=0)),        4),
            "n_predicted_positive": int(y_pred.sum()),
        })
    return results


def _compute_fairness(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    feature_df: pd.DataFrame,
    customers_df: pd.DataFrame | None,
    feat_cols: list[str],
) -> dict[str, Any]:
    """
    Compute demographic parity and equalised-odds proxy metrics for
    gender, senior_citizen, and (if available) education.

    Demographic parity:  difference in average churn score between groups.
    Equalised odds proxy: difference in true-positive rate (recall) between groups.
    """
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        raw = model.predict(X)
        y_proba = raw[:, 1] if hasattr(raw, "shape") and raw.ndim == 2 else raw

    result: dict[str, Any] = {}

    def _group_stats(mask_a: np.ndarray, mask_b: np.ndarray,
                     group_name: str, label_a: str, label_b: str) -> dict:
        if mask_a.sum() == 0 or mask_b.sum() == 0:
            return {}
        score_a = y_proba[mask_a].mean()
        score_b = y_proba[mask_b].mean()
        # Equalised odds (TPR per group, avoid divide-by-zero)
        y_arr   = y.values
        pred_arr = (y_proba >= 0.5).astype(int)
        pos_a = y_arr[mask_a].sum()
        pos_b = y_arr[mask_b].sum()
        tpr_a = float(pred_arr[mask_a & (y_arr == 1)].mean()) if pos_a > 0 else 0.0
        tpr_b = float(pred_arr[mask_b & (y_arr == 1)].mean()) if pos_b > 0 else 0.0

        return {
            f"avg_score_{label_a}":       round(float(score_a), 4),
            f"avg_score_{label_b}":       round(float(score_b), 4),
            "demographic_parity_delta":   round(float(abs(score_a - score_b)), 4),
            f"tpr_{label_a}":             round(tpr_a, 4),
            f"tpr_{label_b}":             round(tpr_b, 4),
            "equalised_odds_delta":       round(abs(tpr_a - tpr_b), 4),
            "fairness_concern":           bool(abs(score_a - score_b) > _MAX_FAIRNESS_DELTA),
        }

    # Use feature_df for engineered binary attributes
    if "gender_male" in feature_df.columns:
        male_mask   = feature_df["gender_male"].values.astype(bool)
        female_mask = ~male_mask
        result["gender"] = _group_stats(male_mask, female_mask, "gender", "male", "female")

    if "senior_citizen" in feature_df.columns:
        senior_mask     = feature_df["senior_citizen"].values.astype(bool)
        nonseniormask   = ~senior_mask
        result["senior_citizen"] = _group_stats(
            senior_mask, nonseniormask, "senior_citizen", "senior", "non_senior"
        )

    # Education: graduate vs non-graduate
    if "edu_graduate" in feature_df.columns:
        grad_mask    = feature_df["edu_graduate"].values.astype(bool)
        nongrad_mask = ~grad_mask
        result["education_graduate"] = _group_stats(
            grad_mask, nongrad_mask, "education", "graduate", "non_graduate"
        )

    return result


def _compute_business_impact(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    customers_df: pd.DataFrame | None,
    threshold: float,
) -> dict[str, Any]:
    """
    Translate model quality metrics into business terms.

    Metrics:
        * ``revenue_protected_usd``: monthly charges of correctly identified churners
        * ``lost_revenue_usd``:      monthly charges of missed churners (FN)
        * ``false_alarm_cost``:      number of wasted retention contacts (FP)
        * ``net_benefit_usd``:       retained revenue minus outreach cost
          (assumes $10 per outreach contact)
    """
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        raw = model.predict(X)
        y_proba = raw[:, 1] if hasattr(raw, "shape") and raw.ndim == 2 else raw

    y_pred = (y_proba >= threshold).astype(int)
    y_arr  = y.values

    tp = int(((y_pred == 1) & (y_arr == 1)).sum())
    fp = int(((y_pred == 1) & (y_arr == 0)).sum())
    fn = int(((y_pred == 0) & (y_arr == 1)).sum())
    tn = int(((y_pred == 0) & (y_arr == 0)).sum())

    avg_monthly = 65.0
    if customers_df is not None and "monthlycharges" in customers_df.columns:
        avg_monthly = float(
            pd.to_numeric(customers_df["monthlycharges"], errors="coerce").median()
        )

    outreach_cost_per_contact = 10.0   # $10 per retention outreach

    revenue_protected = round(tp * avg_monthly, 2)
    lost_revenue      = round(fn * avg_monthly, 2)
    false_alarm_cost  = round(fp * outreach_cost_per_contact, 2)
    net_benefit       = round(revenue_protected - false_alarm_cost, 2)

    return {
        "avg_monthly_charge_usd":    round(avg_monthly, 2),
        "true_positives":            tp,
        "false_positives":           fp,
        "false_negatives":           fn,
        "true_negatives":            tn,
        "revenue_protected_usd":     revenue_protected,
        "lost_revenue_usd":          lost_revenue,
        "false_alarm_cost_usd":      false_alarm_cost,
        "net_benefit_usd":           net_benefit,
        "outreach_cost_per_contact": outreach_cost_per_contact,
    }


def _evaluate_quality_gates(
    metrics: dict[str, Any],
    fairness_report: dict[str, Any],
) -> tuple[bool, list[str]]:
    """
    Evaluate all quality gates. Return (passed, notes).

    All gates must pass for ``passed`` to be True.
    """
    notes: list[str] = []
    passed = True

    def _check(value: float, threshold: float, name: str, above: bool = True) -> None:
        nonlocal passed
        ok = value >= threshold if above else value <= threshold
        op = "≥" if above else "≤"
        if not ok:
            notes.append(f"❌ {name} {value:.4f} {op} {threshold} FAILED")
            passed = False
        else:
            notes.append(f"✅ {name} {value:.4f} {op} {threshold} passed")

    _check(metrics.get("roc_auc",    0.0), _MIN_ROC_AUC,   "ROC-AUC")
    _check(metrics.get("pr_auc",     0.0), _MIN_PR_AUC,    "PR-AUC")
    _check(metrics.get("f1",         0.0), _MIN_F1,        "F1")
    _check(metrics.get("precision",  0.0), _MIN_PRECISION, "Precision")
    _check(metrics.get("recall",     0.0), _MIN_RECALL,    "Recall")
    _check(metrics.get("brier_score",1.0), _MAX_BRIER,     "Brier score", above=False)
    _check(metrics.get("ks_stat",    0.0), _MIN_KS_STAT,   "KS statistic")

    # Fairness gate: no demographic group should have delta > threshold
    for group, stats in fairness_report.items():
        if isinstance(stats, dict) and stats.get("fairness_concern", False):
            delta = stats.get("demographic_parity_delta", 0.0)
            notes.append(
                f"⚠️  Fairness concern in '{group}': "
                f"demographic parity delta = {delta:.4f} > {_MAX_FAIRNESS_DELTA}"
            )
            # Fairness is a warning, not a hard gate failure

    if passed:
        notes.append("🎉 All quality gates passed — model is production-ready")

    return passed, notes

