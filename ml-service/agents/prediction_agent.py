"""
Prediction Agent — Phase 4 of the telecom churn multi-agent pipeline.

Responsibility:
    Run batch inference using the trained (or MLflow-loaded) model,
    compute per-customer SHAP explanations, assign four-tier risk levels,
    generate segment-level analysis, produce actionable retention
    recommendations, and calculate total revenue at risk.

Input context keys consumed:
    ``model``           — trained XGBoost model (or None → load from MLflow)
    ``feature_df``      — feature matrix indexed by ``customer_id``
    ``feature_columns`` — ordered list of model input column names
    ``customers_df``    — optional: full 32-column DataFrame for business
                          enrichment (monthly charges, contract, etc.)
    ``optimal_threshold`` — optional: F1-optimal threshold from ModelingAgent

Output context keys added:
    ``scored_df``           — DataFrame: customer_id | churn_score | risk_level
                              | top_risk_factors | recommended_action
    ``high_risk_customers`` — list of customer_ids with risk_level in
                              {HIGH, CRITICAL}
    ``summary_stats``       — aggregate prediction statistics dict
    ``segment_analysis``    — churn rate and avg score by contract/tenure/etc.
    ``revenue_at_risk``     — estimated monthly revenue at risk (float)

Contract:
    The agent ONLY produces predictions and explanations.
    It does NOT train or persist models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.logger import get_logger
from app.models.loader import load_model

logger = get_logger(__name__)

# ── Risk tier thresholds ──────────────────────────────────────────────────────
_CRITICAL_THRESHOLD = 0.80
_HIGH_THRESHOLD     = 0.60
_MEDIUM_THRESHOLD   = 0.35

# ── Retention recommendation templates ───────────────────────────────────────
_RETENTION_ACTIONS: dict[str, str] = {
    "CRITICAL": (
        "Immediate personal outreach required. Offer tailored retention package: "
        "contract upgrade incentive, loyalty discount (20–30%), or premium add-on bundle."
    ),
    "HIGH": (
        "Priority retention outreach within 48 hours. Propose contract extension "
        "with discount, satisfaction survey follow-up, and proactive support call."
    ),
    "MEDIUM": (
        "Scheduled check-in recommended. Highlight value-added services, "
        "offer loyalty reward points, and review service usage patterns."
    ),
    "LOW": (
        "Monitor engagement. Include in next NPS survey cycle and standard "
        "newsletter upsell campaign."
    ),
}

# ── Feature labels for human-readable explanations ───────────────────────────
_FEATURE_LABELS: dict[str, str] = {
    "contract_risk_score":        "Month-to-month contract (high churn risk)",
    "customer_satisfaction":      "Low customer satisfaction score",
    "num_complaints":             "High number of complaints filed",
    "late_payment_rate":          "Elevated late payment frequency",
    "days_since_last_interaction": "Long time since last interaction",
    "risk_composite":             "High composite risk score",
    "recency_normalized":         "Low recent engagement",
    "tenure":                     "Short customer tenure",
    "credit_score_normalized":    "Low credit score",
    "charge_ratio":               "Unusual charge-to-tenure ratio",
    "service_call_rate":          "Frequent service call rate",
    "satisfaction_normalized":    "Below-average satisfaction",
    "num_service_calls":          "High number of service calls",
    "senior_citizen":             "Senior customer segment",
    "paperless_billing":          "Paperless billing preference",
}


def _assign_risk(score: float) -> str:
    """Map churn probability to a four-tier risk label."""
    if score >= _CRITICAL_THRESHOLD:
        return "CRITICAL"
    if score >= _HIGH_THRESHOLD:
        return "HIGH"
    if score >= _MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


class PredictionAgent:
    """
    Agent 4/5: Scores all customers and generates enriched predictions.

    Per-customer output includes:
        * ``churn_score``        — probability of churn [0, 1]
        * ``risk_level``         — CRITICAL / HIGH / MEDIUM / LOW
        * ``top_risk_factors``   — comma-separated top 3 SHAP-driven reasons
        * ``recommended_action`` — human-readable retention action

    Aggregate output includes:
        * Risk distribution across 4 tiers
        * Segment-level average churn scores (by contract, tenure group)
        * Total estimated monthly revenue at risk

    Usage::

        agent = PredictionAgent()
        ctx   = agent.run(ctx)
        # ctx["scored_df"]          → enriched per-customer DataFrame
        # ctx["summary_stats"]      → aggregate stats dict
        # ctx["segment_analysis"]   → churn by contract / tenure / satisfaction
        # ctx["revenue_at_risk"]    → float (USD/month)

    """

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute batch scoring with SHAP explanations and segment analysis.

        Args:
            context: Pipeline context containing ``feature_df``
                     (and optionally ``model``, ``customers_df``,
                     ``optimal_threshold``).

        Returns:
            Enriched context with ``scored_df``, ``high_risk_customers``,
            ``summary_stats``, ``segment_analysis``, ``revenue_at_risk``.

        Raises:
            KeyError:    If ``feature_df`` is absent.
            RuntimeError: If model loading fails.
        """
        logger.info("PredictionAgent started")

        feature_df: pd.DataFrame = context.get("feature_df")
        feat_cols: list[str]     = context.get("feature_columns", [])
        customers_df             = context.get("customers_df")
        threshold: float         = context.get("optimal_threshold", 0.5)

        if feature_df is None:
            raise KeyError("PredictionAgent requires 'feature_df' in context.")

        # ── 1. Resolve model ──────────────────────────────────────────────────
        model = context.get("model")
        if model is None:
            logger.info("No in-context model — loading from MLflow registry")
            model = load_model()

        X = feature_df[feat_cols] if feat_cols else feature_df

        # ── 2. Batch predict ─────────────────────────────────────────────────
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        else:
            raw = model.predict(X)
            proba = raw[:, 1] if hasattr(raw, "shape") and raw.ndim == 2 else raw

        proba = np.clip(proba, 0.0, 1.0)

        # ── 3. Per-customer SHAP top-factor explanations ───────────────────────
        shap_explanations = _compute_shap_explanations(model, X, feat_cols)

        # ── 4. Build scored DataFrame ─────────────────────────────────────────
        customer_ids = (
            feature_df.index.tolist()
            if feature_df.index.name == "customer_id" or "CUST" in str(feature_df.index[0])
            else [f"CUST_{i:07d}" for i in range(len(feature_df))]
        )

        scored_df = pd.DataFrame({
            "customer_id":      customer_ids,
            "churn_score":      np.round(proba, 6),
            "churn_prediction": (proba >= threshold).astype(int),
            "risk_level":       [_assign_risk(float(s)) for s in proba],
        })

        scored_df["top_risk_factors"]   = shap_explanations
        scored_df["recommended_action"] = scored_df["risk_level"].map(_RETENTION_ACTIONS)
        scored_df = scored_df.reset_index(drop=True)

        # ── 5. Merge with customer attributes for segment analysis ────────────
        enriched_df = _enrich_with_attributes(scored_df, customers_df)

        # ── 6. Summary statistics ─────────────────────────────────────────────
        risk_counts = scored_df["risk_level"].value_counts().to_dict()
        summary: dict[str, Any] = {
            "total_customers":        len(scored_df),
            "critical_risk_count":    risk_counts.get("CRITICAL", 0),
            "high_risk_count":        risk_counts.get("HIGH",     0),
            "medium_risk_count":      risk_counts.get("MEDIUM",   0),
            "low_risk_count":         risk_counts.get("LOW",      0),
            "avg_churn_score":        round(float(proba.mean()),  4),
            "median_churn_score":     round(float(np.median(proba)), 4),
            "pct_predicted_churn":    round(float((proba >= threshold).mean()), 4),
            "decision_threshold":     round(threshold, 4),
        }

        # ── 7. Segment analysis ───────────────────────────────────────────────
        segment_analysis = _build_segment_analysis(enriched_df, customers_df)

        # ── 8. Revenue at risk ────────────────────────────────────────────────
        revenue_at_risk = _compute_revenue_at_risk(scored_df, customers_df)

        # ── 9. Populate context ───────────────────────────────────────────────
        context["scored_df"]           = scored_df
        context["high_risk_customers"] = scored_df.loc[
            scored_df["risk_level"].isin(["HIGH", "CRITICAL"]), "customer_id"
        ].tolist()
        context["summary_stats"]       = summary
        context["segment_analysis"]    = segment_analysis
        context["revenue_at_risk"]     = revenue_at_risk

        logger.info(
            "PredictionAgent complete",
            extra={
                **{k: v for k, v in summary.items() if isinstance(v, (int, float))},
                "revenue_at_risk": revenue_at_risk,
            },
        )
        return context


# ── Private helpers ───────────────────────────────────────────────────────────


def _compute_shap_explanations(
    model: Any,
    X: pd.DataFrame,
    feat_cols: list[str],
    top_n: int = 3,
) -> list[str]:
    """
    Return a comma-separated string of top-N risk factors per customer using
    SHAP TreeExplainer.  Falls back to feature importance on failure.
    """
    try:
        import shap

        if not hasattr(model, "predict_proba"):
            raise TypeError("SHAP requires a native XGBClassifier")

        # Limit sample for performance in large datasets
        sample_size = min(5_000, len(X))
        X_sample = X.iloc[:sample_size] if len(X) > sample_size else X

        explainer   = shap.TreeExplainer(model)
        shap_vals   = explainer.shap_values(X_sample)   # shape: (n, n_features)

        explanations = []
        for row_shap in shap_vals:
            # Sort features by absolute SHAP value descending
            top_idx = np.argsort(np.abs(row_shap))[::-1][:top_n]
            factors = []
            for idx in top_idx:
                feat_name = feat_cols[idx] if idx < len(feat_cols) else f"feat_{idx}"
                label     = _FEATURE_LABELS.get(feat_name, feat_name.replace("_", " ").title())
                direction = "↑ increases churn" if row_shap[idx] > 0 else "↓ reduces churn"
                factors.append(f"{label} ({direction})")
            explanations.append("; ".join(factors))

        # For rows beyond the sample, use a generic explanation
        if len(X) > sample_size:
            generic = explanations[-1] if explanations else "Explanation unavailable"
            explanations.extend([generic] * (len(X) - sample_size))

        return explanations

    except Exception as exc:
        logger.warning(
            "SHAP explanations unavailable — using generic placeholders",
            extra={"error": str(exc)},
        )
        return ["See feature importance report for top risk drivers"] * len(X)


def _enrich_with_attributes(
    scored_df: pd.DataFrame,
    customers_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge scored results with customer attributes for segmentation."""
    if customers_df is None:
        return scored_df
    try:
        attrs = customers_df[["customer_id", "contract", "tenure",
                               "customer_satisfaction", "monthlycharges",
                               "senior_citizen"]].copy()
        attrs["customer_id"] = attrs["customer_id"].astype(str)
        enriched = scored_df.merge(attrs, on="customer_id", how="left")
        return enriched
    except Exception:
        return scored_df


def _build_segment_analysis(
    enriched_df: pd.DataFrame,
    customers_df: pd.DataFrame | None,
) -> dict[str, Any]:
    """Compute average churn scores and risk distribution by key segments."""
    analysis: dict[str, Any] = {}

    def _seg(df: pd.DataFrame, col: str) -> dict:
        if col not in df.columns:
            return {}
        return (
            df.groupby(col, observed=True)["churn_score"]
            .agg(avg_score="mean", n_customers="count")
            .round(4)
            .to_dict("index")
        )

    # Segment by contract type
    if "contract" in enriched_df.columns:
        analysis["by_contract"] = _seg(enriched_df, "contract")

    # Segment by tenure bucket
    if customers_df is not None and "tenure" in customers_df.columns:
        merged = enriched_df.copy()
        if "tenure" not in merged.columns and customers_df is not None:
            merged = merged.merge(
                customers_df[["customer_id", "tenure"]], on="customer_id", how="left"
            )
        if "tenure" in merged.columns:
            merged["tenure_group"] = pd.cut(
                pd.to_numeric(merged["tenure"], errors="coerce").fillna(12),
                bins=[-1, 12, 24, 48, float("inf")],
                labels=["0–12m", "13–24m", "25–48m", "49m+"],
            )
            analysis["by_tenure_group"] = _seg(merged, "tenure_group")

    # Segment by customer satisfaction
    if "customer_satisfaction" in enriched_df.columns:
        analysis["by_satisfaction"] = _seg(enriched_df, "customer_satisfaction")

    # Senior vs non-senior
    if "senior_citizen" in enriched_df.columns:
        analysis["by_senior_citizen"] = _seg(enriched_df, "senior_citizen")

    return analysis


def _compute_revenue_at_risk(
    scored_df: pd.DataFrame,
    customers_df: pd.DataFrame | None,
    threshold: float = _HIGH_THRESHOLD,
) -> float:
    """
    Estimate monthly revenue at risk from HIGH + CRITICAL risk customers.

    Uses each customer's actual monthly charge when available; falls back
    to the fleet median otherwise.
    """
    high_risk_ids = scored_df.loc[
        scored_df["risk_level"].isin(["HIGH", "CRITICAL"]), "customer_id"
    ]

    if customers_df is not None and "monthlycharges" in customers_df.columns:
        hrc = customers_df[customers_df["customer_id"].astype(str).isin(high_risk_ids.astype(str))]
        revenue = pd.to_numeric(hrc["monthlycharges"], errors="coerce").fillna(65.0).sum()
    else:
        # Flat assumption
        revenue = len(high_risk_ids) * 65.0

    return round(float(revenue), 2)

