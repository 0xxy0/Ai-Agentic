"""
Feature Agent — Phase 2 of the telecom churn multi-agent pipeline.

Responsibility:
    Transform the clean 32-column customer DataFrame produced by the
    IngestionAgent into a rich, model-ready feature matrix.

    Produces ~59 engineered features covering:
        * Demographics (age, senior flag, education, marital status)
        * Financial (income, charges, credit score — with log transforms)
        * Tenure & contract (buckets, risk scores, contract type)
        * Payment (method one-hot, paperless flag)
        * Services (bundle counts, individual service flags)
        * Behavioural (satisfaction, complaints, service calls, late payments)
        * Derived rates (complaint rate/tenure, service call rate/tenure, etc.)
        * Composite scores (protection bundle, entertainment bundle, risk index)

Input context keys consumed:
    ``customers_df``        — cleaned 32-column customer DataFrame

Output context keys added:
    ``feature_df``          — feature matrix indexed by ``customer_id``
    ``churn_labels``        — binary Series aligned to ``feature_df``
    ``feature_columns``     — ordered list of model input columns
    ``feature_stats``       — per-feature summary statistics
    ``mi_preview``          — dict of top-N mutual-information scores with
                              the churn target

Contract:
    The agent ONLY engineers features.
    It does NOT load data, train models, or write files.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.core.logger import get_logger
from tools.feature_tools import (
    CHURN_FEATURE_COLUMNS,
    compute_mutual_information_preview,
    engineer_churn_features,
)

logger = get_logger(__name__)


class FeatureAgent:
    """
    Agent 2/5: Builds a rich ~59-feature matrix from the 32-column
    telecom churn customer DataFrame.

    Features produced (grouped by category):
        Demographics:  age, senior_citizen, dependents, gender_male,
                       education_ordinal/one-hot, marital status one-hot
        Financial:     annual_income, monthlycharges, totalcharges,
                       credit_score — raw + log-transformed
        Contract:      tenure, tenure_bucket (0–3), contract_one_year,
                       contract_two_year, contract_risk_score
        Payment:       payment_bank_transfer/credit_card/mailed_check,
                       paperless_billing
        Services:      num_services, 8 service-type flags
        Behavioural:   customer_satisfaction, num_complaints,
                       num_service_calls, late_payments, avg_monthly_gb,
                       days_since_last_interaction
        Derived rates: complaint_rate, service_call_rate, late_payment_rate,
                       charge_ratio, charge_per_service, tenure_charge_ratio,
                       revenue_per_gb, income_to_charge_ratio
        Normalised:    satisfaction_normalized, recency_normalized,
                       age_normalized, credit_score_normalized
        Composite:     protection_score, entertainment_score,
                       service_bundle_density, digital_engagement,
                       risk_composite

    Usage::

        agent = FeatureAgent()
        ctx   = agent.run(ctx)
        # ctx["feature_df"]     → pd.DataFrame, index=customer_id
        # ctx["churn_labels"]   → pd.Series (0/1), index=customer_id
        # ctx["feature_columns"] → list[str]
        # ctx["mi_preview"]     → top-15 mutual-info scores

    """

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute feature engineering.

        Args:
            context: Pipeline context containing ``customers_df``.

        Returns:
            Enriched context with ``feature_df``, ``churn_labels``,
            ``feature_columns``, ``feature_stats``, and ``mi_preview``.

        Raises:
            KeyError: If ``customers_df`` is absent from the context.
        """
        logger.info("FeatureAgent started")

        customers_df: pd.DataFrame = context.get("customers_df")
        if customers_df is None:
            raise KeyError(
                "FeatureAgent requires 'customers_df' in context "
                "(run IngestionAgent first)."
            )

        # ── 1. Extract churn labels before feature engineering ────────────────
        if "churn" not in customers_df.columns:
            raise KeyError(
                "customers_df is missing the 'churn' target column. "
                "Ensure the dataset includes a binary 0/1 'churn' column."
            )
        churn_labels = (
            pd.to_numeric(customers_df["churn"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        if "customer_id" in customers_df.columns:
            churn_labels.index = customers_df["customer_id"].values

        churn_rate = float(churn_labels.mean())
        logger.info(
            "Churn labels extracted",
            extra={
                "n_churned":   int(churn_labels.sum()),
                "n_retained":  int((churn_labels == 0).sum()),
                "churn_rate":  round(churn_rate, 4),
            },
        )

        # ── 2. Feature engineering ────────────────────────────────────────────
        feature_df = engineer_churn_features(customers_df)

        # Guard: ensure all registered columns are present
        missing_cols = set(CHURN_FEATURE_COLUMNS) - set(feature_df.columns)
        if missing_cols:
            logger.warning(
                "Backfilling missing engineered features with 0",
                extra={"missing": sorted(missing_cols)},
            )
            for col in missing_cols:
                feature_df[col] = 0.0

        feature_df = feature_df[CHURN_FEATURE_COLUMNS]

        # ── 3. Feature statistics (for dashboard / debugging) ─────────────────
        feature_stats = _compute_feature_stats(feature_df)

        # ── 4. Mutual-information preview ─────────────────────────────────────
        # Align labels to feature_df index
        aligned_y = churn_labels.reindex(feature_df.index).fillna(0).astype(int)
        mi_preview = compute_mutual_information_preview(
            feature_df, aligned_y, top_n=15
        )

        # ── 5. Populate context ───────────────────────────────────────────────
        context["feature_df"]      = feature_df
        context["churn_labels"]    = aligned_y
        context["feature_columns"] = CHURN_FEATURE_COLUMNS
        context["feature_stats"]   = feature_stats
        context["mi_preview"]      = mi_preview

        logger.info(
            "FeatureAgent complete",
            extra={
                "customers":    len(feature_df),
                "features":     len(CHURN_FEATURE_COLUMNS),
                "churn_rate":   round(churn_rate, 4),
                "top_mi_feature": list(mi_preview.keys())[:3] if mi_preview else [],
            },
        )
        return context


# ── Private helpers ───────────────────────────────────────────────────────────


def _compute_feature_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute per-feature descriptive statistics for logging / reports."""
    stats: dict[str, dict[str, float]] = {}
    for col in df.columns:
        s = df[col].dropna()
        stats[col] = {
            "mean":   round(float(s.mean()),   4) if len(s) else 0.0,
            "std":    round(float(s.std()),    4) if len(s) > 1 else 0.0,
            "min":    round(float(s.min()),    4) if len(s) else 0.0,
            "max":    round(float(s.max()),    4) if len(s) else 0.0,
            "null_pct": round(float(df[col].isnull().mean()), 4),
        }
    return stats

