"""
Ingestion Agent — Phase 1 of the telecom churn multi-agent pipeline.

Responsibility:
    Load the 32-column telecom churn CSV, perform thorough schema
    validation, compute a data-quality profile, impute missing values,
    and normalise column types so downstream agents receive a clean,
    analysis-ready DataFrame.

Input context keys (optional):
    ``csv_path``            — path to the churn CSV file (falls back to
                              ``settings.BATCH_INPUT_PATH``)

Output context keys added:
    ``customers_df``        — cleaned 32-column customer DataFrame
    ``n_customers``         — number of customer rows
    ``n_rows``              — alias for ``n_customers`` (pipeline compat)
    ``data_quality_report`` — dict with missing rates, stats, churn rate,
                              and a 0–100 quality score
    ``churn_rate``          — float: fraction of customers labelled churned

Contract:
    The agent ONLY loads, validates, imputes, and profiles data.
    It does NOT engineer features, train models, or modify state.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.core.config import settings
from app.core.logger import get_logger
from tools.data_tools import (
    impute_churn_data,
    load_churn_csv,
    profile_churn_data,
    validate_churn_schema,
    CHURN_SCHEMA_COLUMNS,
)

logger = get_logger(__name__)


class IngestionAgent:
    """
    Agent 1/5: Loads and validates the 32-column telecom churn dataset.

    Pipeline context flow::

        context["csv_path"]  →  IngestionAgent.run()
            ↓
        context["customers_df"]        — cleaned DataFrame
        context["n_customers"]         — row count
        context["data_quality_report"] — quality profile dict
        context["churn_rate"]          — fraction of churned customers

    The agent enforces the following data quality contract:
        * All required columns present
        * Numeric columns coerced to float/int (non-parseable → NaN → imputed)
        * Categorical columns stripped and normalised
        * Missing values imputed using column medians / modes
        * signup_date parsed to datetime (failures silently coerced to NaT)

    Usage::

        agent = IngestionAgent()
        ctx   = agent.run({"csv_path": "data/churn.csv"})
        df    = ctx["customers_df"]   # clean DataFrame, ready for FeatureAgent

    """

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute data ingestion and quality checking.

        Args:
            context: Pipeline context dict. Must contain ``csv_path`` or
                     ``settings.BATCH_INPUT_PATH`` must point to the churn CSV.

        Returns:
            Enriched context with ``customers_df``, ``n_customers``,
            ``data_quality_report``, and ``churn_rate``.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError:        If required churn schema columns are absent.
        """
        logger.info("IngestionAgent started")

        csv_path = context.get("csv_path") or settings.BATCH_INPUT_PATH

        # ── 1. Load raw CSV ───────────────────────────────────────────────────
        logger.info("Loading churn CSV", extra={"csv_path": str(csv_path)})
        raw_df = load_churn_csv(csv_path, parse_dates=True)

        # ── 2. Schema validation ──────────────────────────────────────────────
        validate_churn_schema(raw_df, source=str(csv_path))

        # ── 3. Pre-imputation data quality profile ────────────────────────────
        logger.info("Computing pre-imputation data quality profile")
        quality_report = profile_churn_data(raw_df)
        logger.info(
            "Pre-imputation quality profile",
            extra={
                "quality_score": quality_report["quality_score"],
                "churn_rate":    quality_report["churn_rate"],
                "n_rows":        quality_report["n_rows"],
                "high_missing":  len(quality_report["high_missing_cols"]),
            },
        )

        # ── 4. Impute missing values ──────────────────────────────────────────
        if quality_report["missing_rates"]:
            total_missing = sum(quality_report["missing_rates"].values())
            if total_missing > 0:
                logger.info("Imputing missing values", extra={"total_null_rate": round(total_missing, 4)})
        clean_df = impute_churn_data(raw_df)

        # ── 5. Type normalisation ─────────────────────────────────────────────
        clean_df = _normalise_types(clean_df)

        # ── 6. Post-imputation profile (final quality report) ─────────────────
        final_profile = profile_churn_data(clean_df)
        quality_report["post_imputation"] = {
            "quality_score":  final_profile["quality_score"],
            "remaining_nulls": int(clean_df.isnull().sum().sum()),
        }

        # ── 7. Populate context ───────────────────────────────────────────────
        churn_rate  = float(clean_df["churn"].mean())
        n_customers = len(clean_df)

        context["customers_df"]        = clean_df
        context["n_customers"]         = n_customers
        context["n_rows"]              = n_customers        # pipeline compatibility
        context["data_quality_report"] = quality_report
        context["churn_rate"]          = churn_rate

        logger.info(
            "IngestionAgent complete",
            extra={
                "n_customers": n_customers,
                "churn_rate":  round(churn_rate, 4),
                "quality_score": quality_report["quality_score"],
                "columns":     clean_df.shape[1],
            },
        )
        return context


# ── Private helpers ───────────────────────────────────────────────────────────

_BINARY_COLS = [
    "paperless_billing", "senior_citizen",
    "has_phone_service", "has_internet_service",
    "has_online_security", "has_online_backup",
    "has_device_protection", "has_tech_support",
    "has_streaming_tv", "has_streaming_movies",
    "churn",
]

_INT_COLS = [
    "age", "dependents", "tenure", "num_services",
    "customer_satisfaction", "num_complaints",
    "num_service_calls",
]

_FLOAT_COLS = [
    "annual_income", "monthlycharges", "totalcharges",
    "avg_monthly_gb", "late_payments",
    "days_since_last_interaction", "credit_score",
]

_STRING_COLS = [
    "gender", "education", "marital_status", "contract", "payment_method",
]


def _normalise_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to their canonical dtypes after imputation."""
    df = df.copy()

    for col in _BINARY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 1).astype(int)

    for col in _INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in _FLOAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

    for col in _STRING_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "signup_date" in df.columns:
        df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")

    return df

