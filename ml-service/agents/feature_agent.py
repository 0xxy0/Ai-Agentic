"""
Feature Agent — Phase 2 of the multi-agent pipeline.

Responsibility:
    Transform the time-series DataFrame produced by the IngestionAgent
    into a flat, model-ready feature matrix per user.

Input context keys consumed:
    ``ts_df``            — normalised time-series DataFrame

Output context keys added:
    ``feature_df``       — flat feature matrix (index = user_id)
    ``feature_columns``  — ordered list of model input columns
    ``reference_period`` — the latest period used for gap calculations

Contract:
    The agent ONLY computes features.
    It does NOT load data, train models, or write files.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.core.logger import get_logger
from tools.feature_tools import FEATURE_COLUMNS, build_feature_dataframe

logger = get_logger(__name__)


class FeatureAgent:
    """
    Agent 2/5: Builds a flat feature matrix from the time-series DataFrame.

    Features produced per user:
        * ``txn_last_month``     — transactions in the most recent month
        * ``txn_3_month_avg``    — rolling 3-month average
        * ``txn_6_month_avg``    — rolling 6-month average
        * ``txn_trend``          — OLS slope of transaction series
        * ``spend_last_month``   — spend in the most recent month
        * ``spend_3_month_avg``  — rolling 3-month avg spend
        * ``spend_6_month_avg``  — rolling 6-month avg spend
        * ``spend_trend``        — OLS slope of spend series
        * ``total_txn``          — lifetime transaction count
        * ``total_spend``        — lifetime spend
        * ``log_total_spend``    — log1p-transformed spend
        * ``activity_gap``       — months since last activity
        * ``n_active_months``    — number of months with activity
        * ``spend_per_txn``      — average spend per transaction

    Usage::

        agent = FeatureAgent()
        ctx   = agent.run(ctx)
        # ctx["feature_df"]  →  pd.DataFrame, index=user_id

    """

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute feature engineering.

        Args:
            context: Pipeline context containing ``ts_df``.

        Returns:
            Enriched context with ``feature_df``, ``feature_columns``,
            and ``reference_period``.

        Raises:
            KeyError: If ``ts_df`` is absent from the context.
        """
        logger.info("FeatureAgent started")

        ts_df: pd.DataFrame = context.get("ts_df")
        if ts_df is None:
            raise KeyError("FeatureAgent requires 'ts_df' in context (run IngestionAgent first).")

        reference_period = ts_df["period"].max()

        feature_df = build_feature_dataframe(ts_df, reference_period=reference_period)

        # Ensure only registered model features are kept (in correct order)
        missing = set(FEATURE_COLUMNS) - set(feature_df.columns)
        if missing:
            logger.warning("Missing feature columns — filling with 0", extra={"missing": missing})
            for col in missing:
                feature_df[col] = 0.0

        context["feature_df"]       = feature_df
        context["feature_columns"]  = FEATURE_COLUMNS
        context["reference_period"] = reference_period

        logger.info(
            "FeatureAgent complete",
            extra={
                "users":    len(feature_df),
                "features": len(FEATURE_COLUMNS),
                "ref_period": str(reference_period),
            },
        )
        return context
