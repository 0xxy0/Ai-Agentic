"""
Ingestion Agent — Phase 1 of the multi-agent pipeline.

Responsibility:
    Load raw data from a CSV or SQLite source, validate it, and convert
    it into the canonical time-series format used by downstream agents.

Input context keys (optional):
    ``csv_path``   — path to a CSV file
    ``db_path``    — path to a SQLite file
    ``db_table``   — table name inside the SQLite file

Output context keys added:
    ``ts_df``      — normalised time-series DataFrame
    ``n_users``    — number of unique users
    ``n_rows``     — number of rows in the time-series DataFrame

Contract:
    The agent ONLY touches data loading and normalisation.
    It does NOT engineer features or modify model state.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.core.config import settings
from app.core.logger import get_logger
from tools.data_tools import load_csv, load_sqlite, to_time_series

logger = get_logger(__name__)


class IngestionAgent:
    """
    Agent 1/5: Loads raw activity data and converts it to time-series format.

    Usage::

        agent = IngestionAgent()
        ctx   = agent.run({"csv_path": "data/activity.csv"})
        # ctx["ts_df"] → cleaned, sorted DataFrame

    """

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute ingestion.

        Args:
            context: Pipeline context dict.  Must contain EITHER
                ``csv_path`` OR ``db_path`` (+ optional ``db_table``).

        Returns:
            Enriched context with ``ts_df``, ``n_users``, ``n_rows``.

        Raises:
            ValueError: If neither source is specified.
            FileNotFoundError: If the specified source does not exist.
        """
        logger.info("IngestionAgent started")

        csv_path = context.get("csv_path") or settings.BATCH_INPUT_PATH
        db_path  = context.get("db_path")
        db_table = context.get("db_table", "activity")

        # ── Load raw data ────────────────────────────────────────────────────
        if db_path:
            logger.info("Loading from SQLite", extra={"db_path": db_path, "table": db_table})
            raw_df = load_sqlite(db_path, table=db_table)
        elif csv_path:
            logger.info("Loading from CSV", extra={"csv_path": csv_path})
            raw_df = load_csv(csv_path)
        else:
            raise ValueError(
                "IngestionAgent requires 'csv_path' or 'db_path' in context."
            )

        # ── Normalise to time-series format ──────────────────────────────────
        ts_df = to_time_series(raw_df)

        context["ts_df"]   = ts_df
        context["n_users"] = ts_df["user_id"].nunique()
        context["n_rows"]  = len(ts_df)

        logger.info(
            "IngestionAgent complete",
            extra={"n_users": context["n_users"], "n_rows": context["n_rows"]},
        )
        return context
