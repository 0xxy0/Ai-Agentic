"""
Training service.

Handles data loading, label generation, and orchestrates trainer.py.
The API route and pipeline scripts both delegate to this service.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.logger import get_logger
from app.features.builder import build_feature_dataframe, FEATURE_COLUMNS
from app.models.trainer import train_churn_model

logger = get_logger(__name__)


def _generate_churn_label(
    df: pd.DataFrame, window_days: int = 30
) -> pd.Series:
    """
    Derive a binary churn label.

    A user is marked as churned (1) if their ``recency_days`` exceeds
    the churn window — i.e., they have not been active within the window.

    Args:
        df: DataFrame with a ``recency_days`` column.
        window_days: Churn definition window (30 | 60 | 90).

    Returns:
        A binary :class:`pandas.Series` (0 = retained, 1 = churned).
    """
    if "recency_days" not in df.columns:
        raise ValueError("DataFrame must contain 'recency_days' to derive churn label")
    label = (df["recency_days"] > window_days).astype(int)
    churn_rate = label.mean()
    logger.info(
        "Churn label generated",
        extra={"window_days": window_days, "churn_rate": round(churn_rate, 4)},
    )
    return label


class TrainingService:
    """Orchestrates end-to-end model training from raw data."""

    def train_from_dataframe(
        self,
        raw_df: pd.DataFrame,
        window_days: int | None = None,
        xgb_params: dict[str, Any] | None = None,
        register_model: bool = True,
    ) -> dict[str, Any]:
        """
        Run full training pipeline from a pre-loaded DataFrame.

        Args:
            raw_df: Raw user activity DataFrame (un-engineered).
            window_days: Churn window; falls back to ``settings.CHURN_WINDOW_DAYS``.
            xgb_params: Optional XGBoost hyperparameter overrides.
            register_model: Whether to register the model in MLflow Registry.

        Returns:
            ``{"run_id": str, "status": str, "metrics": {...}}``
        """
        window = window_days or settings.CHURN_WINDOW_DAYS
        logger.info(
            "Starting training pipeline",
            extra={"n_records": len(raw_df), "churn_window_days": window},
        )

        # Feature engineering
        feature_df = build_feature_dataframe(raw_df.to_dict(orient="records"))

        # Generate labels — sort by a time column if available, else trust row order
        if "event_date" in raw_df.columns:
            raw_df = raw_df.sort_values("event_date").reset_index(drop=True)
            feature_df = feature_df.loc[raw_df.index]

        feature_df["churn_label"] = _generate_churn_label(
            raw_df.reset_index(drop=True), window_days=window
        ).values

        run_id = train_churn_model(
            df=feature_df,
            target_col="churn_label",
            xgb_params=xgb_params,
            register_model=register_model,
        )

        return {
            "run_id": run_id,
            "status": "success",
            "n_samples": len(feature_df),
            "churn_window_days": window,
        }

    def train_from_csv(
        self,
        csv_path: str | Path | None = None,
        window_days: int | None = None,
        xgb_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Convenience wrapper: load CSV then call :meth:`train_from_dataframe`.

        Args:
            csv_path: Path to the CSV file; defaults to ``settings.BATCH_INPUT_PATH``.
            window_days: Churn window.
            xgb_params: XGBoost overrides.

        Returns:
            Same as :meth:`train_from_dataframe`.
        """
        path = Path(csv_path or settings.BATCH_INPUT_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Training data not found at {path}")

        logger.info("Loading training CSV", extra={"path": str(path)})
        raw_df = pd.read_csv(path)

        return self.train_from_dataframe(raw_df, window_days=window_days, xgb_params=xgb_params)


# Module-level singleton
training_service = TrainingService()
