"""
Feature engineering pipeline.

Transforms raw user-activity records into a clean, model-ready feature
dictionary.  All transformations are deterministic and stateless so they
can run identically during training and inference.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.core.logger import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_EPSILON = 1e-9          # Prevent division-by-zero
_RECENCY_CAP = 365       # Cap recency outliers at 1 year
_MONETARY_CAP_FACTOR = 5 # Cap monetary at 5× IQR (applied per batch)

# Feature names in the exact order expected by the model
FEATURE_COLUMNS: list[str] = [
    "txn_7d",
    "txn_30d",
    "txn_90d",
    "recency_days",
    "frequency",
    "monetary",
    "usage_decay",
    "txn_30d_90d_ratio",
    "log_monetary",
    "log_frequency",
    "account_age_days",
    "recency_normalized",
]


def build_single_feature_dict(raw: dict[str, Any]) -> dict[str, float]:
    """
    Build a model-ready feature dictionary from a single raw activity record.

    Args:
        raw: Dictionary matching :class:`~app.features.validators.RawUserActivity`.

    Returns:
        A flat ``dict[str, float]`` with all engineered features.

    Raises:
        ValueError: If required keys are missing.
    """
    required = {"txn_7d", "txn_30d", "txn_90d", "recency_days", "frequency", "monetary"}
    missing = required - set(raw.keys())
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    txn_7d   = float(raw["txn_7d"])
    txn_30d  = float(raw["txn_30d"])
    txn_90d  = float(raw["txn_90d"])
    recency  = min(float(raw["recency_days"]), _RECENCY_CAP)
    freq     = max(float(raw.get("frequency", 0)), 0.0)
    monetary = max(float(raw.get("monetary", 0.0)), 0.0)
    acct_age = float(raw.get("account_age_days") or 0.0)

    # ── Derived features ──────────────────────────────────────────────────
    usage_decay       = txn_7d / (txn_30d + _EPSILON)
    txn_30d_90d_ratio = txn_30d / (txn_90d + _EPSILON)
    log_monetary      = math.log1p(monetary)
    log_frequency     = math.log1p(freq)
    recency_normalized = recency / _RECENCY_CAP

    features = {
        "txn_7d":             txn_7d,
        "txn_30d":            txn_30d,
        "txn_90d":            txn_90d,
        "recency_days":       recency,
        "frequency":          freq,
        "monetary":           monetary,
        "usage_decay":        usage_decay,
        "txn_30d_90d_ratio":  txn_30d_90d_ratio,
        "log_monetary":       log_monetary,
        "log_frequency":      log_frequency,
        "account_age_days":   acct_age,
        "recency_normalized": recency_normalized,
    }

    logger.debug("Built features for single record", extra={"features": features})
    return features


def build_feature_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Vectorised feature engineering for a batch of records.

    Applies the same transformations as :func:`build_single_feature_dict`
    but uses pandas for efficiency on large batches.  Also applies
    IQR-based outlier capping on the monetary column.

    Args:
        records: List of raw activity dictionaries.

    Returns:
        A :class:`pandas.DataFrame` with columns matching :data:`FEATURE_COLUMNS`.
    """
    df = pd.DataFrame(records)

    # ── Impute missing values ─────────────────────────────────────────────
    numeric_cols = ["txn_7d", "txn_30d", "txn_90d", "frequency", "monetary",
                    "account_age_days"]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "recency_days" not in df.columns:
        df["recency_days"] = 0
    df["recency_days"] = pd.to_numeric(
        df["recency_days"], errors="coerce"
    ).fillna(0).clip(upper=_RECENCY_CAP)

    # ── Outlier capping on monetary (IQR method) ──────────────────────────
    if len(df) >= 4:
        q1 = df["monetary"].quantile(0.25)
        q3 = df["monetary"].quantile(0.75)
        iqr = q3 - q1
        upper_cap = q3 + _MONETARY_CAP_FACTOR * iqr
        df["monetary"] = df["monetary"].clip(upper=upper_cap)

    # ── Derived features ──────────────────────────────────────────────────
    df["usage_decay"]        = df["txn_7d"] / (df["txn_30d"] + _EPSILON)
    df["txn_30d_90d_ratio"]  = df["txn_30d"] / (df["txn_90d"] + _EPSILON)
    df["log_monetary"]       = np.log1p(df["monetary"])
    df["log_frequency"]      = np.log1p(df["frequency"])
    df["recency_normalized"] = df["recency_days"] / _RECENCY_CAP

    logger.info(
        "Batch feature engineering complete",
        extra={"n_records": len(df), "columns": list(df.columns)},
    )
    return df[FEATURE_COLUMNS]
