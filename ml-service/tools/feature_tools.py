"""
Feature tools — stateless functions for time-series feature computation.

Each function takes a per-user time-ordered DataFrame slice and returns
a scalar or Series.  The FeatureAgent calls these in a vectorised pipeline.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.core.logger import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_EPS            = 1e-9     # Division-by-zero guard
_MAX_GAP_DAYS   = 365      # Cap activity gap outliers
_MONETARY_IQR_K = 5        # IQR multiplier for spend outlier capping


# ── Per-user feature helpers ──────────────────────────────────────────────────


def compute_txn_trend(series: pd.Series) -> float:
    """
    Compute the linear trend (slope) of transaction counts over time.

    Positive slope → growing engagement.
    Negative slope → declining engagement (churn signal).

    Args:
        series: Ordered :class:`pandas.Series` of ``txn_count`` values.

    Returns:
        OLS slope as a float (0.0 if fewer than 2 data points).
    """
    n = len(series)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    slope, *_ = stats.linregress(x, series.values.astype(float))
    return float(slope)


def compute_spend_trend(series: pd.Series) -> float:
    """
    Compute the linear trend (slope) of spend over time.

    Args:
        series: Ordered :class:`pandas.Series` of ``spend`` values.

    Returns:
        OLS slope (0.0 if fewer than 2 points).
    """
    return compute_txn_trend(series)   # same logic, reuse


def compute_activity_gap(periods: pd.Series, reference_period: pd.Period) -> int:
    """
    Compute the number of months since the user's last recorded activity.

    Args:
        periods: Ordered :class:`pandas.Series` of :class:`pandas.Period` values.
        reference_period: The "current" period used as reference (e.g. max period in dataset).

    Returns:
        Integer gap in months (capped at 24 to limit outlier influence).
    """
    if periods.empty:
        return 24
    last_active = periods.max()
    gap = (reference_period - last_active).n
    return min(max(int(gap), 0), 24)


def rolling_mean(series: pd.Series, window: int) -> float:
    """
    Return the mean of the last ``window`` values in ``series``.

    Args:
        series: Ordered numeric Series.
        window: Look-back window size.

    Returns:
        Float mean; 0.0 if series is empty.
    """
    tail = series.tail(window)
    if tail.empty:
        return 0.0
    return float(tail.mean())


# ── Batch feature builder ─────────────────────────────────────────────────────




def build_feature_dataframe(
    ts_df: pd.DataFrame,
    reference_period: pd.Period | None = None,
) -> pd.DataFrame:
    """
    Build a flat feature matrix from a full time-series DataFrame.

    Groups by ``user_id``, computes :func:`build_user_features` per user,
    then applies outlier capping on ``total_spend``.

    Args:
        ts_df: Normalised time-series DataFrame (output of ``data_tools.to_time_series``).
        reference_period: Override the reference period; defaults to ``ts_df["period"].max()``.

    Returns:
        Feature :class:`pandas.DataFrame` with one row per user.
    """
    if reference_period is None:
        reference_period = ts_df["period"].max()

    records: list[dict[str, Any]] = []
    for user_id, grp in ts_df.groupby("user_id"):
        grp = grp.sort_values("period")
        feat = build_user_features(grp, reference_period)
        feat["user_id"] = user_id
        records.append(feat)

    feat_df = pd.DataFrame(records)

    # IQR-based outlier capping on spend
    if len(feat_df) >= 4:
        for col in ("total_spend", "spend_last_month", "spend_3_month_avg", "spend_6_month_avg"):
            if col in feat_df.columns:
                q1, q3 = feat_df[col].quantile([0.25, 0.75])
                cap = q3 + _MONETARY_IQR_K * (q3 - q1)
                feat_df[col] = feat_df[col].clip(upper=cap)

    logger.info(
        "Feature matrix built",
        extra={"users": len(feat_df), "features": list(feat_df.columns)},
    )
    return feat_df.set_index("user_id")


# ── Feature column registry (model input contract) ───────────────────────────

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


def build_user_features(
    user_df: pd.DataFrame,
    reference_period: pd.Period,
) -> dict[str, Any]:
    """
    Compute 12 summary features for a user from their time-series history.
    """
    txn = user_df["txn_count"]
    spend = user_df["spend"]
    periods = user_df["period"]

    # Current month (last period in user data)
    txn_30d = float(txn.iloc[-1]) if not txn.empty else 0.0
    monetary = float(spend.sum())
    freq = float(txn.sum())
    
    # 7d is approximated as 1/4 of the current month in this synthetic setup
    txn_7d = txn_30d * 0.25 
    
    # 90d is last 3 months
    txn_90d = float(txn.tail(3).sum())
    
    recency = compute_activity_gap(periods, reference_period) * 30 # convert to approx days
    
    usage_decay = txn_7d / (txn_30d + _EPS)
    ratio_30_90 = txn_30d / (txn_90d + _EPS)
    
    # account age (diff between first and reference period)
    first_active = periods.min()
    age_days = (reference_period - first_active).n * 30

    return {
        "txn_7d":             txn_7d,
        "txn_30d":            txn_30d,
        "txn_90d":            txn_90d,
        "recency_days":       float(recency),
        "frequency":          freq,
        "monetary":           monetary,
        "usage_decay":        usage_decay,
        "txn_30d_90d_ratio":  ratio_30_90,
        "log_monetary":       math.log1p(monetary),
        "log_frequency":      math.log1p(freq),
        "account_age_days":   float(age_days),
        "recency_normalized": recency / 365.0,
    }
