"""
Data tools — low-level utilities for loading and normalising raw data.

Supports two dataset modes:

1. Legacy time-series activity data (``user_id``, ``month``, ``year``,
   ``txn_count``, ``spend``).
2. Telecom churn dataset — 32-column flat customer table with ``churn``
   as the binary target variable.

All functions are stateless and side-effect free so they can be safely
imported and called from any agent or pipeline script.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.core.logger import get_logger

logger = get_logger(__name__)

# ── Legacy column contract ────────────────────────────────────────────────────
_REQUIRED_COLS: set[str] = {"user_id", "month", "year", "txn_count", "spend"}

# ── Telecom churn schema ──────────────────────────────────────────────────────
CHURN_SCHEMA_COLUMNS: list[str] = [
    "customer_id", "signup_date", "age", "gender", "annual_income",
    "education", "marital_status", "dependents", "tenure", "contract",
    "payment_method", "paperless_billing", "senior_citizen",
    "monthlycharges", "totalcharges", "num_services",
    "has_phone_service", "has_internet_service", "has_online_security",
    "has_online_backup", "has_device_protection", "has_tech_support",
    "has_streaming_tv", "has_streaming_movies", "customer_satisfaction",
    "num_complaints", "num_service_calls", "late_payments",
    "avg_monthly_gb", "days_since_last_interaction", "credit_score", "churn",
]

# Minimum required subset (customer_id and target are always required)
_CHURN_REQUIRED: set[str] = {
    "customer_id", "tenure", "contract", "monthlycharges",
    "totalcharges", "churn",
}

# Columns that must be numeric
_CHURN_NUMERIC: list[str] = [
    "age", "annual_income", "dependents", "tenure", "paperless_billing",
    "senior_citizen", "monthlycharges", "totalcharges", "num_services",
    "has_phone_service", "has_internet_service", "has_online_security",
    "has_online_backup", "has_device_protection", "has_tech_support",
    "has_streaming_tv", "has_streaming_movies", "customer_satisfaction",
    "num_complaints", "num_service_calls", "late_payments",
    "avg_monthly_gb", "days_since_last_interaction", "credit_score", "churn",
]

# Columns that must be categorical strings
_CHURN_CATEGORICAL: list[str] = [
    "gender", "education", "marital_status", "contract", "payment_method",
]

# Imputation defaults (median/mode for each column)
_CHURN_DEFAULTS: dict[str, Any] = {
    "age": 35,
    "annual_income": 50_000.0,
    "dependents": 0,
    "tenure": 12,
    "paperless_billing": 0,
    "senior_citizen": 0,
    "monthlycharges": 65.0,
    "totalcharges": 780.0,
    "num_services": 2,
    "has_phone_service": 1,
    "has_internet_service": 1,
    "has_online_security": 0,
    "has_online_backup": 0,
    "has_device_protection": 0,
    "has_tech_support": 0,
    "has_streaming_tv": 0,
    "has_streaming_movies": 0,
    "customer_satisfaction": 3,
    "num_complaints": 0,
    "num_service_calls": 1,
    "late_payments": 0.0,
    "avg_monthly_gb": 20.0,
    "days_since_last_interaction": 30.0,
    "credit_score": 680.0,
    "gender": "Male",
    "education": "Bachelor",
    "marital_status": "Single",
    "contract": "Month-to-month",
    "payment_method": "Electronic check",
}


# ── Churn dataset loaders ────────────────────────────────────────────────────


def load_churn_csv(path: str | Path, parse_dates: bool = True) -> pd.DataFrame:
    """
    Load a 32-column telecom churn CSV into a DataFrame.

    Args:
        path:        Path to the CSV file.
        parse_dates: If True, attempt to parse the ``signup_date`` column.

    Returns:
        Raw :class:`pandas.DataFrame` with 32 columns.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Churn CSV not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    if parse_dates and "signup_date" in df.columns:
        df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")

    validate_churn_schema(df, source=str(path))
    logger.info(
        "Churn CSV loaded",
        extra={"path": str(path), "rows": len(df), "cols": df.shape[1]},
    )
    return df


def validate_churn_schema(df: pd.DataFrame, source: str = "<DataFrame>") -> None:
    """
    Validate that a DataFrame conforms to the 32-column churn schema.

    Args:
        df:     DataFrame to validate.
        source: Description for error messages.

    Raises:
        ValueError: If required columns are absent.
    """
    missing = _CHURN_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"Churn dataset from '{source}' is missing required columns: "
            f"{sorted(missing)}.  Present: {sorted(df.columns.tolist())}"
        )

    # Warn about extra columns beyond the 32-column schema
    extra = set(df.columns) - set(CHURN_SCHEMA_COLUMNS)
    if extra:
        logger.warning(
            "Churn dataset has unexpected columns (will be ignored)",
            extra={"extra_cols": sorted(extra)},
        )


def profile_churn_data(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute a comprehensive data quality profile for the churn dataset.

    Covers: missing rates, zero-variance columns, numeric range stats,
    churn rate, class imbalance ratio.

    Args:
        df: Raw or partially cleaned churn DataFrame.

    Returns:
        Dict with sub-keys ``missing``, ``dtypes``, ``stats``, ``churn_rate``,
        ``imbalance_ratio``, ``quality_score``.
    """
    n = len(df)

    # Missing value analysis
    missing_counts  = df.isnull().sum()
    missing_rates   = (missing_counts / n).round(4).to_dict()
    high_missing    = {c: r for c, r in missing_rates.items() if r > 0.05}

    # Numeric stats
    num_df   = df.select_dtypes(include=[np.number])
    num_stats: dict[str, Any] = {}
    for col in num_df.columns:
        s = num_df[col].dropna()
        num_stats[col] = {
            "mean":   round(float(s.mean()),  4) if len(s) else None,
            "std":    round(float(s.std()),   4) if len(s) > 1 else None,
            "min":    round(float(s.min()),   4) if len(s) else None,
            "max":    round(float(s.max()),   4) if len(s) else None,
            "q25":    round(float(s.quantile(0.25)), 4) if len(s) else None,
            "median": round(float(s.quantile(0.50)), 4) if len(s) else None,
            "q75":    round(float(s.quantile(0.75)), 4) if len(s) else None,
            "null_pct": round(float(missing_rates.get(col, 0.0)), 4),
        }

    # Churn statistics
    churn_rate       = 0.0
    imbalance_ratio  = 1.0
    if "churn" in df.columns:
        churn_series    = pd.to_numeric(df["churn"], errors="coerce").fillna(0)
        churn_rate      = float(churn_series.mean())
        n_pos           = int(churn_series.sum())
        n_neg           = n - n_pos
        imbalance_ratio = round(n_neg / n_pos, 2) if n_pos > 0 else float("inf")

    # Zero-variance columns
    zero_var = [c for c in num_df.columns if num_df[c].nunique() <= 1]

    # Quality score (0–100): penalise missing data and zero-variance
    missing_penalty   = min(len(high_missing) * 5, 30)
    zero_var_penalty  = min(len(zero_var) * 3, 15)
    quality_score     = max(0, 100 - missing_penalty - zero_var_penalty)

    profile: dict[str, Any] = {
        "n_rows":           n,
        "n_cols":           df.shape[1],
        "missing_rates":    missing_rates,
        "high_missing_cols": high_missing,
        "zero_variance_cols": zero_var,
        "numeric_stats":    num_stats,
        "churn_rate":       round(churn_rate, 4),
        "imbalance_ratio":  imbalance_ratio,
        "quality_score":    quality_score,
    }

    logger.info(
        "Data quality profile computed",
        extra={
            "n_rows":          n,
            "churn_rate":      profile["churn_rate"],
            "quality_score":   quality_score,
            "high_missing":    len(high_missing),
        },
    )
    return profile


def impute_churn_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in the churn dataset using column-specific defaults.

    Strategy:
        * Numeric cols  → column median (falls back to ``_CHURN_DEFAULTS`` if all NaN)
        * Categorical cols → column mode  (falls back to ``_CHURN_DEFAULTS``)

    Args:
        df: Raw churn DataFrame.

    Returns:
        DataFrame with no missing values in known columns.
    """
    df = df.copy()

    for col in _CHURN_NUMERIC:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isnull().any():
            fill = df[col].median()
            if pd.isna(fill):
                fill = _CHURN_DEFAULTS.get(col, 0)
            df[col] = df[col].fillna(fill)

    for col in _CHURN_CATEGORICAL:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str)
        df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})
        if df[col].isnull().any():
            mode_series = df[col].mode()
            fill = mode_series.iloc[0] if len(mode_series) else _CHURN_DEFAULTS.get(col, "Unknown")
            df[col] = df[col].fillna(fill)

    remaining_nulls = int(df.isnull().sum().sum())
    logger.info(
        "Imputation complete",
        extra={"remaining_nulls": remaining_nulls, "rows": len(df)},
    )
    return df


def load_csv(path: str | Path, parse_dates: bool = True) -> pd.DataFrame:
    """
    Load a raw activity CSV into a DataFrame.

    The CSV must contain at minimum:
        ``user_id``, ``month``, ``year``, ``txn_count``, ``spend``

    Args:
        path: Absolute or relative path to the CSV file.
        parse_dates: If True, try to parse a ``date`` column if present.

    Returns:
        Raw :class:`pandas.DataFrame`.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    if parse_dates and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    _validate_columns(df, path)
    logger.info("CSV loaded", extra={"path": str(path), "rows": len(df), "cols": list(df.columns)})
    return df


def load_sqlite(db_path: str | Path, table: str = "activity") -> pd.DataFrame:
    """
    Load user activity from a SQLite database table.

    Args:
        db_path: Path to the SQLite ``.db`` file.
        table: Table name to read from.

    Returns:
        Raw :class:`pandas.DataFrame`.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)  # noqa: S608
    finally:
        conn.close()

    _validate_columns(df, db_path)
    logger.info("SQLite loaded", extra={"db": str(db_path), "table": table, "rows": len(df)})
    return df


def _validate_columns(df: pd.DataFrame, source: Path) -> None:
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Source '{source}' is missing required columns: {missing}. "
            f"Got: {set(df.columns)}"
        )


def to_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a raw DataFrame into a canonical time-series format.

    Sorts by ``user_id`` → ``year`` → ``month`` so downstream agents
    always operate on temporally ordered data.

    Returns a DataFrame with columns:
        ``user_id``, ``year``, ``month``, ``period``, ``txn_count``, ``spend``

    where ``period`` is a :class:`pandas.Period` (monthly resolution).
    """
    df = df.copy()

    # Coerce numeric types
    df["txn_count"] = pd.to_numeric(df["txn_count"], errors="coerce").fillna(0)
    df["spend"]     = pd.to_numeric(df["spend"],     errors="coerce").fillna(0.0)
    df["year"]      = pd.to_numeric(df["year"],       errors="coerce").fillna(0).astype(int)
    df["month"]     = pd.to_numeric(df["month"],      errors="coerce").fillna(1).astype(int)

    # Build Period column for easy resampling / offset arithmetic
    df["period"] = df.apply(
        lambda r: pd.Period(year=int(r["year"]), month=int(r["month"]), freq="M"),
        axis=1,
    )

    df = df.sort_values(["user_id", "year", "month"]).reset_index(drop=True)

    logger.info("Time-series normalised", extra={"users": df["user_id"].nunique(), "rows": len(df)})
    return df[["user_id", "year", "month", "period", "txn_count", "spend"]]


def split_time_based(
    df: pd.DataFrame,
    train_end_month: int,
    train_end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time-series DataFrame chronologically.

    All rows where ``(year, month) <= (train_end_year, train_end_month)``
    go into the training set; the rest form the test set.

    Args:
        df: Normalised time-series DataFrame.
        train_end_month: Last month (1–12) included in training.
        train_end_year: Year of the last training month.

    Returns:
        ``(train_df, test_df)`` tuple.
    """
    cutoff = pd.Period(year=train_end_year, month=train_end_month, freq="M")
    train = df[df["period"] <= cutoff].copy()
    test  = df[df["period"] >  cutoff].copy()

    logger.info(
        "Time-based split",
        extra={
            "cutoff": str(cutoff),
            "train_rows": len(train),
            "test_rows": len(test),
        },
    )
    return train, test
