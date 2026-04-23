"""
Data tools — low-level utilities for loading and normalising raw data.

All functions are stateless and side-effect free so they can be safely
imported and called from any agent or pipeline script.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from app.core.logger import get_logger

logger = get_logger(__name__)

# ── Expected raw column names ─────────────────────────────────────────────────
_REQUIRED_COLS: set[str] = {"user_id", "month", "year", "txn_count", "spend"}


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
