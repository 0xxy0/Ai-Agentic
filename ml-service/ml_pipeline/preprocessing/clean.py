"""
preprocessing/clean.py  —  Production-grade data cleaning
=====================================================
Optimized for single-row inference and batch processing.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Constants
_BINARY_MAP = {"Yes": 1, "No": 0}

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input DataFrame with production-grade safety.
    Handles dtypes, missing values, and outlier removal.
    """
    if df.empty:
        return df
        
    df = df.copy()
    
    # 1. Ensure unique columns (fix for the duplicate column bug)
    if df.columns.duplicated().any():
        logger.warning(f"Duplicate columns detected: {df.columns[df.columns.duplicated()].tolist()}. Keeping first.")
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # 2. Parse dates
    date_cols = ["signup_date", "last_active_date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 3. Numeric conversion and imputation
    # We select all columns that SHOULD be numeric
    numeric_candidates = [
        "age", "annual_income", "tenure", "monthlycharges", "totalcharges",
        "num_services", "customer_satisfaction", "num_complaints",
        "num_service_calls", "late_payments", "avg_monthly_gb",
        "days_since_last_interaction", "credit_score", "dependents",
        "senior_citizen", "predicted_churn_probability", "predicted_revenue_loss"
    ]
    
    for col in numeric_candidates:
        if col in df.columns:
            # Handle currency symbols and percentages if they leak into numeric fields
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(r'[$,%]', '', regex=True)
            
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Impute with 0 or median (for single row, median doesn't make sense, so we use 0 or common sense defaults)
            if df[col].isna().any():
                fill_val = 0.0
                if col in ["age"]: fill_val = 35.0
                if col in ["customer_satisfaction"]: fill_val = 3.0
                df[col] = df[col].fillna(fill_val)

    # 4. Binary Encoding (Yes/No)
    # Safely identify object columns and map Yes/No
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        # Check if column is likely binary
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({"Yes", "No", "YES", "NO", "yes", "no"}):
            df[col] = df[col].astype(str).str.capitalize().map(_BINARY_MAP).fillna(0).astype(int)

    # 5. Outlier clipping (instead of dropping rows during inference)
    # Production inference shouldn't drop the row the user just submitted!
    # Instead, we clip to a reasonable range.
    if "monthlycharges" in df.columns:
        df["monthlycharges"] = df["monthlycharges"].clip(0, 500)
    if "totalcharges" in df.columns:
        df["totalcharges"] = df["totalcharges"].clip(0, 50000)

    # 6. Ensure churn label (if present) is clean
    if "churn" in df.columns:
        df["churn"] = pd.to_numeric(df["churn"], errors="coerce").fillna(0).astype(int)

    return df
