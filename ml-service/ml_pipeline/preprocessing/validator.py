"""
preprocessing/validator.py  —  Input schema validation
===================================================
Ensures the incoming JSON matches the expected ML schema.
"""

import pandas as pd
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# The source of truth for what the model needs (Raw features before engineering)
REQUIRED_FEATURES = [
    "customer_id",
    "monthlycharges",
    "totalcharges",
    "tenure",
    "days_since_last_interaction",
    "customer_satisfaction",
    "num_complaints"
]

# Map of expected dtypes for basic validation
DTYPE_MAP = {
    "monthlycharges": "float64",
    "totalcharges": "float64",
    "tenure": "int64",
    "age": "int64"
}

class SchemaValidationError(Exception):
    """Raised when the input data violates the required schema."""
    pass

def validate_input(df: pd.DataFrame) -> bool:
    """
    Validates if the DataFrame has the minimal required features.
    Logs warnings for missing optional features.
    """
    if df.empty:
        raise SchemaValidationError("Input data is empty")

    # 1. Check for required columns
    missing = [col for col in REQUIRED_FEATURES if col not in df.columns]
    if missing:
        # We raise error if critical features are missing
        # However, for this demo, we might want to be more lenient if we have fallbacks
        # For now, let's be strict to ensure data quality
        logger.error(f"Missing required features: {missing}")
        raise SchemaValidationError(f"Required columns missing: {missing}")

    # 2. Check for unexpected columns (Informational only)
    # Production models should ignore unknown columns
    known_cols = set(REQUIRED_FEATURES)
    unknown = [col for col in df.columns if col not in known_cols and col not in ["name", "segment", "subscription_plan", "current_status"]]
    if unknown:
        logger.info(f"Ignoring unknown input columns: {unknown}")

    return True

def align_features(df: pd.DataFrame, trained_columns: List[str]) -> pd.DataFrame:
    """
    Ensures the final feature set matches the EXACT order and set of 
    columns the model was trained on.
    """
    df = df.copy()
    
    # 1. Add missing columns with default values (0)
    for col in trained_columns:
        if col not in df.columns:
            logger.warning(f"Feature '{col}' missing from processed data. Filling with 0.")
            df[col] = 0.0
            
    # 2. Reorder and filter to exactly match trained_columns
    return df[trained_columns]
