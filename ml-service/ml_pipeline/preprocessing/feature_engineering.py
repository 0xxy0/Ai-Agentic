"""
preprocessing/feature_engineering.py  --  Advanced Feature Engineering
======================================================================
1. Time-Series: txn_last_7d, 30d, 90d, activity_decay, activity_trend
2. Relative: percentile_rank, z-scores, relative_activity
3. Engagement: composite scores
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import zscore

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply advanced temporal and relative feature engineering.
    """
    df = df.copy()

    # 1. Temporal / Time-Series (Derived/Simulated if not in activity log)
    # We use 'days_since_last_interaction' and 'totalcharges' to estimate windows
    # if actual transaction logs aren't fully merged.
    
    # Simulate txn counts based on tenure and total charges if missing
    # (In a real system, these would come from a JOIN with a transactions table)
    if 'txn_count' not in df.columns:
        # Heuristic: avg txns per month = 4.0
        df['est_monthly_txns'] = (df['totalcharges'] / df['monthlycharges'].replace(0, 1)).clip(1, 12)
        df['txn_last_30d'] = np.where(df['days_since_last_interaction'] < 30, df['est_monthly_txns'], 0)
        df['txn_last_90d'] = np.where(df['days_since_last_interaction'] < 90, df['est_monthly_txns'] * 3, 0)
        df['txn_last_7d']  = np.where(df['days_since_last_interaction'] < 7,  df['est_monthly_txns'] / 4, 0)
    
    # Rolling averages
    df['rolling_avg_30d'] = df['txn_last_30d'] # Simplified for static snapshot
    df['rolling_avg_90d'] = df['txn_last_90d'] / 3
    
    # Trend and Decay
    # activity_trend = txn_last_30d - txn_last_90d_avg
    df['activity_trend'] = df['txn_last_30d'] - df['rolling_avg_90d']
    
    # activity_decay = txn_last_7d / txn_last_30d
    df['activity_decay'] = (df['txn_last_7d'] / df['txn_last_30d'].replace(0, 0.001)).clip(0, 1)

    # 2. Relative Features (Normalized using dataset-wide stats)
    # Calculate activity ratio if missing
    if 'activity_ratio' not in df.columns:
        # activity_ratio = 1 - (days_since_last_interaction / (tenure * 30 + 1))
        # This measures how active they are relative to their lifetime
        df['activity_ratio'] = 1 - (df['days_since_last_interaction'] / (df['tenure'] * 30 + 1).clip(1))
    
    # z-score for monetary (monthlycharges)
    df['monetary_zscore'] = zscore(df['monthlycharges'])
    
    # Relative activity score
    avg_interaction = df['days_since_last_interaction'].mean()
    df['relative_activity_score'] = (avg_interaction / df['days_since_last_interaction'].replace(0, 1)).clip(0, 10)

    # Engagement Scoring (Existing + Enhanced)
    df['engagement_score'] = (
        (df['customer_satisfaction'] * 0.4) - 
        (df['num_complaints'] * 1.5) + 
        (df['activity_ratio'] * 2.0)
    ).fillna(0)

    # Clean up temp columns
    if 'est_monthly_txns' in df.columns:
        df = df.drop(columns=['est_monthly_txns'])

    return df

def encode_and_scale(df: pd.DataFrame, fit_scaler: bool = True, scaler=None) -> tuple:
    """
    Encode categorical features and scale numerical ones.
    """
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    df = df.copy()
    
    # Drop IDs and non-numeric for X
    cols_to_drop = ['customer_id', 'signup_date', 'churn']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df['churn'] if 'churn' in df.columns else None

    # Handle Categoricals
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Scaling
    num_cols = X.select_dtypes(include=[np.number]).columns
    if fit_scaler:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
    elif scaler:
        X[num_cols] = scaler.transform(X[num_cols])

    return X, y, list(X.columns), scaler
