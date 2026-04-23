"""
models/predict.py  --  Advanced Segmentation & Decision Matrix
==============================================================
1. Hybrid Segmentation (VIP, Critical High Value, etc.)
2. Decision Matrix with Business Recommendations
3. Optimized Next-Month Prediction
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_OUT_DIR    = Path("outputs")
_MODEL_PATH = _OUT_DIR / "xgb_model.pkl"
_SEP = "=" * 70


def _get_segment(row: pd.Series, monetary_p80: float) -> str:
    """
    Advanced Segmentation Engine.
    """
    score    = row["churn_score"]
    decay    = row.get("activity_decay", 1.0)
    monetary = row.get("monthlycharges", 0)
    
    if score > 0.8 and monetary >= monetary_p80:
        return "CRITICAL_HIGH_VALUE"
    elif score > 0.7:
        return "HIGH_RISK"
    elif decay < 0.3:
        return "DECLINING"
    elif monetary >= monetary_p80:
        return "VIP"
    else:
        return "STABLE"

def _get_recommendation(segment: str) -> str:
    """Business recommendation based on segment."""
    mapping = {
        "CRITICAL_HIGH_VALUE": "Immediate 1-on-1 Retention Campaign",
        "HIGH_RISK":           "Aggressive Re-engagement Discount",
        "DECLINING":           "Personalized Engagement Push / Survey",
        "VIP":                 "Exclusive Premium Offers & Loyalty Perks",
        "STABLE":              "Nurture / Standard Communication"
    }
    return mapping.get(segment, "Standard Communication")

def predict_churn(df_clean: pd.DataFrame, X_scaled: pd.DataFrame, model=None) -> pd.DataFrame:
    """Score all customers and apply segmentation + recommendations."""
    if model is None:
        model = joblib.load(_MODEL_PATH)

    proba = model.predict_proba(X_scaled)[:, 1]
    
    # Prep output DF
    pred_df = df_clean.copy()
    pred_df.index = X_scaled.index
    pred_df["churn_score"] = np.round(proba, 4)
    
    # Calculate thresholds for segmentation
    monetary_p80 = df_clean["monthlycharges"].quantile(0.8)
    
    # Apply Segmentation
    pred_df["segment"] = pred_df.apply(lambda r: _get_segment(r, monetary_p80), axis=1)
    pred_df["recommendation"] = pred_df["segment"].apply(_get_recommendation)
    
    # Decision Matrix Risk Label (Simplified for the UI)
    def _risk_label(score):
        if score > 0.8: return "CRITICAL"
        if score > 0.5: return "AT RISK"
        return "SAFE"
    
    pred_df["risk_label"] = pred_df["churn_score"].apply(_risk_label)
    
    return pred_df

def simulate_next_month_churn(pred_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Predict Next-Month Churn using real risk simulation.
    final_risk_score = churn_score * (1 - activity_decay)
    """
    df = pred_df.copy()
    
    # Real risk logic
    df["final_risk_score"] = df["churn_score"] * (1 - df.get("activity_decay", 0.5))
    
    # Filters
    mask = (df["churn_score"] > 0.6) | (df.get("activity_decay", 1.0) < 0.2)
    
    at_risk = df[mask].sort_values("final_risk_score", ascending=False).head(top_n)
    return at_risk

def print_top_churn_customers(top5: pd.DataFrame):
    print(f"\n{_SEP}")
    print("  [!] Top 5 High-Risk Customers (Next Month Prediction)")
    print(_SEP)
    for i, (_, r) in enumerate(top5.iterrows(), 1):
        print(f"  #{i} {r['customer_id']} | Risk Score: {r['final_risk_score']:.3f} | Segment: {r['segment']}")
        print(f"      Rec: {r['recommendation']}")

def print_decision_matrix_summary(df: pd.DataFrame):
    summary = df["segment"].value_counts(normalize=True) * 100
    print(f"\n{_SEP}")
    print("  Segment Distribution")
    print(_SEP)
    for seg, val in summary.items():
        print(f"  {seg:<20}: {val:.1f}%")
