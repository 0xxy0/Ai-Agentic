"""
analysis/correlation.py  --  Insightful Correlation Analysis
=============================================================
Computes correlation matrix and highlights top churn drivers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_OUT_DIR = Path("outputs")
_OUT_DIR.mkdir(parents=True, exist_ok=True)
_SEP = "=" * 60

def run_correlation_analysis(df: pd.DataFrame):
    print(f"\n{_SEP}")
    print("  [Step 5] Insightful Correlation Analysis")
    print(_SEP)
    
    numeric_df = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if 'churn' not in numeric_df.columns:
        print("[Correlation] No churn label found. Skipping.")
        return pd.DataFrame()

    corr = numeric_df.corr()['churn'].drop('churn', errors='ignore').sort_values()
    
    # 1. Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#e63946' if x > 0 else '#2a9d8f' for x in corr]
    corr.plot(kind='barh', color=colors, ax=ax)
    ax.set_title("Feature Correlation with Churn", fontsize=14, fontweight='bold')
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    
    path = _OUT_DIR / "correlation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    
    # 2. Highlights
    top_pos = corr[corr > 0].tail(5)[::-1]
    top_neg = corr[corr < 0].head(5)

    print("\n  Top Positive Drivers (Increases Churn):")
    for feat, val in top_pos.items():
        print(f"    [+] {feat:<30} {val:+.3f}")
    
    print("\n  Top Negative Drivers (Decreases Churn):")
    for feat, val in top_neg.items():
        print(f"    [-] {feat:<30} {val:+.3f}")
        
    print(f"\n[Correlation] Plot saved -> {path}")
    return corr.to_frame()
