"""
models/train.py  --  Advanced Model Training & Explainability
=============================================================
1. XGBoost with scale_pos_weight & Early Stopping
2. 5-Fold Cross-Validation
3. SHAP Explainability (Global + Local)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

_OUT_DIR = Path("outputs")
_OUT_DIR.mkdir(parents=True, exist_ok=True)
_SEP = "=" * 60

warnings.filterwarnings("ignore")

def _calculate_shap(model, X: pd.DataFrame):
    """Generate SHAP values and plots."""
    print("\n  [SHAP] Calculating explainability...")
    explainer = shap.TreeExplainer(model)
    # Use a sample for SHAP if X is too large
    X_sample = X.sample(min(1000, len(X)), random_state=42)
    shap_values = explainer.shap_values(X_sample)

    # Global summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    path = _OUT_DIR / "09_shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SHAP] Global summary saved -> {path}")

    # Extract top features for MongoDB
    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['col_name', 'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    return feature_importance.head(20).to_dict(orient="records")

def train_models(X: pd.DataFrame, y: pd.Series, feature_names: list):
    """Train XGBoost with CV, early stopping and scale_pos_weight."""
    print(f"\n{_SEP}")
    print("  [Step 7] Advanced Model Training (XGBoost + CV)")
    print(_SEP)

    # 1. Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Cross-Validation
    print(f"\n  Running 5-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    scale_pos = (y == 0).sum() / max(y.sum(), 1)

    for i, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        m = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=scale_pos,
            random_state=42,
            n_jobs=-1,
            eval_metric="auc",
            early_stopping_rounds=20
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        y_prob = m.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_prob)
        cv_scores.append(score)
        print(f"    Fold {i}: ROC-AUC = {score:.4f}")

    print(f"  [CV] Mean ROC-AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # 3. Final Model with Early Stopping
    print("\n  Training final XGBoost model...")
    best_xgb = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=7,
        scale_pos_weight=scale_pos,
        random_state=42,
        n_jobs=-1,
        eval_metric=["auc", "logloss"],
        early_stopping_rounds=50
    )
    best_xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # 4. Evaluation
    y_prob = best_xgb.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
    }

    print(f"\n  [Final XGBoost] Metrics")
    for k, v in metrics.items():
        print(f"    {k:<12}: {v}")

    # 5. SHAP Explainability
    shap_insights = _calculate_shap(best_xgb, X_test)

    # 6. Persist
    joblib.dump(best_xgb, _OUT_DIR / "xgb_model.pkl")
    print(f"\n[train] Model saved -> {_OUT_DIR}/xgb_model.pkl")

    return best_xgb, {"xgb": metrics}, shap_insights
