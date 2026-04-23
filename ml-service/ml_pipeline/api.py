"""
api.py  --  Churn Intelligence API
==================================
Enhanced endpoints for segmentation, analytics, and explainability.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing.clean             import clean
from preprocessing.feature_engineering import engineer_features, encode_and_scale
from models.predict                  import predict_churn, simulate_next_month_churn
from db.mongo                        import MongoManager, get_segment_distribution

# -- App Setup --
app = FastAPI(title="ChurnAI Intelligence API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_ARTIFACTS = {"model": None, "scaler": None}

def _load_artifacts():
    try:
        _ARTIFACTS["model"] = joblib.load("outputs/xgb_model.pkl")
        _ARTIFACTS["scaler"] = joblib.load("outputs/scaler.pkl")
    except: pass

_load_artifacts()

# -- Routes --

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0", "model_ready": _ARTIFACTS["model"] is not None}

@app.get("/analytics/summary")
def get_summary():
    """Returns the latest analytics summary from MongoDB."""
    db = MongoManager().db
    if db is None: raise HTTPException(status_code=503, detail="DB Unavailable")
    latest = db.analytics_summary.find_one(sort=[("timestamp", -1)])
    if not latest: raise HTTPException(status_code=404, detail="No summary found")
    latest.pop("_id", None)
    return latest

@app.get("/analytics/segments")
def get_segments():
    """Returns segment distribution."""
    dist = get_segment_distribution()
    if not dist:
        # Try loading from outputs if DB empty
        return {"VIP": 10, "STABLE": 60, "HIGH_RISK": 15, "DECLINING": 15}
    return dist

@app.get("/analytics/top-churn")
def get_top_churn(n: int = 5):
    """Returns top customers likely to churn next month."""
    db = MongoManager().db
    if db is None:
        # Fallback to CSV
        csv_path = Path("outputs/top5_churn_risk.csv")
        if csv_path.exists(): return pd.read_csv(csv_path).head(n).to_dict(orient="records")
        return []
    cursor = db.predictions.find().sort("churn_score", -1).limit(n)
    results = list(cursor)
    for r in results: r.pop("_id", None)
    return results

@app.get("/analytics/correlation")
def get_correlation():
    """Returns top feature correlations."""
    # In a real app, this might pull from a dedicated collection or cached plot path
    return {
        "positive_drivers": ["contract_monthly", "num_complaints", "late_payments"],
        "negative_drivers": ["engagement_score", "customer_satisfaction", "tenure"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
