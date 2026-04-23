"""
db/mongo.py  --  MongoDB CRUD & Analytics Storage
================================================
Handles:
    * Customer ingestion
    * Predictions & Segments
    * Feature insights (SHAP)
    * Analytics summaries
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Load env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
_DB_NAME = os.getenv("MONGO_DB", "agentic")

class MongoManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            try:
                cls._instance.client = MongoClient(_URI, serverSelectionTimeoutMS=2000)
                cls._instance.db = cls._instance.client[_DB_NAME]
                # Ping
                cls._instance.client.admin.command("ping")
            except Exception:
                cls._instance.client = None
        return cls._instance

def ping() -> bool:
    mgr = MongoManager()
    if mgr.client is None: return False
    try:
        mgr.client.admin.command("ping")
        return True
    except:
        return False

def insert_customers(df: pd.DataFrame):
    if not ping(): return
    db = MongoManager().db
    records = df.to_dict(orient="records")
    operations = [
        UpdateOne({"customer_id": r["customer_id"]}, {"$set": r}, upsert=True)
        for r in records
    ]
    db.customers.bulk_write(operations)
    print(f"[mongo] Upserted {len(records):,} customers.")

def save_predictions(df: pd.DataFrame):
    """Saves churn_score, segment, and risk_label."""
    if not ping(): return
    db = MongoManager().db
    df = df.copy()
    df["timestamp"] = datetime.utcnow()
    
    records = df.to_dict(orient="records")
    # Store in 'predictions' and 'segments'
    db.predictions.insert_many(records)
    
    # Update segments in customers collection too
    ops = [
        UpdateOne({"customer_id": r["customer_id"]}, {"$set": {"segment": r["segment"], "last_score": r["churn_score"]}})
        for r in records
    ]
    db.customers.bulk_write(ops)
    print(f"[mongo] Saved {len(records):,} predictions & segments.")

def save_feature_insights(insights: list[dict]):
    """Saves SHAP global importance or local insights."""
    if not ping(): return
    db = MongoManager().db
    db.feature_insights.insert_many(insights)
    print(f"[mongo] Saved {len(insights)} feature insights.")

def save_analytics_summary(summary: dict):
    """Saves aggregated pipeline stats."""
    if not ping(): return
    db = MongoManager().db
    summary["timestamp"] = datetime.utcnow()
    db.analytics_summary.insert_one(summary)
    print("[mongo] Saved analytics summary.")

def save_forecasts(df: pd.DataFrame):
    """Saves quarterly forecast predictions."""
    if not ping(): return
    db = MongoManager().db
    df = df.copy()
    df["timestamp"] = datetime.utcnow()
    
    records = df.to_dict(orient="records")
    # Clean keys for MongoDB (spaces to underscores)
    cleaned_records = []
    for r in records:
        cleaned_r = {k.lower().replace(" ", "_"): v for k, v in r.items()}
        cleaned_records.append(cleaned_r)
    
    db.forecasts.insert_many(cleaned_records)
    print(f"[mongo] Saved {len(cleaned_records):,} forecast predictions.")

def load_forecasts(month: str = None) -> pd.DataFrame:
    if not ping(): return pd.DataFrame()
    db = MongoManager().db
    query = {"forecast_month": month} if month else {}
    cursor = db.forecasts.find(query).sort("timestamp", -1)
    return pd.DataFrame(list(cursor))

def load_top_churn_risk(n: int = 5) -> pd.DataFrame:
    if not ping(): return pd.DataFrame()
    db = MongoManager().db
    cursor = db.predictions.find().sort("churn_score", -1).limit(n)
    return pd.DataFrame(list(cursor))

def get_user_history(customer_id: str, days: int = 90) -> pd.DataFrame:
    """Fetches transaction and activity history for a specific user."""
    if not ping(): return pd.DataFrame()
    db = MongoManager().db
    # We look in 'transactions' or 'activity' collection
    # For now, let's assume 'activity' collection exists and has 'timestamp' and 'customer_id'
    cursor = db.activity.find({"customer_id": customer_id}).sort("timestamp", -1)
    return pd.DataFrame(list(cursor))

def get_segment_distribution() -> dict:
    if not ping(): return {}
    db = MongoManager().db
    pipeline = [{"$group": {"_id": "$segment", "count": {"$sum": 1}}}]
    results = list(db.predictions.aggregate(pipeline))
    return {r["_id"]: r["count"] for r in results}
