"""
preprocessing/feature_mapper.py  —  Enterprise Feature Abstraction Layer
======================================================================
Derives complex ML features from raw business inputs and historical MongoDB data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any

from ml_pipeline.db.mongo import get_user_history

logger = logging.getLogger(__name__)

# Strategic Business Mappings
PLAN_PRICING = {
    "Basic": 20.0,
    "Professional": 70.0,
    "Enterprise": 150.0,
    "Premium": 120.0
}

SEGMENT_BASE_SATISFACTION = {
    "Enterprise": 5,
    "Mid-Market": 4,
    "SMB": 3,
    "Startup": 4
}

def map_business_to_ml_features(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforms UI data + Historical MongoDB state into production ML features.
    """
    data = raw_data.copy()
    customer_id = data.get("customer_id", data.get("user_id", "unknown"))
    
    # 1. Fetch History from Feature Store (MongoDB)
    history_df = get_user_history(customer_id)
    
    # 2. Derive Monetary (Plan-based + Historical weighted)
    plan = data.get("subscription_plan", data.get("plan", "Basic"))
    base_monthly = PLAN_PRICING.get(plan, 50.0)
    
    if not history_df.empty and "amount" in history_df.columns:
        # If we have real transaction data, use the last 3 months average
        recent_spend = history_df[history_df["timestamp"] > (datetime.now() - timedelta(days=90))]
        if not recent_spend.empty:
            data["monthlycharges"] = recent_spend["amount"].mean()
        else:
            data["monthlycharges"] = base_monthly
    else:
        data["monthlycharges"] = base_monthly

    # 3. Derive Tenure (Data-driven)
    if not history_df.empty:
        first_seen = history_df["timestamp"].min()
        tenure_months = (datetime.now() - first_seen).days // 30
        data["tenure"] = max(1, tenure_months)
    else:
        # Fallback to business heuristic if new user
        data["tenure"] = 12 if data.get("segment") == "Enterprise" else 6

    # 4. Calculate Total Charges
    data["totalcharges"] = data["monthlycharges"] * data["tenure"]

    # 5. Compute Recency & Frequency (RFM)
    if not history_df.empty:
        last_activity = history_df["timestamp"].max()
        data["days_since_last_interaction"] = (datetime.now() - last_activity).days
        
        # Frequency (txn count last 30d)
        last_30d = history_df[history_df["timestamp"] > (datetime.now() - timedelta(days=30))]
        data["frequency_30d"] = len(last_30d)
        
        # Activity Decay (7d vs 30d ratio)
        last_7d = history_df[history_df["timestamp"] > (datetime.now() - timedelta(days=7))]
        data["activity_decay"] = len(last_7d) / max(1, len(last_30d))
    else:
        # Fallback for cold-start users
        data["days_since_last_interaction"] = 30
        data["frequency_30d"] = 1
        data["activity_decay"] = 1.0

    # 6. Engagement & Sentiment
    segment = data.get("segment", "SMB")
    status = data.get("current_status", data.get("status", "Active"))
    
    base_sat = SEGMENT_BASE_SATISFACTION.get(segment, 3)
    # Deduct satisfaction if inactive
    if data["days_since_last_interaction"] > 45:
        base_sat = max(1, base_sat - 1.5)
        
    data["customer_satisfaction"] = base_sat
    data["num_complaints"] = 0 if status == "Active" else 2

    return data
