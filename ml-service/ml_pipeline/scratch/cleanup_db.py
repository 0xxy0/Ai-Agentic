"""
scratch/cleanup_db.py  --  Clear previous predictions for fresh start
"""
import os
import sys
from pathlib import Path
from pymongo import MongoClient

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def cleanup():
    uri = "mongodb://localhost:27017"
    db_name = "churn_db"
    
    client = MongoClient(uri)
    db = client[db_name]
    
    print(f"Cleaning up database: {db_name}...")
    
    # Drop collections
    db.predictions.drop()
    db.segments.drop()
    db.analytics_summary.drop()
    db.feature_insights.drop()
    
    print("Dropped: predictions, segments, analytics_summary, feature_insights")
    
    # Optional: Clear customers if you want a TOTAL fresh start
    # db.customers.drop()
    
    print("Database cleanup complete.")

if __name__ == "__main__":
    cleanup()
