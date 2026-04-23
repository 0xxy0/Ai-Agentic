"""
ml_pipeline/fast_batch_predict.py  --  Optimized 1M Row Ingestion
==============================================================
Predicts and uploads the full 1,000,000 rows in chunks.
Skips training to save time.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from tqdm import tqdm

# Add local paths
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing.clean             import clean
from preprocessing.feature_engineering import engineer_features, encode_and_scale
from models.predict                  import predict_churn, _get_segment, _get_recommendation
from db.mongo                        import MongoManager, ping

def run_fast_upload(csv_path: str, chunk_size: int = 50000):
    print(f"🚀 Starting Full 1M Row Upload: {csv_path}")
    
    if not ping():
        print("❌ MongoDB not reachable. Exiting.")
        return

    # Load artifacts
    model = joblib.load("outputs/xgb_model.pkl")
    scaler = joblib.load("outputs/scaler.pkl")
    db = MongoManager().db
    
    # Pre-calculate monetary p80 (heuristic for large dataset)
    monetary_p80 = 80.0 

    total_uploaded = 0
    
    # Process in chunks
    reader = pd.read_csv(csv_path, chunksize=chunk_size)
    
    for i, chunk in enumerate(reader):
        print(f"  [Chunk {i+1}] Processing {len(chunk):,} rows...")
        
        # 1. Pipeline
        df_clean = clean(chunk)
        df_feat  = engineer_features(df_clean)
        X, _, _, _ = encode_and_scale(df_feat, fit_scaler=False, scaler=scaler)
        
        # 2. Predict
        probas = model.predict_proba(X)[:, 1]
        df_feat["churn_score"] = np.round(probas, 4)
        
        # 3. Segment & Recommendation
        df_feat["segment"] = df_feat.apply(lambda r: _get_segment(r, monetary_p80), axis=1)
        df_feat["recommendation"] = df_feat["segment"].apply(_get_recommendation)
        
        # 4. Bulk Mongo Upload
        records = df_feat.to_dict(orient="records")
        db.predictions.insert_many(records)
        
        total_uploaded += len(records)
        print(f"  [Chunk {i+1}] Done. Total: {total_uploaded:,}")

    print(f"\n✅ SUCCESS! Uploaded {total_uploaded:,} predictions to MongoDB (agentic.predictions).")

if __name__ == "__main__":
    csv = r"data\customer_churn_1M.csv"
    run_fast_upload(csv)
