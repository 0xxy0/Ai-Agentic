"""
main.py  --  Churn Intelligence Pipeline (E2E)
==============================================
Orchestrates the advanced churn prediction system.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import pandas as pd

# -- Local Imports --
from preprocessing.clean               import clean
from preprocessing.feature_engineering import engineer_features, encode_and_scale
from analysis.eda                      import run_eda
from analysis.correlation              import run_correlation_analysis
from models.train                      import train_models
from models.predict                    import (
    predict_churn,
    simulate_next_month_churn,
    print_top_churn_customers,
    print_decision_matrix_summary,
)
from db.mongo import (
    insert_customers,
    save_predictions,
    save_feature_insights,
    save_analytics_summary,
    ping
)

_BANNER = """
+==============================================================+
|        ChurnAI v2.0 -- Churn Intelligence System             |
|     Advanced Features | XGBoost + CV | SHAP | MongoDB        |
+==============================================================+
"""

def main():
    print(_BANNER)
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=r"data\customer_churn_1M.csv")
    parser.add_argument("--sample", type=int, default=100000)
    parser.add_argument("--no-mongo", action="store_true")
    parser.add_argument("--clear-db", action="store_true", help="Clear previous predictions before run")
    args = parser.parse_args()

    # Step 0: Cleanup
    if args.clear_db and not args.no_mongo and ping():
        print("[cleanup] Dropping previous predictions and analytics data...")
        from db.mongo import MongoManager
        db = MongoManager().db
        db.predictions.drop()
        db.analytics_summary.drop()
        db.feature_insights.drop()
        print("[cleanup] Collections cleared.")

    # Step 1: Load
    df_raw = pd.read_csv(args.csv, nrows=args.sample if args.sample > 0 else None)
    print(f"[load] Loaded {len(df_raw):,} rows.")

    # Step 2: Mongo Ingest
    if not args.no_mongo: insert_customers(df_raw)

    # Step 3: Preprocess
    df_clean = clean(df_raw)
    df_feat  = engineer_features(df_clean)
    X, y, feature_names, scaler = encode_and_scale(df_feat)
    joblib.dump(scaler, "outputs/scaler.pkl")

    # Step 4: Correlation
    run_correlation_analysis(df_feat)

    # Step 5: Train
    model, metrics, shap_insights = train_models(X, y, feature_names)
    
    # Step 6: Predict & Segment
    pred_df = predict_churn(df_feat, X, model=model)
    print_decision_matrix_summary(pred_df)

    # Step 7: Future Churn
    top5 = simulate_next_month_churn(pred_df, top_n=5)
    print_top_churn_customers(top5)
    top5.to_csv("outputs/top5_churn_risk.csv", index=False)

    # Step 8: Persist Analytics to Mongo
    if not args.no_mongo and ping():
        save_predictions(pred_df)
        save_feature_insights(shap_insights)
        
        summary = {
            "total_customers": len(pred_df),
            "avg_churn_score": float(pred_df["churn_score"].mean()),
            "segments": pred_df["segment"].value_counts().to_dict(),
            "model_metrics": metrics["xgb"]
        }
        save_analytics_summary(summary)

    print("\n[DONE] System upgraded to Churn Intelligence + Decision System.")

if __name__ == "__main__":
    main()
