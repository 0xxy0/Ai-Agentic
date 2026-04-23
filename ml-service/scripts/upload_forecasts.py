import os
import sys
from pathlib import Path
import pandas as pd

# Add ml_pipeline to path to import MongoManager
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "ml_pipeline"))

from db.mongo import save_forecasts

def upload_forecast_data():
    csv_path = PROJECT_ROOT / "data" / "quarterly_forecast_raw_predictions.csv"
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print("Uploading to MongoDB...")
    save_forecasts(df)
    print("Upload complete.")

if __name__ == "__main__":
    upload_forecast_data()
