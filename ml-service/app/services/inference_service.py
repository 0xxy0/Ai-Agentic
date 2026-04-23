import os
import pickle
import logging
import pandas as pd
import numpy as np
from typing import Any, Optional
import warnings
import joblib

# Suppress sklearn version mismatch warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

from ml_pipeline.preprocessing.clean import clean
from ml_pipeline.preprocessing.feature_engineering import engineer_features, encode_and_scale
from ml_pipeline.preprocessing.validator import validate_input, align_features, SchemaValidationError
from ml_pipeline.preprocessing.feature_mapper import map_business_to_ml_features
from ml_pipeline.analysis.explainability import ExplainabilityEngine
from ml_pipeline.analysis.decision_engine import DecisionEngine

logger = logging.getLogger(__name__)

class InferenceService:
    def __init__(self, model_path: str, scaler_path: str):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.feature_names: list[str] = []
        self.explainer: Optional[ExplainabilityEngine] = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Load pickled model and scaler from disk."""
        try:
            if os.path.exists(self.model_path):
                try:
                    self.model = joblib.load(self.model_path)
                except Exception:
                    with open(self.model_path, 'rb') as f:
                        self.model = pickle.load(f)
                logger.info(f"Loaded model from {self.model_path}")
                if hasattr(self.model, 'get_booster'):
                    self.feature_names = self.model.get_booster().feature_names
                
                # Initialize SHAP Explainer
                self.explainer = ExplainabilityEngine(self.model)
            
            if os.path.exists(self.scaler_path):
                try:
                    self.scaler = joblib.load(self.scaler_path)
                except Exception:
                    with open(self.scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                logger.info(f"Loaded scaler from {self.scaler_path}")
        except Exception as e:
            logger.warning(f"ML artifacts unavailable, using fallback responses: {e}")

    def predict_one(self, raw: dict[str, Any]) -> dict[str, Any]:
        """
        Enterprise Churn Intelligence + Decision Pipeline.
        """
        customer_id = raw.get("customer_id", raw.get("user_id", "unknown"))
        logger.info(f"Intelligence processing for {customer_id}")

        try:
            # 1. Feature Abstraction (UI -> Technical + History)
            mapped_data = map_business_to_ml_features(raw)
            
            # 2. Pipeline Preparation
            df = pd.DataFrame([mapped_data])
            # Ensure no ID duplicates
            if "user_id" in df.columns and "customer_id" in df.columns:
                df = df.drop(columns=["user_id"])
            elif "user_id" in df.columns:
                df = df.rename(columns={"user_id": "customer_id"})

            # 3. Validation & Cleaning
            validate_input(df)
            df_clean = clean(df)

            if not self.model or not self.scaler:
                return self._generate_fallback_response(raw, "Intelligence Engine Offline")

            # 4. Core ML Pipeline
            df_feat = engineer_features(df_clean)
            X, _, _, _ = encode_and_scale(df_feat, fit_scaler=False, scaler=self.scaler)

            # Alignment
            if self.feature_names:
                X = align_features(X, self.feature_names)
            
            # 5. ML Inference
            proba = float(self.model.predict_proba(X)[:, 1][0])
            
            # 6. Explainability Engine (SHAP)
            top_factors = []
            if self.explainer:
                top_factors = self.explainer.explain_prediction(X)

            # 7. Decision Engine (Strategic Rules)
            base_segment = "Enterprise" if mapped_data.get("monthlycharges", 0) > 100 else "SMB"
            decision = DecisionEngine.evaluate(proba, base_segment, mapped_data)

            return {
                "churn_score": round(proba, 4),
                "risk_level": decision["risk_level"],
                "segment": decision["segment"],
                "insight": decision["insight"],
                "top_factors": top_factors,
                "recommended_action": decision["recommended_action"],
                "confidence": decision["confidence"],
                "customer_id": customer_id,
                "name": raw.get("name"),
                "source": "Churn-Decision-System-v2.5",
                "status": "success"
            }

        except SchemaValidationError as e:
            return self._generate_fallback_response(raw, f"Validation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Critical System Error for {customer_id}: {e}", exc_info=True)
            return self._generate_fallback_response(raw, str(e))

    def _generate_fallback_response(self, raw: dict[str, Any], reason: str) -> dict[str, Any]:
        """Returns a safe, historical-based response when ML fails."""
        return {
            "churn_score": 0.2,
            "risk_level": "LOW",
            "segment": raw.get("segment", "SMB"),
            "insight": f"System in fallback mode. Reason: {reason}",
            "top_factors": [],
            "recommended_action": "Manual review required.",
            "confidence": 0.5,
            "customer_id": raw.get("customer_id", "unknown"),
            "source": "Decision-System-Fallback",
            "status": "degraded"
        }

    def predict_many(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Best-effort batch scoring that never raises for individual records."""
        return [self.predict_one(record) for record in records]

# Singleton instance
inference_service = InferenceService(
    model_path="ml_pipeline/outputs/xgb_model.pkl",
    scaler_path="ml_pipeline/outputs/scaler.pkl"
)
