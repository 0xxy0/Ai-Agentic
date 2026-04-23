"""
analysis/explainability.py  —  AI Factor Analysis (SHAP)
======================================================
Explains individual churn predictions by identifying top risk drivers.
"""

import numpy as np
import pandas as pd
import shap
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ExplainabilityEngine:
    def __init__(self, model: Any):
        self.model = model
        # For Tree models (XGBoost), TreeExplainer is most efficient
        try:
            self.explainer = shap.TreeExplainer(model)
        except Exception as e:
            logger.warning(f"TreeExplainer failed, falling back to KernelExplainer: {e}")
            self.explainer = None

    def explain_prediction(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Returns the top factors contributing to the churn score.
        """
        if self.explainer is None:
            return [{"feature": "System Fallback", "impact": "INFO", "reason": "Explainer offline"}]

        try:
            shap_values = self.explainer.shap_values(X)
            
            # For binary classification, shap_values might be a list or single array
            if isinstance(shap_values, list):
                # Use the positive class values
                vals = shap_values[1][0]
            else:
                vals = shap_values[0]
            
            # Pair features with their importance values
            feature_names = X.columns
            factors = []
            for name, val in zip(feature_names, vals):
                factors.append({
                    "feature": name,
                    "val": float(val),
                    "impact": "HIGH" if abs(val) > 0.5 else "MEDIUM" if abs(val) > 0.1 else "LOW"
                })
            
            # Sort by absolute impact and return top 3
            factors.sort(key=lambda x: abs(x["val"]), reverse=True)
            return factors[:5]

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return [{"feature": "Analysis Error", "impact": "ERROR", "reason": str(e)}]
