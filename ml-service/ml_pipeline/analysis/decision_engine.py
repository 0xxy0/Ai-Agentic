"""
analysis/decision_engine.py  —  Strategic Decision System
========================================================
Combines ML scores with business rules to recommend specific retention actions.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DecisionEngine:
    @staticmethod
    def evaluate(churn_score: float, segment: str, mapped_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applies business heuristics to ML output to generate a final decision.
        """
        # Determine Risk Level
        if churn_score >= 0.8:
            risk_level = "CRITICAL"
        elif churn_score >= 0.6:
            risk_level = "HIGH"
        elif churn_score >= 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Strategic Segmentation override
        # Even if churn is medium, if they are high spend, they might be 'CRITICAL' for business
        monetary = mapped_data.get("monthlycharges", 0)
        is_high_value = monetary > 120.0
        
        final_segment = segment
        if is_high_value and risk_level in ["HIGH", "CRITICAL"]:
            final_segment = "VIP-AT-RISK"

        # Actionable Recommendations
        recommendation = "Standard monitoring."
        confidence = 0.95 # Base confidence for ML logic

        if risk_level == "CRITICAL":
            if final_segment == "VIP-AT-RISK":
                recommendation = "IMMEDIATE: Personal Account Manager outreach & Executive call."
            else:
                recommendation = "Urgent retention offer: Provide 30% discount for 6 months."
        elif risk_level == "HIGH":
            recommendation = "Schedule proactive feedback session and offer loyalty points."
        elif risk_level == "MEDIUM":
            if mapped_data.get("activity_decay", 1.0) < 0.5:
                 recommendation = "Declining activity detected. Send re-engagement feature highlights."
                 confidence = 0.82
            else:
                recommendation = "Send personalized 'We miss you' email with survey."

        # Insight extraction
        insight = f"User shows {risk_level.lower()} churn risk."
        days_inactive = mapped_data.get("days_since_last_interaction", 0)
        if days_inactive > 30:
            insight += f" Alert: Inactive for {days_inactive} days."

        return {
            "risk_level": risk_level,
            "segment": final_segment,
            "recommended_action": recommendation,
            "insight": insight,
            "confidence": round(confidence, 2)
        }
