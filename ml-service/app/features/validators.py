"""
Feature input validators.

Pydantic models ensure that raw data arriving at the API or batch
pipeline is structurally sound before feature engineering begins.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class RawUserActivity(BaseModel):
    """
    Raw per-user activity record submitted for churn scoring.

    All monetary/count fields must be non-negative.
    Recency is expressed in days (non-negative integer).
    """

    user_id: Optional[str] = Field(None, description="Unique user identifier")
    customer_id: Optional[str] = Field(None, description="Enterprise customer identifier")

    # Metadata / New Schema Fields
    name: Optional[str] = Field(None, description="Customer full name")
    segment: Optional[str] = Field(None, description="Market segment (Enterprise, SMB, etc.)")
    subscription_plan: Optional[str] = Field(None, description="Current subscription plan")
    current_status: Optional[str] = Field(None, description="Account status (Active, Churned, etc.)")
    predicted_churn_probability: Optional[str] = Field(None, description="Historical churn probability")
    risk_level: Optional[str] = Field(None, description="Historical risk level")
    predicted_revenue_loss: Optional[str] = Field(None, description="Estimated revenue impact")
    last_active_date: Optional[str] = Field(None, description="Last activity timestamp")
    forecast_month: Optional[str] = Field(None, description="Strategic forecast window")
    recommended_action: Optional[str] = Field(None, description="Suggested retention strategy")

    # Transaction counts per time window
    txn_7d: float = Field(0.0, ge=0, description="Transactions in last 7 days")
    txn_30d: float = Field(0.0, ge=0, description="Transactions in last 30 days")
    txn_90d: float = Field(0.0, ge=0, description="Transactions in last 90 days")

    # RFM components
    recency_days: int = Field(
        0, ge=0, le=3650, description="Days since last activity (0–3650)"
    )
    frequency: int = Field(0, ge=0, description="Total activity count")
    monetary: float = Field(0.0, ge=0, description="Total spend / lifetime value")

    @field_validator("txn_30d")
    @classmethod
    def txn_30d_gte_7d(cls, v: float, info) -> float:
        """30-day count should be >= 7-day count (business logic guard)."""
        txn_7d = info.data.get("txn_7d", 0.0)
        if v < txn_7d:
            raise ValueError("txn_30d must be >= txn_7d")
        return v

    @field_validator("txn_90d")
    @classmethod
    def txn_90d_gte_30d(cls, v: float, info) -> float:
        """90-day count should be >= 30-day count."""
        txn_30d = info.data.get("txn_30d", 0.0)
        if v < txn_30d:
            raise ValueError("txn_90d must be >= txn_30d")
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "user_id": "usr_abc123",
            "txn_7d": 3,
            "txn_30d": 10,
            "txn_90d": 28,
            "recency_days": 5,
            "frequency": 45,
            "monetary": 1250.00,
            "account_age_days": 365,
            "plan_tier": "pro",
        }
    }}


class BatchScoringRequest(BaseModel):
    """Request body for batch scoring endpoint (list of users)."""

    users: list[RawUserActivity] = Field(
        ..., min_length=1, description="List of user activity records"
    )
    churn_window_days: int = Field(
        30, description="Churn definition window: 30 | 60 | 90"
    )
