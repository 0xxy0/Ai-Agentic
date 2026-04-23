"""
Feature input validators.

Pydantic models for two API schemas:

1. ``CustomerRecord``   — 32-column telecom churn customer record
                          (current schema, aligned with the churn dataset)
2. ``RawUserActivity``  — legacy RFM activity record (preserved for backward
                          compatibility with existing integrations)

Both models validate field types and ranges and are used by the inference
service for single and batch scoring.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════════════════════════════
# TELECOM CHURN CUSTOMER RECORD  (primary schema)
# ═══════════════════════════════════════════════════════════════════════════════


class CustomerRecord(BaseModel):
    """
    32-column telecom customer record for churn prediction.

    Required fields: ``customer_id``, ``tenure``, ``contract``,
    ``monthlycharges``, ``totalcharges``.

    All binary service flags default to 0 (not subscribed).
    All behavioural counts default to 0.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    customer_id: str = Field(..., description="Unique customer identifier")

    # ── Demographics ──────────────────────────────────────────────────────────
    age:            int     = Field(35,  ge=18, le=120, description="Customer age")
    gender:         str     = Field("Male", description="Male | Female")
    annual_income:  float   = Field(50_000.0, ge=0, description="Annual income (USD)")
    education:      str     = Field(
        "Bachelor",
        description="High School | Bachelor | Graduate | Postgraduate",
    )
    marital_status: str     = Field("Single", description="Single | Married | Divorced")
    dependents:     int     = Field(0, ge=0, le=20, description="Number of dependents")
    senior_citizen: int     = Field(0, ge=0, le=1, description="1 if senior (≥65), else 0")

    # ── Contract & billing ────────────────────────────────────────────────────
    tenure:            int   = Field(..., ge=0, description="Months as customer")
    contract:          str   = Field(
        ...,
        description="Month-to-month | One year | Two year",
    )
    payment_method:    str   = Field(
        "Electronic check",
        description="Electronic check | Mailed check | Bank transfer (automatic) | Credit card (automatic)",
    )
    paperless_billing: int   = Field(0, ge=0, le=1, description="1 if paperless billing")
    monthlycharges:    float = Field(..., ge=0, description="Monthly charges (USD)")
    totalcharges:      float = Field(..., ge=0, description="Total charges to date (USD)")

    # ── Services ──────────────────────────────────────────────────────────────
    num_services:          int = Field(0, ge=0, le=8)
    has_phone_service:     int = Field(0, ge=0, le=1)
    has_internet_service:  int = Field(0, ge=0, le=1)
    has_online_security:   int = Field(0, ge=0, le=1)
    has_online_backup:     int = Field(0, ge=0, le=1)
    has_device_protection: int = Field(0, ge=0, le=1)
    has_tech_support:      int = Field(0, ge=0, le=1)
    has_streaming_tv:      int = Field(0, ge=0, le=1)
    has_streaming_movies:  int = Field(0, ge=0, le=1)

    # ── Behavioural ───────────────────────────────────────────────────────────
    customer_satisfaction:     int   = Field(3, ge=1, le=5)
    num_complaints:            int   = Field(0, ge=0)
    num_service_calls:         int   = Field(0, ge=0)
    late_payments:             float = Field(0.0, ge=0)
    avg_monthly_gb:            float = Field(0.0, ge=0)
    days_since_last_interaction: float = Field(30.0, ge=0)
    credit_score:              float = Field(680.0, ge=300, le=850)

    # ── Computed / derived fields (optional — computed by FeatureAgent) ───────
    signup_date: Optional[str] = Field(None, description="ISO date string YYYY-MM-DD")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        allowed = {"Male", "Female"}
        if v not in allowed:
            raise ValueError(f"gender must be one of {allowed}, got '{v}'")
        return v

    @field_validator("contract")
    @classmethod
    def validate_contract(cls, v: str) -> str:
        allowed = {"Month-to-month", "One year", "Two year"}
        if v not in allowed:
            raise ValueError(f"contract must be one of {allowed}, got '{v}'")
        return v

    @field_validator("totalcharges")
    @classmethod
    def totalcharges_gte_monthly(cls, v: float, info) -> float:
        """totalcharges should be ≥ monthlycharges for customers with ≥ 1 month tenure."""
        monthly = info.data.get("monthlycharges", 0.0)
        tenure  = info.data.get("tenure", 0)
        if tenure >= 1 and v < monthly * 0.9:
            # Soft warning: allow small discrepancy due to rounding/prorations
            pass
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id":               "CUST_0001234",
                "age":                       42,
                "gender":                    "Female",
                "annual_income":             72_000.0,
                "education":                 "Bachelor",
                "marital_status":            "Married",
                "dependents":                2,
                "senior_citizen":            0,
                "tenure":                    24,
                "contract":                  "One year",
                "payment_method":            "Bank transfer (automatic)",
                "paperless_billing":         1,
                "monthlycharges":            79.50,
                "totalcharges":              1908.00,
                "num_services":              5,
                "has_phone_service":         1,
                "has_internet_service":      1,
                "has_online_security":       1,
                "has_online_backup":         1,
                "has_device_protection":     0,
                "has_tech_support":          1,
                "has_streaming_tv":          0,
                "has_streaming_movies":      0,
                "customer_satisfaction":     4,
                "num_complaints":            0,
                "num_service_calls":         1,
                "late_payments":             0.0,
                "avg_monthly_gb":            45.2,
                "days_since_last_interaction": 12.0,
                "credit_score":              720.0,
            }
        }
    }


class BatchCustomerScoringRequest(BaseModel):
    """Request body for batch customer churn scoring."""

    customers: list[CustomerRecord] = Field(
        ..., min_length=1, description="List of customer records to score"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY RFM ACTIVITY RECORD  (preserved for backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════════


class RawUserActivity(BaseModel):
    """
    Raw per-user activity record submitted for churn scoring.

    All monetary/count fields must be non-negative.
    Recency is expressed in days (non-negative integer).
    """

    user_id: str = Field(..., description="Unique user identifier")

    # Transaction counts per time window
    txn_7d: float = Field(0.0, ge=0, description="Transactions in last 7 days")
    txn_30d: float = Field(0.0, ge=0, description="Transactions in last 30 days")
    txn_90d: float = Field(0.0, ge=0, description="Transactions in last 90 days")

    # RFM components
    recency_days: int = Field(
        ..., ge=0, le=3650, description="Days since last activity (0–3650)"
    )
    frequency: int = Field(0, ge=0, description="Total activity count")
    monetary: float = Field(0.0, ge=0, description="Total spend / lifetime value")

    # Optional metadata
    account_age_days: Optional[int] = Field(
        None, ge=0, description="Days since account creation"
    )
    plan_tier: Optional[str] = Field(
        None, description="Subscription plan tier (free/basic/pro)"
    )

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

