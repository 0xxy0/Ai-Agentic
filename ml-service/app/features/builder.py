"""
Feature engineering pipeline.

Supports two schemas:

1. **Telecom churn schema** (primary) — ``build_customer_feature_dict`` and
   ``build_customer_feature_dataframe`` accept ``CustomerRecord``-shaped dicts
   with all 32 columns.

2. **Legacy RFM schema** — ``build_single_feature_dict`` and
   ``build_feature_dataframe`` accept the old ``RawUserActivity``-shaped dicts.

All transformations are deterministic and stateless so they can run
identically during training and inference.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from app.core.logger import get_logger

logger = get_logger(__name__)

# ── Shared constants ──────────────────────────────────────────────────────────
_EPSILON           = 1e-9
_RECENCY_CAP       = 365.0
_MONETARY_CAP_FACTOR = 5
_CREDIT_MAX        = 850.0

# ── Telecom churn feature columns (must match tools.feature_tools) ────────────
from tools.feature_tools import CHURN_FEATURE_COLUMNS  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# TELECOM CHURN SCHEMA  (primary)
# ═══════════════════════════════════════════════════════════════════════════════

_CONTRACT_RISK = {"Month-to-month": 2, "One year": 1, "Two year": 0}
_EDU_ORDINAL   = {"High School": 0, "Bachelor": 1, "Graduate": 2, "Postgraduate": 3}


def build_customer_feature_dict(raw: dict[str, Any]) -> dict[str, float]:
    """
    Build a model-ready feature dict from a single ``CustomerRecord``-shaped
    dictionary.

    Mirrors the vectorised logic in ``tools.feature_tools.engineer_churn_features``
    but operates on a single row for real-time inference.

    Args:
        raw: Dictionary matching ``CustomerRecord`` field names.

    Returns:
        Flat ``dict[str, float]`` with all :data:`CHURN_FEATURE_COLUMNS`.

    Raises:
        ValueError: If required fields are missing.
    """
    required = {"customer_id", "tenure", "contract", "monthlycharges", "totalcharges"}
    missing  = required - set(raw.keys())
    if missing:
        raise ValueError(f"Missing required CustomerRecord fields: {sorted(missing)}")

    # ── Demographics ──────────────────────────────────────────────────────────
    age            = float(raw.get("age", 35))
    senior_citizen = int(raw.get("senior_citizen", 0))
    dependents     = float(raw.get("dependents", 0))
    gender_male    = int(str(raw.get("gender", "Male")).strip().lower() == "male")
    edu            = str(raw.get("education", "Bachelor")).strip()
    edu_ordinal    = _EDU_ORDINAL.get(edu, 1)
    edu_bachelor   = int(edu == "Bachelor")
    edu_graduate   = int(edu == "Graduate")
    edu_postgrad   = int(edu == "Postgraduate")
    mstat          = str(raw.get("marital_status", "Single")).strip()
    marital_married  = int(mstat == "Married")
    marital_divorced = int(mstat == "Divorced")

    # ── Financial ─────────────────────────────────────────────────────────────
    income  = max(float(raw.get("annual_income", 50_000)), 0.0)
    monthly = max(float(raw.get("monthlycharges", 65.0)), 0.0)
    total   = max(float(raw.get("totalcharges",   0.0)),  0.0)
    cscore  = float(raw.get("credit_score", 680.0))
    cscore  = max(300.0, min(cscore, _CREDIT_MAX))

    # ── Tenure & contract ─────────────────────────────────────────────────────
    tenure = max(float(raw.get("tenure", 1)), 0.0)
    contract = str(raw.get("contract", "Month-to-month")).strip()
    contract_one_year  = int(contract == "One year")
    contract_two_year  = int(contract == "Two year")
    contract_risk      = _CONTRACT_RISK.get(contract, 2)

    if   tenure <= 12: tenure_bucket = 0
    elif tenure <= 24: tenure_bucket = 1
    elif tenure <= 48: tenure_bucket = 2
    else:              tenure_bucket = 3

    # ── Payment ───────────────────────────────────────────────────────────────
    pm = str(raw.get("payment_method", "Electronic check"))
    payment_bank   = int("Bank transfer" in pm)
    payment_credit = int("Credit card"   in pm)
    payment_mailed = int("Mailed check"  in pm)
    paperless      = int(raw.get("paperless_billing", 0))

    # ── Services ──────────────────────────────────────────────────────────────
    num_svc  = int(raw.get("num_services",          0))
    has_ph   = int(raw.get("has_phone_service",     0))
    has_int  = int(raw.get("has_internet_service",  0))
    has_sec  = int(raw.get("has_online_security",   0))
    has_bk   = int(raw.get("has_online_backup",     0))
    has_dp   = int(raw.get("has_device_protection", 0))
    has_ts   = int(raw.get("has_tech_support",      0))
    has_stv  = int(raw.get("has_streaming_tv",      0))
    has_smov = int(raw.get("has_streaming_movies",  0))

    # ── Behavioural ───────────────────────────────────────────────────────────
    csat   = float(raw.get("customer_satisfaction",     3))
    ncomp  = float(raw.get("num_complaints",            0))
    nscall = float(raw.get("num_service_calls",         0))
    latep  = float(raw.get("late_payments",             0.0))
    gb     = float(raw.get("avg_monthly_gb",            0.0))
    days   = float(raw.get("days_since_last_interaction", 30.0))

    # ── Derived rates ─────────────────────────────────────────────────────────
    tenure_safe      = max(tenure, 1.0)
    num_svc_safe     = max(num_svc, 1)
    expected_total   = monthly * tenure + _EPSILON

    complaint_rate    = ncomp  / tenure_safe
    service_call_rate = nscall / tenure_safe
    late_payment_rate = latep  / tenure_safe
    charge_ratio      = min(total / expected_total, 5.0)
    cps               = monthly / num_svc_safe
    log_cps           = math.log1p(cps)
    tenure_charge_ratio = min(total / expected_total, 5.0)
    revenue_per_gb    = min(monthly / max(gb, 0.1), 1000.0)
    income_to_charge  = min(income / (monthly * 12 + _EPSILON), 100.0)

    # ── Normalised ────────────────────────────────────────────────────────────
    sat_norm     = (csat - 1) / 4.0
    rec_norm     = min(days / _RECENCY_CAP, 1.0)
    age_norm     = age / 100.0
    cs_norm      = cscore / _CREDIT_MAX

    # ── Composite scores ──────────────────────────────────────────────────────
    protection_score    = has_sec + has_bk + has_dp + has_ts
    entertainment_score = has_stv + has_smov
    bundle_density      = min(num_svc / 8.0, 1.0)
    digital_engagement  = paperless + has_int
    credit_risk         = 1.0 - cs_norm
    late_norm_cap       = min(late_payment_rate / 2.0, 1.0)   # cap at 2/month
    risk_composite      = max(0.0, min(1.0,
        0.30 * (1.0 - sat_norm)
        + 0.20 * (contract_risk / 2.0)
        + 0.20 * rec_norm
        + 0.15 * credit_risk
        + 0.15 * late_norm_cap
    ))

    features: dict[str, float] = {
        # Demographics
        "age":                    age,
        "senior_citizen":         float(senior_citizen),
        "dependents":             dependents,
        "gender_male":            float(gender_male),
        "education_ordinal":      float(edu_ordinal),
        "edu_bachelor":           float(edu_bachelor),
        "edu_graduate":           float(edu_graduate),
        "edu_postgrad":           float(edu_postgrad),
        "marital_married":        float(marital_married),
        "marital_divorced":       float(marital_divorced),
        # Financial
        "annual_income":          income,
        "log_annual_income":      math.log1p(income),
        "monthlycharges":         monthly,
        "log_monthly_charges":    math.log1p(monthly),
        "totalcharges":           total,
        "log_total_charges":      math.log1p(total),
        "credit_score":           cscore,
        "credit_score_normalized": cs_norm,
        # Tenure & contract
        "tenure":                 tenure,
        "tenure_bucket":          float(tenure_bucket),
        "contract_one_year":      float(contract_one_year),
        "contract_two_year":      float(contract_two_year),
        "contract_risk_score":    float(contract_risk),
        # Payment
        "payment_bank_transfer":  float(payment_bank),
        "payment_credit_card":    float(payment_credit),
        "payment_mailed_check":   float(payment_mailed),
        "paperless_billing":      float(paperless),
        # Services
        "num_services":           float(num_svc),
        "has_phone_service":      float(has_ph),
        "has_internet_service":   float(has_int),
        "has_online_security":    float(has_sec),
        "has_online_backup":      float(has_bk),
        "has_device_protection":  float(has_dp),
        "has_tech_support":       float(has_ts),
        "has_streaming_tv":       float(has_stv),
        "has_streaming_movies":   float(has_smov),
        # Behavioural
        "customer_satisfaction":  csat,
        "num_complaints":         ncomp,
        "num_service_calls":      nscall,
        "late_payments":          latep,
        "avg_monthly_gb":         gb,
        "days_since_last_interaction": days,
        # Derived rates
        "complaint_rate":         complaint_rate,
        "service_call_rate":      service_call_rate,
        "late_payment_rate":      late_payment_rate,
        "charge_ratio":           charge_ratio,
        "charge_per_service":     cps,
        "log_charge_per_service": log_cps,
        "tenure_charge_ratio":    tenure_charge_ratio,
        "revenue_per_gb":         revenue_per_gb,
        "income_to_charge_ratio": income_to_charge,
        # Normalised
        "satisfaction_normalized": sat_norm,
        "recency_normalized":      rec_norm,
        "age_normalized":          age_norm,
        # Composite
        "protection_score":       float(protection_score),
        "entertainment_score":    float(entertainment_score),
        "service_bundle_density": bundle_density,
        "digital_engagement":     float(digital_engagement),
        "risk_composite":         risk_composite,
    }

    logger.debug("Built customer features", extra={"customer_id": raw.get("customer_id")})
    return features


def build_customer_feature_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Vectorised feature engineering for a batch of ``CustomerRecord`` dicts.

    Args:
        records: List of CustomerRecord-shaped dicts.

    Returns:
        :class:`pandas.DataFrame` with columns matching
        :data:`CHURN_FEATURE_COLUMNS`.
    """
    rows = [build_customer_feature_dict(r) for r in records]
    df   = pd.DataFrame(rows)[CHURN_FEATURE_COLUMNS]
    logger.info(
        "Batch customer feature engineering complete",
        extra={"n_records": len(df), "features": len(CHURN_FEATURE_COLUMNS)},
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY RFM SCHEMA  (preserved for backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

# Legacy feature names (in the exact order expected by the legacy model)
FEATURE_COLUMNS: list[str] = [
    "txn_7d", "txn_30d", "txn_90d", "recency_days", "frequency", "monetary",
    "usage_decay", "txn_30d_90d_ratio", "log_monetary", "log_frequency",
    "account_age_days", "recency_normalized",
]


def build_single_feature_dict(raw: dict[str, Any]) -> dict[str, float]:
    """
    Build a model-ready feature dictionary from a single raw RFM activity record.

    Args:
        raw: Dictionary matching :class:`~app.features.validators.RawUserActivity`.

    Returns:
        A flat ``dict[str, float]`` with all engineered features.
    """
    required = {"txn_7d", "txn_30d", "txn_90d", "recency_days", "frequency", "monetary"}
    missing  = required - set(raw.keys())
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    txn_7d   = float(raw["txn_7d"])
    txn_30d  = float(raw["txn_30d"])
    txn_90d  = float(raw["txn_90d"])
    recency  = min(float(raw["recency_days"]), _RECENCY_CAP)
    freq     = max(float(raw.get("frequency", 0)), 0.0)
    monetary = max(float(raw.get("monetary", 0.0)), 0.0)
    acct_age = float(raw.get("account_age_days") or 0.0)

    usage_decay       = txn_7d / (txn_30d + _EPSILON)
    txn_30d_90d_ratio = txn_30d / (txn_90d + _EPSILON)
    log_monetary      = math.log1p(monetary)
    log_frequency     = math.log1p(freq)
    recency_normalized = recency / _RECENCY_CAP

    return {
        "txn_7d":             txn_7d,
        "txn_30d":            txn_30d,
        "txn_90d":            txn_90d,
        "recency_days":       recency,
        "frequency":          freq,
        "monetary":           monetary,
        "usage_decay":        usage_decay,
        "txn_30d_90d_ratio":  txn_30d_90d_ratio,
        "log_monetary":       log_monetary,
        "log_frequency":      log_frequency,
        "account_age_days":   acct_age,
        "recency_normalized": recency_normalized,
    }


def build_feature_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Vectorised feature engineering for a batch of RFM records.

    Args:
        records: List of raw activity dictionaries.

    Returns:
        :class:`pandas.DataFrame` with columns matching :data:`FEATURE_COLUMNS`.
    """
    df = pd.DataFrame(records)

    numeric_cols = ["txn_7d", "txn_30d", "txn_90d", "frequency", "monetary", "account_age_days"]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "recency_days" not in df.columns:
        df["recency_days"] = 0
    df["recency_days"] = pd.to_numeric(
        df["recency_days"], errors="coerce"
    ).fillna(0).clip(upper=_RECENCY_CAP)

    if len(df) >= 4:
        q1 = df["monetary"].quantile(0.25)
        q3 = df["monetary"].quantile(0.75)
        iqr = q3 - q1
        upper_cap = q3 + _MONETARY_CAP_FACTOR * iqr
        df["monetary"] = df["monetary"].clip(upper=upper_cap)

    df["usage_decay"]        = df["txn_7d"] / (df["txn_30d"] + _EPSILON)
    df["txn_30d_90d_ratio"]  = df["txn_30d"] / (df["txn_90d"] + _EPSILON)
    df["log_monetary"]       = np.log1p(df["monetary"])
    df["log_frequency"]      = np.log1p(df["frequency"])
    df["recency_normalized"] = df["recency_days"] / _RECENCY_CAP

    logger.info(
        "Legacy batch feature engineering complete",
        extra={"n_records": len(df), "columns": list(df.columns)},
    )
    return df[FEATURE_COLUMNS]

