"""
Feature tools — stateless functions for time-series feature computation
and telecom-churn feature engineering.

Two feature sets are supported:

1. **Legacy RFM set** (``FEATURE_COLUMNS``) — 12 features derived from
   monthly activity time-series.
2. **Telecom churn set** (``CHURN_FEATURE_COLUMNS``) — ~59 engineered
   features for the 32-column customer dataset.

Each function is stateless so it can be safely imported from agents,
services, or test suites.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.core.logger import get_logger

logger = get_logger(__name__)

# ── Shared constants ──────────────────────────────────────────────────────────
_EPS            = 1e-9
_MAX_GAP_DAYS   = 365
_MONETARY_IQR_K = 5

# ── Telecom churn schema constants ────────────────────────────────────────────
_RECENCY_CAP      = 365.0   # days
_CREDIT_MAX       = 850.0
_INCOME_LOG_SCALE = True

# ── Contract risk ordinal mapping ─────────────────────────────────────────────
_CONTRACT_RISK: dict[str, int] = {
    "Month-to-month": 2,
    "One year":       1,
    "Two year":       0,
}

# ── Education ordinal mapping ─────────────────────────────────────────────────
_EDU_ORDINAL: dict[str, int] = {
    "High School":  0,
    "Bachelor":     1,
    "Graduate":     2,
    "Postgraduate": 3,
}


# ═══════════════════════════════════════════════════════════════════════════════
# TELECOM CHURN FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

# Full ordered feature column list (model input contract)
CHURN_FEATURE_COLUMNS: list[str] = [
    # --- Demographics ---
    "age",
    "senior_citizen",
    "dependents",
    "gender_male",
    "education_ordinal",
    "edu_bachelor",
    "edu_graduate",
    "edu_postgrad",
    "marital_married",
    "marital_divorced",
    # --- Financial ---
    "annual_income",
    "log_annual_income",
    "monthlycharges",
    "log_monthly_charges",
    "totalcharges",
    "log_total_charges",
    "credit_score",
    "credit_score_normalized",
    # --- Tenure & Contract ---
    "tenure",
    "tenure_bucket",
    "contract_one_year",
    "contract_two_year",
    "contract_risk_score",
    # --- Payment ---
    "payment_bank_transfer",
    "payment_credit_card",
    "payment_mailed_check",
    "paperless_billing",
    # --- Services ---
    "num_services",
    "has_phone_service",
    "has_internet_service",
    "has_online_security",
    "has_online_backup",
    "has_device_protection",
    "has_tech_support",
    "has_streaming_tv",
    "has_streaming_movies",
    # --- Behavioural ---
    "customer_satisfaction",
    "num_complaints",
    "num_service_calls",
    "late_payments",
    "avg_monthly_gb",
    "days_since_last_interaction",
    # --- Derived rates ---
    "complaint_rate",
    "service_call_rate",
    "late_payment_rate",
    "charge_ratio",
    "charge_per_service",
    "log_charge_per_service",
    "tenure_charge_ratio",
    "revenue_per_gb",
    "income_to_charge_ratio",
    # --- Normalised scalars ---
    "satisfaction_normalized",
    "recency_normalized",
    "age_normalized",
    # --- Composite scores ---
    "protection_score",
    "entertainment_score",
    "service_bundle_density",
    "digital_engagement",
    "risk_composite",
]


def engineer_churn_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a clean 32-column churn DataFrame into a model-ready feature
    matrix.

    The function is fully vectorised (no row-wise loops) and handles edge
    cases (divide-by-zero, missing values, unseen category values).

    Args:
        df: Cleaned customer DataFrame (output of ``impute_churn_data``).
            Must contain at minimum the columns defined in
            :data:`tools.data_tools.CHURN_SCHEMA_COLUMNS` excluding ``churn``.

    Returns:
        :class:`pandas.DataFrame` indexed by ``customer_id`` with columns
        matching :data:`CHURN_FEATURE_COLUMNS`.
    """
    f = pd.DataFrame(index=df.index)

    # ── 1. Demographics ───────────────────────────────────────────────────────
    f["age"]           = pd.to_numeric(df["age"], errors="coerce").fillna(35).clip(0, 120)
    f["senior_citizen"]= pd.to_numeric(df["senior_citizen"], errors="coerce").fillna(0)
    f["dependents"]    = pd.to_numeric(df["dependents"],    errors="coerce").fillna(0)
    f["gender_male"]   = (df["gender"].str.lower().str.strip() == "male").astype(int)

    edu = df["education"].str.strip()
    f["education_ordinal"] = edu.map(_EDU_ORDINAL).fillna(0).astype(int)
    f["edu_bachelor"]  = (edu == "Bachelor").astype(int)
    f["edu_graduate"]  = (edu == "Graduate").astype(int)
    f["edu_postgrad"]  = (edu == "Postgraduate").astype(int)

    mstat = df["marital_status"].str.strip()
    f["marital_married"]  = (mstat == "Married").astype(int)
    f["marital_divorced"] = (mstat == "Divorced").astype(int)

    # ── 2. Financial ─────────────────────────────────────────────────────────
    income = pd.to_numeric(df["annual_income"], errors="coerce").fillna(50_000).clip(0)
    f["annual_income"]     = income
    f["log_annual_income"] = np.log1p(income)

    monthly = pd.to_numeric(df["monthlycharges"], errors="coerce").fillna(65.0).clip(0)
    f["monthlycharges"]       = monthly
    f["log_monthly_charges"]  = np.log1p(monthly)

    total = pd.to_numeric(df["totalcharges"], errors="coerce").fillna(0).clip(0)
    f["totalcharges"]       = total
    f["log_total_charges"]  = np.log1p(total)

    cscore = pd.to_numeric(df["credit_score"], errors="coerce").fillna(680).clip(300, 850)
    f["credit_score"]            = cscore
    f["credit_score_normalized"] = cscore / _CREDIT_MAX

    # ── 3. Tenure & Contract ─────────────────────────────────────────────────
    tenure = pd.to_numeric(df["tenure"], errors="coerce").fillna(1).clip(0).astype(float)
    f["tenure"] = tenure
    f["tenure_bucket"] = pd.cut(
        tenure,
        bins=[-1, 12, 24, 48, float("inf")],
        labels=[0, 1, 2, 3],
    ).astype(int)

    contract = df["contract"].str.strip()
    f["contract_one_year"]   = (contract == "One year").astype(int)
    f["contract_two_year"]   = (contract == "Two year").astype(int)
    f["contract_risk_score"] = contract.map(_CONTRACT_RISK).fillna(2).astype(int)

    # ── 4. Payment ────────────────────────────────────────────────────────────
    pm = df["payment_method"].str.strip()
    f["payment_bank_transfer"] = pm.str.contains("Bank transfer",  case=False, na=False).astype(int)
    f["payment_credit_card"]   = pm.str.contains("Credit card",    case=False, na=False).astype(int)
    f["payment_mailed_check"]  = pm.str.contains("Mailed check",   case=False, na=False).astype(int)
    f["paperless_billing"]     = pd.to_numeric(df["paperless_billing"], errors="coerce").fillna(0).astype(int)

    # ── 5. Services ───────────────────────────────────────────────────────────
    svc_cols = [
        "num_services", "has_phone_service", "has_internet_service",
        "has_online_security", "has_online_backup", "has_device_protection",
        "has_tech_support", "has_streaming_tv", "has_streaming_movies",
    ]
    for col in svc_cols:
        f[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype(int)

    # ── 6. Behavioural ───────────────────────────────────────────────────────
    f["customer_satisfaction"]     = pd.to_numeric(df["customer_satisfaction"], errors="coerce").fillna(3).clip(1, 5)
    f["num_complaints"]            = pd.to_numeric(df["num_complaints"],        errors="coerce").fillna(0).clip(0)
    f["num_service_calls"]         = pd.to_numeric(df["num_service_calls"],     errors="coerce").fillna(0).clip(0)
    f["late_payments"]             = pd.to_numeric(df["late_payments"],         errors="coerce").fillna(0).clip(0)
    f["avg_monthly_gb"]            = pd.to_numeric(df["avg_monthly_gb"],        errors="coerce").fillna(0).clip(0)
    f["days_since_last_interaction"] = pd.to_numeric(
        df["days_since_last_interaction"], errors="coerce"
    ).fillna(30).clip(0)

    # ── 7. Derived rates (tenure-normalised) ─────────────────────────────────
    tenure_safe = tenure.clip(lower=1)   # avoid division by zero for new customers
    f["complaint_rate"]    = (f["num_complaints"]    / tenure_safe).round(6)
    f["service_call_rate"] = (f["num_service_calls"] / tenure_safe).round(6)
    f["late_payment_rate"] = (f["late_payments"]     / tenure_safe).round(6)

    # Charge ratio: how much of the expected lifetime has been paid
    #   charge_ratio → 1 = normal; >1 = overpaid; <1 = underpaid / new
    expected = monthly * tenure + _EPS
    f["charge_ratio"]  = (total / expected).clip(0, 5).round(6)

    # Charge per active service
    num_svc_safe = f["num_services"].clip(lower=1)
    cps = monthly / num_svc_safe
    f["charge_per_service"]     = cps.round(4)
    f["log_charge_per_service"] = np.log1p(cps)

    # Deviation of actual total from expected total (payment consistency)
    f["tenure_charge_ratio"] = (total / expected).clip(0, 5).round(6)

    # Revenue per GB of internet usage
    gb_safe = f["avg_monthly_gb"].clip(lower=0.1)
    f["revenue_per_gb"] = (monthly / gb_safe).clip(0, 1000).round(4)

    # Income-to-charge ratio: affordability proxy
    annual_charge = monthly * 12
    f["income_to_charge_ratio"] = (income / (annual_charge + _EPS)).clip(0, 100).round(4)

    # ── 8. Normalised scalars ─────────────────────────────────────────────────
    f["satisfaction_normalized"] = ((f["customer_satisfaction"] - 1) / 4.0).round(6)
    f["recency_normalized"]      = (f["days_since_last_interaction"] / _RECENCY_CAP).clip(0, 1).round(6)
    f["age_normalized"]          = (f["age"] / 100.0).round(6)

    # ── 9. Composite scores ───────────────────────────────────────────────────
    f["protection_score"]    = (
        f["has_online_security"] + f["has_online_backup"]
        + f["has_device_protection"] + f["has_tech_support"]
    )
    f["entertainment_score"] = f["has_streaming_tv"] + f["has_streaming_movies"]
    f["service_bundle_density"] = (f["num_services"] / 8.0).clip(0, 1).round(4)
    f["digital_engagement"]  = (f["paperless_billing"] + f["has_internet_service"]).clip(0, 2)

    # Risk composite: weighted combination of strong churn signals (no target leakage)
    credit_norm = (1.0 - f["credit_score_normalized"]).clip(0, 1)
    late_norm   = (f["late_payment_rate"] / f["late_payment_rate"].clip(lower=1).quantile(0.95 + _EPS)).clip(0, 1)
    f["risk_composite"] = (
        0.30 * (1.0 - f["satisfaction_normalized"])
        + 0.20 * (f["contract_risk_score"] / 2.0)
        + 0.20 * f["recency_normalized"]
        + 0.15 * credit_norm
        + 0.15 * late_norm
    ).clip(0, 1).round(6)

    # ── 10. IQR-based outlier capping on monetary features ────────────────────
    cap_cols = ["monthlycharges", "totalcharges", "annual_income",
                "charge_per_service", "revenue_per_gb"]
    if len(f) >= 4:
        for col in cap_cols:
            if col in f.columns:
                q1 = f[col].quantile(0.25)
                q3 = f[col].quantile(0.75)
                cap = q3 + _MONETARY_IQR_K * (q3 - q1)
                f[col] = f[col].clip(upper=cap)
                if col == "log_monthly_charges":
                    f["log_monthly_charges"] = np.log1p(f["monthlycharges"])

    # Re-compute log transforms after capping
    f["log_monthly_charges"] = np.log1p(f["monthlycharges"])
    f["log_total_charges"]   = np.log1p(f["totalcharges"])
    f["log_annual_income"]   = np.log1p(f["annual_income"])

    # Set customer_id as index
    if "customer_id" in df.columns:
        f.index = df["customer_id"].values

    logger.info(
        "Churn feature matrix built",
        extra={"rows": len(f), "features": len(CHURN_FEATURE_COLUMNS)},
    )
    return f[CHURN_FEATURE_COLUMNS]


def compute_mutual_information_preview(
    feature_df: pd.DataFrame,
    y: pd.Series,
    top_n: int = 15,
) -> dict[str, float]:
    """
    Compute a quick mutual-information preview between features and the
    churn target to provide an early signal of feature relevance.

    Uses sklearn's ``mutual_info_classif`` with default parameters.

    Args:
        feature_df: Feature matrix (columns = CHURN_FEATURE_COLUMNS).
        y:          Binary target Series aligned to ``feature_df``.
        top_n:      Number of top features to return.

    Returns:
        Dict mapping feature name → mutual information score, sorted
        descending, limited to ``top_n`` entries.
    """
    try:
        from sklearn.feature_selection import mutual_info_classif
        X = feature_df.fillna(0).values
        mi = mutual_info_classif(X, y.values, random_state=42, n_jobs=-1)
        scores = dict(zip(feature_df.columns, mi.tolist()))
        top = dict(
            sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        )
        logger.info("Mutual information preview computed", extra={"top_feature": list(top.keys())[:3]})
        return top
    except Exception as exc:
        logger.warning("Mutual information preview failed", extra={"error": str(exc)})
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY RFM / TIME-SERIES FEATURE ENGINEERING (preserved for backward compat)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_txn_trend(series: pd.Series) -> float:
    """
    Compute the linear trend (slope) of transaction counts over time.

    Positive slope → growing engagement.
    Negative slope → declining engagement (churn signal).

    Args:
        series: Ordered :class:`pandas.Series` of ``txn_count`` values.

    Returns:
        OLS slope as a float (0.0 if fewer than 2 data points).
    """
    n = len(series)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    slope, *_ = stats.linregress(x, series.values.astype(float))
    return float(slope)


def compute_spend_trend(series: pd.Series) -> float:
    """
    Compute the linear trend (slope) of spend over time.

    Args:
        series: Ordered :class:`pandas.Series` of ``spend`` values.

    Returns:
        OLS slope (0.0 if fewer than 2 points).
    """
    return compute_txn_trend(series)   # same logic, reuse


def compute_activity_gap(periods: pd.Series, reference_period: pd.Period) -> int:
    """
    Compute the number of months since the user's last recorded activity.

    Args:
        periods: Ordered :class:`pandas.Series` of :class:`pandas.Period` values.
        reference_period: The "current" period used as reference (e.g. max period in dataset).

    Returns:
        Integer gap in months (capped at 24 to limit outlier influence).
    """
    if periods.empty:
        return 24
    last_active = periods.max()
    gap = (reference_period - last_active).n
    return min(max(int(gap), 0), 24)


def rolling_mean(series: pd.Series, window: int) -> float:
    """
    Return the mean of the last ``window`` values in ``series``.

    Args:
        series: Ordered numeric Series.
        window: Look-back window size.

    Returns:
        Float mean; 0.0 if series is empty.
    """
    tail = series.tail(window)
    if tail.empty:
        return 0.0
    return float(tail.mean())


# ── Batch feature builder ─────────────────────────────────────────────────────




def build_feature_dataframe(
    ts_df: pd.DataFrame,
    reference_period: pd.Period | None = None,
) -> pd.DataFrame:
    """
    Build a flat feature matrix from a full time-series DataFrame.

    Groups by ``user_id``, computes :func:`build_user_features` per user,
    then applies outlier capping on ``total_spend``.

    Args:
        ts_df: Normalised time-series DataFrame (output of ``data_tools.to_time_series``).
        reference_period: Override the reference period; defaults to ``ts_df["period"].max()``.

    Returns:
        Feature :class:`pandas.DataFrame` with one row per user.
    """
    if reference_period is None:
        reference_period = ts_df["period"].max()

    records: list[dict[str, Any]] = []
    for user_id, grp in ts_df.groupby("user_id"):
        grp = grp.sort_values("period")
        feat = build_user_features(grp, reference_period)
        feat["user_id"] = user_id
        records.append(feat)

    feat_df = pd.DataFrame(records)

    # IQR-based outlier capping on spend
    if len(feat_df) >= 4:
        for col in ("total_spend", "spend_last_month", "spend_3_month_avg", "spend_6_month_avg"):
            if col in feat_df.columns:
                q1, q3 = feat_df[col].quantile([0.25, 0.75])
                cap = q3 + _MONETARY_IQR_K * (q3 - q1)
                feat_df[col] = feat_df[col].clip(upper=cap)

    logger.info(
        "Feature matrix built",
        extra={"users": len(feat_df), "features": list(feat_df.columns)},
    )
    return feat_df.set_index("user_id")


# ── Feature column registry (model input contract) ───────────────────────────

FEATURE_COLUMNS: list[str] = [
    "txn_7d",
    "txn_30d",
    "txn_90d",
    "recency_days",
    "frequency",
    "monetary",
    "usage_decay",
    "txn_30d_90d_ratio",
    "log_monetary",
    "log_frequency",
    "account_age_days",
    "recency_normalized",
]


def build_user_features(
    user_df: pd.DataFrame,
    reference_period: pd.Period,
) -> dict[str, Any]:
    """
    Compute 12 summary features for a user from their time-series history.
    """
    txn = user_df["txn_count"]
    spend = user_df["spend"]
    periods = user_df["period"]

    # Current month (last period in user data)
    txn_30d = float(txn.iloc[-1]) if not txn.empty else 0.0
    monetary = float(spend.sum())
    freq = float(txn.sum())
    
    # 7d is approximated as 1/4 of the current month in this synthetic setup
    txn_7d = txn_30d * 0.25 
    
    # 90d is last 3 months
    txn_90d = float(txn.tail(3).sum())
    
    recency = compute_activity_gap(periods, reference_period) * 30 # convert to approx days
    
    usage_decay = txn_7d / (txn_30d + _EPS)
    ratio_30_90 = txn_30d / (txn_90d + _EPS)
    
    # account age (diff between first and reference period)
    first_active = periods.min()
    age_days = (reference_period - first_active).n * 30

    return {
        "txn_7d":             txn_7d,
        "txn_30d":            txn_30d,
        "txn_90d":            txn_90d,
        "recency_days":       float(recency),
        "frequency":          freq,
        "monetary":           monetary,
        "usage_decay":        usage_decay,
        "txn_30d_90d_ratio":  ratio_30_90,
        "log_monetary":       math.log1p(monetary),
        "log_frequency":      math.log1p(freq),
        "account_age_days":   float(age_days),
        "recency_normalized": recency / 365.0,
    }
