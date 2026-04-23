"""
Telecom Churn Dataset Generator — 32-column industry-grade synthetic data.

Generates realistic customer records with proper correlations between
features and the churn target, class imbalance (~25–30% churn rate), and
domain-accurate distributions for all field types.

Column schema (32 total)
------------------------
String  (6): customer_id, gender, education, marital_status, contract,
             payment_method
Integer (17): age, dependents, tenure, paperless_billing, senior_citizen,
              num_services, has_phone_service, has_internet_service,
              has_online_security, has_online_backup, has_device_protection,
              has_tech_support, has_streaming_tv, has_streaming_movies,
              customer_satisfaction, num_complaints, num_service_calls
Decimal (7): annual_income, monthlycharges, totalcharges, avg_monthly_gb,
             late_payments, days_since_last_interaction, credit_score
Other   (2): signup_date, churn

Usage:
    python data/generate_churn_dataset.py --n 10000 --output data/churn.csv
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sample_dates(n: int, start: str, end: str, rng: np.random.Generator) -> np.ndarray:
    start_d = date.fromisoformat(start)
    end_d   = date.fromisoformat(end)
    days    = (end_d - start_d).days
    offsets = rng.integers(0, days, n)
    return np.array([start_d + timedelta(days=int(o)) for o in offsets])


def generate(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic 32-column telecom churn dataset.

    Args:
        n:    Number of customer records.
        seed: Random seed for reproducibility.

    Returns:
        :class:`pandas.DataFrame` with exactly 32 columns.
    """
    rng = np.random.default_rng(seed)

    # ── Demographic base ──────────────────────────────────────────────────────
    age            = rng.integers(18, 80, n)
    senior_citizen = (age >= 65).astype(int)
    gender         = rng.choice(["Male", "Female"], n, p=[0.50, 0.50])
    gender_male    = (gender == "Male").astype(int)

    education = rng.choice(
        ["High School", "Bachelor", "Graduate", "Postgraduate"],
        n, p=[0.30, 0.40, 0.20, 0.10],
    )
    edu_num = np.select(
        [education == "High School", education == "Bachelor",
         education == "Graduate",   education == "Postgraduate"],
        [0, 1, 2, 3],
    )

    marital_status = rng.choice(
        ["Single", "Married", "Divorced"], n, p=[0.35, 0.50, 0.15]
    )
    dependents = rng.integers(0, 6, n)

    # ── Financial base ────────────────────────────────────────────────────────
    income_base  = rng.lognormal(10.8, 0.7, n).clip(15_000, 300_000).round(2)
    credit_score = (rng.normal(680, 85, n)).clip(300, 850).round(1)

    # ── Contract / tenure ─────────────────────────────────────────────────────
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        n, p=[0.55, 0.25, 0.20],
    )
    # Tenure shaped by contract type
    tenure = np.where(
        contract == "Month-to-month",
        rng.integers(1, 36, n),
        np.where(
            contract == "One year",
            rng.integers(12, 60, n),
            rng.integers(24, 72, n),
        ),
    )

    # ── Services ──────────────────────────────────────────────────────────────
    has_phone_service   = (rng.random(n) > 0.10).astype(int)
    has_internet_service = (rng.random(n) > 0.20).astype(int)
    has_online_security  = np.where(has_internet_service, (rng.random(n) > 0.55).astype(int), 0)
    has_online_backup    = np.where(has_internet_service, (rng.random(n) > 0.55).astype(int), 0)
    has_device_protection = np.where(has_internet_service, (rng.random(n) > 0.55).astype(int), 0)
    has_tech_support     = np.where(has_internet_service, (rng.random(n) > 0.55).astype(int), 0)
    has_streaming_tv     = np.where(has_internet_service, (rng.random(n) > 0.45).astype(int), 0)
    has_streaming_movies = np.where(has_internet_service, (rng.random(n) > 0.45).astype(int), 0)

    num_services = (
        has_phone_service + has_internet_service + has_online_security
        + has_online_backup + has_device_protection + has_tech_support
        + has_streaming_tv + has_streaming_movies
    )

    # ── Billing ───────────────────────────────────────────────────────────────
    monthly_base = (
        20.0
        + has_phone_service * rng.uniform(15, 25, n)
        + has_internet_service * rng.uniform(20, 40, n)
        + has_online_security * 5
        + has_online_backup * 5
        + has_device_protection * 5
        + has_tech_support * 5
        + has_streaming_tv * 8
        + has_streaming_movies * 8
    )
    noise         = rng.normal(0, 3, n)
    monthlycharges = (monthly_base + noise).clip(18.0, 200.0).round(2)
    totalcharges   = (monthlycharges * tenure * rng.uniform(0.85, 1.05, n)).round(2)

    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)",
         "Credit card (automatic)"],
        n, p=[0.34, 0.22, 0.22, 0.22],
    )
    paperless_billing = (rng.random(n) > 0.40).astype(int)

    # ── Usage ─────────────────────────────────────────────────────────────────
    avg_monthly_gb = np.where(
        has_internet_service == 1,
        rng.lognormal(3.2, 1.1, n).clip(0.5, 500).round(2),
        rng.uniform(0, 1, n).round(2),
    )

    # ── Behavioural ───────────────────────────────────────────────────────────
    customer_satisfaction = rng.integers(1, 6, n)  # 1–5
    num_complaints        = rng.poisson(0.5, n).clip(0, 20)
    num_service_calls     = rng.poisson(1.5, n).clip(0, 25)
    late_payments         = rng.poisson(0.8, n).clip(0, 30).astype(float)
    days_since_last_interaction = rng.integers(0, 180, n).astype(float)
    credit_score_arr      = credit_score

    # ── Signup date ───────────────────────────────────────────────────────────
    signup_date = _sample_dates(n, "2018-01-01", "2024-12-31", rng)

    # ── Churn probability (logistic model) ───────────────────────────────────
    # Contract risk
    contract_risk = np.where(contract == "Month-to-month", 1.0,
                    np.where(contract == "One year", 0.0, -1.0))
    # Payment risk (electronic check is highest churn)
    payment_risk  = np.where(payment_method == "Electronic check", 0.5, 0.0)

    z = (
        -1.5                                          # intercept → ~25% base rate
        + 1.4  * contract_risk
        + 0.5  * payment_risk
        - 0.03 * tenure
        - 0.5  * (customer_satisfaction - 3) / 2.0   # low satisfaction ↑ churn
        + 0.5  * (num_complaints / (num_complaints.max() + 1e-9))
        + 0.4  * (num_service_calls / (num_service_calls.max() + 1e-9))
        + 0.6  * (late_payments    / (late_payments.max()    + 1e-9))
        + 0.3  * senior_citizen
        - 0.3  * (num_services / 8.0)                # more services → stickier
        - 0.002 * (credit_score_arr - 680) / 85.0
        + 0.2  * (days_since_last_interaction / 180.0)
        + rng.normal(0, 0.4, n)
    )
    churn_prob = _sigmoid(z)
    churn      = (rng.random(n) < churn_prob).astype(int)

    # ── customer_id ───────────────────────────────────────────────────────────
    customer_id = np.array([f"CUST_{i:07d}" for i in range(1, n + 1)])

    df = pd.DataFrame({
        "customer_id":               customer_id,
        "signup_date":               signup_date,
        "age":                       age,
        "gender":                    gender,
        "annual_income":             income_base,
        "education":                 education,
        "marital_status":            marital_status,
        "dependents":                dependents,
        "tenure":                    tenure,
        "contract":                  contract,
        "payment_method":            payment_method,
        "paperless_billing":         paperless_billing,
        "senior_citizen":            senior_citizen,
        "monthlycharges":            monthlycharges,
        "totalcharges":              totalcharges,
        "num_services":              num_services,
        "has_phone_service":         has_phone_service,
        "has_internet_service":      has_internet_service,
        "has_online_security":       has_online_security,
        "has_online_backup":         has_online_backup,
        "has_device_protection":     has_device_protection,
        "has_tech_support":          has_tech_support,
        "has_streaming_tv":          has_streaming_tv,
        "has_streaming_movies":      has_streaming_movies,
        "customer_satisfaction":     customer_satisfaction,
        "num_complaints":            num_complaints,
        "num_service_calls":         num_service_calls,
        "late_payments":             late_payments,
        "avg_monthly_gb":            avg_monthly_gb,
        "days_since_last_interaction": days_since_last_interaction,
        "credit_score":              credit_score_arr,
        "churn":                     churn,
    })

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 32-column telecom churn dataset")
    parser.add_argument("--n",      type=int,  default=10_000, help="Number of rows")
    parser.add_argument("--seed",   type=int,  default=42,     help="Random seed")
    parser.add_argument("--output", type=str,  default="data/churn.csv")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = generate(n=args.n, seed=args.seed)
    df.to_csv(out, index=False)

    churn_rate = df["churn"].mean() * 100
    print(f"✅  Generated {len(df):,} customer records → {out}")
    print(f"   Columns    : {list(df.columns)}")
    print(f"   Churn rate : {churn_rate:.1f}%")
    print(f"   Shape      : {df.shape}")
