"""
Agent integration tests for the 32-column telecom churn pipeline.

Tests the full pipeline context flow end-to-end using synthetic in-memory
data — no CSV files or MLflow server required.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_customers_df(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """
    Build a minimal synthetic 32-column telecom churn DataFrame.
    Designed so that both churn classes (0 and 1) are present.
    """
    rng = np.random.default_rng(seed)

    churned = int(n * 0.30)
    not_churned = n - churned

    contracts = (
        ["Month-to-month"] * int(churned * 0.7)
        + ["One year"] * int(churned * 0.2)
        + ["Two year"] * (churned - int(churned * 0.7) - int(churned * 0.2))
        + ["Month-to-month"] * int(not_churned * 0.4)
        + ["One year"] * int(not_churned * 0.3)
        + ["Two year"] * (not_churned - int(not_churned * 0.4) - int(not_churned * 0.3))
    )
    # Trim/pad to exactly n
    contracts = (contracts * 2)[:n]

    churn_labels = [1] * churned + [0] * not_churned
    churn_labels = churn_labels[:n]

    df = pd.DataFrame({
        "customer_id":               [f"CUST_{i:07d}" for i in range(1, n + 1)],
        "signup_date":               ["2022-01-01"] * n,
        "age":                       rng.integers(18, 80, n).tolist(),
        "gender":                    rng.choice(["Male", "Female"], n).tolist(),
        "annual_income":             rng.uniform(30_000, 120_000, n).round(2).tolist(),
        "education":                 rng.choice(["High School", "Bachelor", "Graduate", "Postgraduate"], n).tolist(),
        "marital_status":            rng.choice(["Single", "Married", "Divorced"], n).tolist(),
        "dependents":                rng.integers(0, 5, n).tolist(),
        "tenure":                    rng.integers(1, 72, n).tolist(),
        "contract":                  contracts,
        "payment_method":            rng.choice(
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"], n
        ).tolist(),
        "paperless_billing":         rng.integers(0, 2, n).tolist(),
        "senior_citizen":            rng.integers(0, 2, n).tolist(),
        "monthlycharges":            rng.uniform(20, 150, n).round(2).tolist(),
        "totalcharges":              rng.uniform(100, 8000, n).round(2).tolist(),
        "num_services":              rng.integers(1, 9, n).tolist(),
        "has_phone_service":         rng.integers(0, 2, n).tolist(),
        "has_internet_service":      rng.integers(0, 2, n).tolist(),
        "has_online_security":       rng.integers(0, 2, n).tolist(),
        "has_online_backup":         rng.integers(0, 2, n).tolist(),
        "has_device_protection":     rng.integers(0, 2, n).tolist(),
        "has_tech_support":          rng.integers(0, 2, n).tolist(),
        "has_streaming_tv":          rng.integers(0, 2, n).tolist(),
        "has_streaming_movies":      rng.integers(0, 2, n).tolist(),
        "customer_satisfaction":     rng.integers(1, 6, n).tolist(),
        "num_complaints":            rng.integers(0, 5, n).tolist(),
        "num_service_calls":         rng.integers(0, 10, n).tolist(),
        "late_payments":             rng.uniform(0, 5, n).round(2).tolist(),
        "avg_monthly_gb":            rng.uniform(0, 200, n).round(2).tolist(),
        "days_since_last_interaction": rng.integers(0, 180, n).tolist(),
        "credit_score":              rng.uniform(300, 850, n).round(1).tolist(),
        "churn":                     churn_labels,
    })
    return df


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def customers_df() -> pd.DataFrame:
    """Synthetic 32-column telecom churn DataFrame with both classes."""
    return _make_customers_df(n=80, seed=42)


@pytest.fixture
def ingested_context(customers_df: pd.DataFrame) -> dict[str, Any]:
    """Simulate IngestionAgent output without touching disk."""
    from tools.data_tools import impute_churn_data
    clean_df = impute_churn_data(customers_df)
    return {
        "customers_df":  clean_df,
        "n_customers":   len(clean_df),
        "n_rows":        len(clean_df),
        "churn_rate":    float(clean_df["churn"].mean()),
        "data_quality_report": {"quality_score": 95, "churn_rate": float(clean_df["churn"].mean())},
    }


# ── IngestionAgent ────────────────────────────────────────────────────────────


def test_ingestion_agent_raises_without_source():
    from agents.ingestion_agent import IngestionAgent
    agent = IngestionAgent()
    with pytest.raises((ValueError, FileNotFoundError)):
        agent.run({})


def test_ingestion_agent_missing_required_columns():
    """Should raise ValueError when required churn schema columns are absent."""
    from tools.data_tools import validate_churn_schema
    bad_df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_churn_schema(bad_df, source="test")


def test_data_quality_profile_structure(customers_df):
    """profile_churn_data should return a dict with known keys."""
    from tools.data_tools import profile_churn_data
    profile = profile_churn_data(customers_df)
    assert "quality_score" in profile
    assert "churn_rate" in profile
    assert "missing_rates" in profile
    assert "imbalance_ratio" in profile
    assert 0.0 <= profile["churn_rate"] <= 1.0
    assert 0 <= profile["quality_score"] <= 100


def test_imputation_removes_nulls(customers_df):
    """impute_churn_data should leave no nulls in known numeric columns."""
    from tools.data_tools import impute_churn_data
    df_with_nulls = customers_df.copy()
    df_with_nulls.loc[0:5, "tenure"] = np.nan
    df_with_nulls.loc[0:3, "monthlycharges"] = np.nan
    clean = impute_churn_data(df_with_nulls)
    assert clean["tenure"].isnull().sum() == 0
    assert clean["monthlycharges"].isnull().sum() == 0


# ── FeatureAgent ──────────────────────────────────────────────────────────────


def test_feature_agent_builds_matrix(ingested_context: dict):
    from agents.feature_agent import FeatureAgent
    ctx = FeatureAgent().run(ingested_context.copy())
    assert "feature_df"      in ctx
    assert "churn_labels"    in ctx
    assert "feature_columns" in ctx
    feat_df = ctx["feature_df"]
    assert len(feat_df) == ingested_context["n_customers"]


def test_feature_agent_all_churn_columns_present(ingested_context: dict):
    from agents.feature_agent import FeatureAgent
    from tools.feature_tools import CHURN_FEATURE_COLUMNS
    ctx = FeatureAgent().run(ingested_context.copy())
    missing = set(CHURN_FEATURE_COLUMNS) - set(ctx["feature_df"].columns)
    assert not missing, f"Missing CHURN_FEATURE_COLUMNS: {missing}"


def test_feature_agent_churn_labels_binary(ingested_context: dict):
    """Churn labels extracted by FeatureAgent must be 0 or 1 only."""
    from agents.feature_agent import FeatureAgent
    ctx = FeatureAgent().run(ingested_context.copy())
    labels = ctx["churn_labels"]
    assert set(labels.unique()).issubset({0, 1})


def test_feature_agent_no_nan_in_features(ingested_context: dict):
    """Engineered feature matrix should contain no NaN values."""
    from agents.feature_agent import FeatureAgent
    ctx = FeatureAgent().run(ingested_context.copy())
    nan_count = int(ctx["feature_df"].isnull().sum().sum())
    assert nan_count == 0, f"Feature matrix has {nan_count} NaN values"


def test_feature_agent_risk_composite_in_range(ingested_context: dict):
    """risk_composite must be in [0, 1]."""
    from agents.feature_agent import FeatureAgent
    ctx = FeatureAgent().run(ingested_context.copy())
    rc = ctx["feature_df"]["risk_composite"]
    assert (rc >= 0.0).all() and (rc <= 1.0).all()


def test_feature_stats_structure(ingested_context: dict):
    """feature_stats should contain a dict entry per feature column."""
    from agents.feature_agent import FeatureAgent
    from tools.feature_tools import CHURN_FEATURE_COLUMNS
    ctx = FeatureAgent().run(ingested_context.copy())
    stats = ctx["feature_stats"]
    for col in CHURN_FEATURE_COLUMNS[:5]:  # spot-check first 5
        assert col in stats
        assert "mean" in stats[col]
        assert "std"  in stats[col]


# ── PredictionAgent ───────────────────────────────────────────────────────────


def test_prediction_agent_scores_in_range(ingested_context: dict):
    from agents.feature_agent import FeatureAgent
    from agents.prediction_agent import PredictionAgent
    from tools.feature_tools import CHURN_FEATURE_COLUMNS

    ctx = FeatureAgent().run(ingested_context.copy())

    mock_model = MagicMock()
    n = len(ctx["feature_df"])
    scores = np.random.rand(n)
    mock_model.predict_proba.return_value = np.column_stack([1 - scores, scores])
    ctx["model"] = mock_model

    ctx = PredictionAgent().run(ctx)
    scored = ctx["scored_df"]

    assert "churn_score" in scored.columns
    assert "risk_level"  in scored.columns
    assert (scored["churn_score"] >= 0.0).all()
    assert (scored["churn_score"] <= 1.0).all()


def test_prediction_agent_four_tier_risk_labels(ingested_context: dict):
    """Risk levels must only be CRITICAL, HIGH, MEDIUM, or LOW."""
    from agents.feature_agent import FeatureAgent
    from agents.prediction_agent import PredictionAgent

    ctx = FeatureAgent().run(ingested_context.copy())
    n   = len(ctx["feature_df"])
    mock_model = MagicMock()
    scores = np.linspace(0.05, 0.95, n)
    mock_model.predict_proba.return_value = np.column_stack([1 - scores, scores])
    ctx["model"] = mock_model

    ctx = PredictionAgent().run(ctx)
    valid_levels = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    assert set(ctx["scored_df"]["risk_level"].unique()).issubset(valid_levels)


def test_prediction_agent_summary_stats_keys(ingested_context: dict):
    """summary_stats must contain mandatory aggregate keys."""
    from agents.feature_agent import FeatureAgent
    from agents.prediction_agent import PredictionAgent

    ctx = FeatureAgent().run(ingested_context.copy())
    n   = len(ctx["feature_df"])
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.column_stack([
        np.zeros(n), np.ones(n)
    ])
    ctx["model"] = mock_model

    ctx = PredictionAgent().run(ctx)
    stats = ctx["summary_stats"]
    assert stats["total_customers"] == n
    assert stats["critical_risk_count"] + stats["high_risk_count"] > 0


def test_prediction_agent_retention_actions_present(ingested_context: dict):
    """Every scored customer must have a recommended_action."""
    from agents.feature_agent import FeatureAgent
    from agents.prediction_agent import PredictionAgent

    ctx = FeatureAgent().run(ingested_context.copy())
    n   = len(ctx["feature_df"])
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.column_stack([
        np.random.rand(n), np.random.rand(n)
    ])
    ctx["model"] = mock_model

    ctx = PredictionAgent().run(ctx)
    assert "recommended_action" in ctx["scored_df"].columns
    assert ctx["scored_df"]["recommended_action"].notna().all()


# ── ValidationAgent ───────────────────────────────────────────────────────────


def test_validation_agent_produces_full_report(ingested_context: dict):
    from agents.feature_agent import FeatureAgent
    from agents.validation_agent import ValidationAgent
    from tools.feature_tools import CHURN_FEATURE_COLUMNS
    import xgboost as xgb

    ctx = FeatureAgent().run(ingested_context.copy())

    feat_df  = ctx["feature_df"]
    churn_y  = ctx["churn_labels"]
    X = feat_df[CHURN_FEATURE_COLUMNS].fillna(0)

    model = xgb.XGBClassifier(
        n_estimators=10, random_state=42,
        use_label_encoder=False, eval_metric="logloss"
    )
    model.fit(X, churn_y)
    ctx["model"] = model

    ctx = ValidationAgent().run(ctx)

    assert "validation_report"  in ctx
    assert "quality_gate_pass"  in ctx
    assert "quality_gate_notes" in ctx

    report  = ctx["validation_report"]["evaluation_metrics"]
    assert 0.0 <= report["roc_auc"]    <= 1.0
    assert 0.0 <= report["precision"]  <= 1.0
    assert 0.0 <= report["recall"]     <= 1.0
    assert 0.0 <= report["f1"]         <= 1.0
    assert 0.0 <= report["brier_score"] <= 1.0


def test_validation_agent_calibration_present(ingested_context: dict):
    """validation_report must include calibration_analysis sub-section."""
    from agents.feature_agent import FeatureAgent
    from agents.validation_agent import ValidationAgent
    from tools.feature_tools import CHURN_FEATURE_COLUMNS
    import xgboost as xgb

    ctx = FeatureAgent().run(ingested_context.copy())
    feat_df  = ctx["feature_df"]
    churn_y  = ctx["churn_labels"]
    X = feat_df[CHURN_FEATURE_COLUMNS].fillna(0)

    model = xgb.XGBClassifier(n_estimators=10, random_state=42,
                               use_label_encoder=False, eval_metric="logloss")
    model.fit(X, churn_y)
    ctx["model"] = model

    ctx = ValidationAgent().run(ctx)
    cal = ctx["validation_report"]["calibration_analysis"]
    assert "ece" in cal or "error" in cal  # 'error' key allowed if sklearn unavailable


def test_validation_agent_business_impact_present(ingested_context: dict):
    """validation_report must include business_impact sub-section."""
    from agents.feature_agent import FeatureAgent
    from agents.validation_agent import ValidationAgent
    from tools.feature_tools import CHURN_FEATURE_COLUMNS
    import xgboost as xgb

    ctx = FeatureAgent().run(ingested_context.copy())
    feat_df = ctx["feature_df"]
    churn_y = ctx["churn_labels"]
    X = feat_df[CHURN_FEATURE_COLUMNS].fillna(0)

    model = xgb.XGBClassifier(n_estimators=10, random_state=42,
                               use_label_encoder=False, eval_metric="logloss")
    model.fit(X, churn_y)
    ctx["model"] = model

    ctx = ValidationAgent().run(ctx)
    bi = ctx["validation_report"]["business_impact"]
    assert "revenue_protected_usd" in bi
    assert "net_benefit_usd"       in bi


def test_validation_agent_quality_gate_structure(ingested_context: dict):
    """quality_gate_notes must be a list of strings."""
    from agents.feature_agent import FeatureAgent
    from agents.validation_agent import ValidationAgent
    from tools.feature_tools import CHURN_FEATURE_COLUMNS
    import xgboost as xgb

    ctx = FeatureAgent().run(ingested_context.copy())
    feat_df = ctx["feature_df"]
    churn_y = ctx["churn_labels"]
    X = feat_df[CHURN_FEATURE_COLUMNS].fillna(0)

    model = xgb.XGBClassifier(n_estimators=10, random_state=42,
                               use_label_encoder=False, eval_metric="logloss")
    model.fit(X, churn_y)
    ctx["model"] = model

    ctx = ValidationAgent().run(ctx)
    notes = ctx["quality_gate_notes"]
    assert isinstance(notes, list)
    assert all(isinstance(n, str) for n in notes)


# ── Builder / Validators ──────────────────────────────────────────────────────


def test_customer_record_validator_valid():
    """CustomerRecord should accept a valid 32-column payload."""
    from app.features.validators import CustomerRecord
    rec = CustomerRecord(
        customer_id="CUST_0000001",
        tenure=24,
        contract="One year",
        monthlycharges=79.50,
        totalcharges=1908.00,
    )
    assert rec.tenure == 24
    assert rec.contract == "One year"


def test_customer_record_validator_invalid_contract():
    """CustomerRecord should reject an unknown contract type."""
    from pydantic import ValidationError
    from app.features.validators import CustomerRecord
    with pytest.raises(ValidationError):
        CustomerRecord(
            customer_id="X",
            tenure=6,
            contract="Biennial",    # not allowed
            monthlycharges=50.0,
            totalcharges=300.0,
        )


def test_build_customer_feature_dict():
    """build_customer_feature_dict should produce all CHURN_FEATURE_COLUMNS."""
    from app.features.builder import build_customer_feature_dict
    from tools.feature_tools import CHURN_FEATURE_COLUMNS

    raw = {
        "customer_id": "CUST_TEST",
        "age": 40, "gender": "Male", "annual_income": 60_000,
        "education": "Bachelor", "marital_status": "Married",
        "dependents": 1, "senior_citizen": 0,
        "tenure": 18, "contract": "One year",
        "payment_method": "Bank transfer (automatic)",
        "paperless_billing": 1,
        "monthlycharges": 75.0, "totalcharges": 1350.0,
        "num_services": 4,
        "has_phone_service": 1, "has_internet_service": 1,
        "has_online_security": 1, "has_online_backup": 0,
        "has_device_protection": 0, "has_tech_support": 1,
        "has_streaming_tv": 0, "has_streaming_movies": 0,
        "customer_satisfaction": 4, "num_complaints": 0,
        "num_service_calls": 1, "late_payments": 0.0,
        "avg_monthly_gb": 30.5, "days_since_last_interaction": 15.0,
        "credit_score": 720.0,
    }
    features = build_customer_feature_dict(raw)
    missing = set(CHURN_FEATURE_COLUMNS) - set(features.keys())
    assert not missing, f"Missing features: {missing}"
    assert 0.0 <= features["risk_composite"] <= 1.0
    assert features["log_monthly_charges"] > 0


# ── FastAPI endpoint tests ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def api_client():
    """FastAPI TestClient with mocked model."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.25, 0.75]])

    with patch("app.models.loader.load_model", return_value=mock_model), \
         patch("app.models.loader._model_cache", mock_model):
        from app.main import app
        from fastapi.testclient import TestClient
        with TestClient(app) as c:
            yield c


def test_health_returns_ok(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

