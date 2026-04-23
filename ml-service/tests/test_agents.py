"""
Agent integration tests.

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


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_ts_df() -> pd.DataFrame:
    """Minimal synthetic monthly time-series DataFrame."""
    rows = []
    for uid in range(30):
        for mo in range(1, 13):
            rows.append({
                "user_id":   f"usr_{uid:03d}",
                "year":      2024,
                "month":     mo,
                "txn_count": max(0, 10 - mo) if uid < 5 else (mo % 4 + 1),
                "spend":     float(max(0, 500 - 30 * mo)) if uid < 5 else float(100 + 20 * (mo % 3)),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def ingested_context(sample_ts_df: pd.DataFrame) -> dict[str, Any]:
    """Simulate the output of the IngestionAgent without touching disk."""
    from tools.data_tools import to_time_series
    ts_df = to_time_series(sample_ts_df)
    return {
        "ts_df":   ts_df,
        "n_users": ts_df["user_id"].nunique(),
        "n_rows":  len(ts_df),
    }


# ── IngestionAgent ────────────────────────────────────────────────────────────


def test_ingestion_agent_raises_without_source():
    from agents.ingestion_agent import IngestionAgent
    agent = IngestionAgent()
    with pytest.raises((ValueError, FileNotFoundError)):
        agent.run({})


# ── FeatureAgent ──────────────────────────────────────────────────────────────


def test_feature_agent_builds_matrix(ingested_context: dict):
    from agents.feature_agent import FeatureAgent
    ctx = FeatureAgent().run(ingested_context.copy())

    assert "feature_df"      in ctx
    assert "feature_columns" in ctx
    feat_df = ctx["feature_df"]
    assert len(feat_df) == ingested_context["n_users"]


def test_feature_agent_all_columns_present(ingested_context: dict):
    from agents.feature_agent import FeatureAgent
    from tools.feature_tools import FEATURE_COLUMNS
    ctx = FeatureAgent().run(ingested_context.copy())
    missing = set(FEATURE_COLUMNS) - set(ctx["feature_df"].columns)
    assert not missing, f"Missing columns: {missing}"


def test_activity_gap_non_negative(ingested_context: dict):
    from agents.feature_agent import FeatureAgent
    ctx = FeatureAgent().run(ingested_context.copy())
    assert (ctx["feature_df"]["activity_gap"] >= 0).all()


# ── PredictionAgent ───────────────────────────────────────────────────────────


def test_prediction_agent_scores_in_range(ingested_context: dict):
    from agents.feature_agent import FeatureAgent
    from agents.prediction_agent import PredictionAgent
    from tools.feature_tools import FEATURE_COLUMNS

    ctx = FeatureAgent().run(ingested_context.copy())

    # Mock a trained XGBClassifier
    mock_model = MagicMock()
    n = len(ctx["feature_df"])
    mock_model.predict_proba.return_value = np.column_stack([
        np.random.rand(n), np.random.rand(n)
    ])
    ctx["model"] = mock_model

    ctx = PredictionAgent().run(ctx)
    scored = ctx["scored_df"]

    assert "churn_score" in scored.columns
    assert "risk_level"  in scored.columns
    assert (scored["churn_score"] >= 0.0).all()
    assert (scored["churn_score"] <= 1.0).all()


def test_prediction_agent_risk_labels_valid(ingested_context: dict):
    from agents.feature_agent import FeatureAgent
    from agents.prediction_agent import PredictionAgent

    ctx = FeatureAgent().run(ingested_context.copy())

    mock_model = MagicMock()
    n = len(ctx["feature_df"])
    mock_model.predict_proba.return_value = np.column_stack([
        np.zeros(n), np.ones(n)   # All scores = 1.0 → all HIGH
    ])
    ctx["model"] = mock_model

    ctx = PredictionAgent().run(ctx)
    assert set(ctx["scored_df"]["risk_level"].unique()) <= {"HIGH", "MEDIUM", "LOW"}


def test_prediction_agent_summary_stats_correct(ingested_context: dict):
    from agents.feature_agent import FeatureAgent
    from agents.prediction_agent import PredictionAgent

    ctx = FeatureAgent().run(ingested_context.copy())
    n = len(ctx["feature_df"])

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.column_stack([
        np.zeros(n), np.ones(n)
    ])
    ctx["model"] = mock_model

    ctx = PredictionAgent().run(ctx)
    stats = ctx["summary_stats"]
    assert stats["total_users"]     == n
    assert stats["high_risk_count"] == n    # all scores = 1.0


# ── ValidationAgent ───────────────────────────────────────────────────────────


def test_validation_agent_produces_report(ingested_context: dict):
    from agents.feature_agent import FeatureAgent
    from agents.validation_agent import ValidationAgent
    from tools.feature_tools import FEATURE_COLUMNS
    import xgboost as xgb

    ctx = FeatureAgent().run(ingested_context.copy())

    # Train a real tiny XGBoost model in-memory
    feat_df = ctx["feature_df"]
    X = feat_df[FEATURE_COLUMNS].fillna(0)
    y = (feat_df["activity_gap"] >= 1).astype(int)

    model = xgb.XGBClassifier(n_estimators=5, random_state=42, use_label_encoder=False,
                               eval_metric="logloss")
    model.fit(X, y)

    ctx["model"] = model
    ctx = ValidationAgent().run(ctx)

    assert "validation_report"  in ctx
    assert "quality_gate_pass"  in ctx
    assert "quality_gate_notes" in ctx
    report = ctx["validation_report"]["evaluation_metrics"]
    assert 0.0 <= report["roc_auc"]   <= 1.0
    assert 0.0 <= report["precision"] <= 1.0
    assert 0.0 <= report["recall"]    <= 1.0


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


def test_predict_churn_score_in_range(api_client):
    with patch("app.models.predictor.load_model") as ml:
        mock = MagicMock()
        mock.predict.return_value = np.array([[0.25, 0.75]])
        ml.return_value = mock

        payload = {
            "user_id": "usr_test",
            "txn_7d": 2, "txn_30d": 8, "txn_90d": 22,
            "recency_days": 10,
            "frequency": 30,
            "monetary": 800.0,
        }
        r = api_client.post("/predict", json=payload)
        assert r.status_code == 200
        score = r.json()["churn_score"]
        assert 0.0 <= score <= 1.0
