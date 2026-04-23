"""
Unit tests for the prediction endpoint.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client():
    """
    Create a TestClient with a mocked MLflow model.

    We mock ``app.models.loader.load_model`` so tests run without
    a live MLflow server.
    """
    mock_model = MagicMock()
    # Simulate XGBoost predict_proba output shape: [[prob_0, prob_1], ...]
    import numpy as np
    mock_model.predict.return_value = np.array([[0.3, 0.72]])

    with patch("app.models.loader.load_model", return_value=mock_model):
        with patch("app.models.loader._model_cache", mock_model):
            from app.main import app
            with TestClient(app) as c:
                yield c


@pytest.fixture
def valid_payload() -> dict[str, Any]:
    return {
        "user_id": "test_user_001",
        "txn_7d": 2,
        "txn_30d": 8,
        "txn_90d": 25,
        "recency_days": 10,
        "frequency": 30,
        "monetary": 750.0,
        "account_age_days": 180,
        "plan_tier": "basic",
    }


@pytest.fixture
def high_churn_payload() -> dict[str, Any]:
    """User with very high recency — likely churned."""
    return {
        "user_id": "dormant_user",
        "txn_7d": 0,
        "txn_30d": 0,
        "txn_90d": 1,
        "recency_days": 88,
        "frequency": 3,
        "monetary": 50.0,
    }


# ── Health tests ──────────────────────────────────────────────────────────────


def test_health_endpoint_returns_ok(client: TestClient):
    """Health check must return HTTP 200 with status=ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert "version" in data


# ── Prediction tests ──────────────────────────────────────────────────────────


def test_predict_returns_200(client: TestClient, valid_payload: dict):
    """POST /predict must return HTTP 200."""
    with patch("app.models.predictor.load_model") as mock_load:
        import numpy as np
        mock = MagicMock()
        mock.predict.return_value = np.array([[0.28, 0.72]])
        mock_load.return_value = mock

        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200


def test_churn_score_in_valid_range(client: TestClient, valid_payload: dict):
    """churn_score must be strictly within [0, 1]."""
    with patch("app.models.predictor.load_model") as mock_load:
        import numpy as np
        mock = MagicMock()
        mock.predict.return_value = np.array([[0.28, 0.72]])
        mock_load.return_value = mock

        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200

        body = response.json()
        assert "churn_score" in body
        score = body["churn_score"]
        assert 0.0 <= score <= 1.0, f"churn_score out of range: {score}"


def test_risk_level_is_valid(client: TestClient, valid_payload: dict):
    """risk_level must be one of LOW / MEDIUM / HIGH."""
    with patch("app.models.predictor.load_model") as mock_load:
        import numpy as np
        mock = MagicMock()
        mock.predict.return_value = np.array([[0.28, 0.72]])
        mock_load.return_value = mock

        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200
        assert response.json()["risk_level"] in {"LOW", "MEDIUM", "HIGH"}


def test_predict_returns_user_id(client: TestClient, valid_payload: dict):
    """Response must echo the user_id from the request."""
    with patch("app.models.predictor.load_model") as mock_load:
        import numpy as np
        mock = MagicMock()
        mock.predict.return_value = np.array([[0.28, 0.72]])
        mock_load.return_value = mock

        response = client.post("/predict", json=valid_payload)
        assert response.json()["user_id"] == valid_payload["user_id"]


def test_predict_missing_required_field_returns_422(client: TestClient):
    """Payload missing recency_days must fail validation with HTTP 422."""
    incomplete = {
        "user_id": "bad_user",
        "txn_7d": 1,
        "txn_30d": 5,
        "txn_90d": 10,
        # recency_days intentionally omitted
        "frequency": 10,
        "monetary": 200.0,
    }
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_invalid_txn_order_returns_422(client: TestClient):
    """txn_7d > txn_30d must fail Pydantic validator with HTTP 422."""
    bad_payload = {
        "user_id": "bad_user",
        "txn_7d": 20,       # violates: txn_7d > txn_30d
        "txn_30d": 5,
        "txn_90d": 10,
        "recency_days": 3,
        "frequency": 10,
        "monetary": 200.0,
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


# ── Batch prediction tests ────────────────────────────────────────────────────


def test_batch_predict_returns_correct_count(client: TestClient, valid_payload: dict):
    """Batch response total must equal number of submitted users."""
    with patch("app.models.predictor.load_model") as mock_load:
        import numpy as np
        mock = MagicMock()
        mock.predict.return_value = np.array([[0.28, 0.72], [0.6, 0.4]])
        mock_load.return_value = mock

        payload = {
            "users": [valid_payload, {**valid_payload, "user_id": "user_002"}],
            "churn_window_days": 30,
        }
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["predictions"]) == 2


def test_batch_predict_all_scores_in_range(client: TestClient, valid_payload: dict):
    """All churn_scores in a batch response must be in [0, 1]."""
    with patch("app.models.predictor.load_model") as mock_load:
        import numpy as np
        mock = MagicMock()
        mock.predict.return_value = np.array([[0.28, 0.72], [0.6, 0.4]])
        mock_load.return_value = mock

        payload = {
            "users": [valid_payload, {**valid_payload, "user_id": "user_002"}],
            "churn_window_days": 30,
        }
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        for pred in response.json()["predictions"]:
            assert 0.0 <= pred["churn_score"] <= 1.0
