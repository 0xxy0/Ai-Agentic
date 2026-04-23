"""
Core configuration module.

Loads all settings from environment variables with sane defaults.
Never hardcode secrets — use a .env file or deployment secrets manager.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── App ───────────────────────────────────────────────────────────────
    APP_NAME: str = "ChurnPredictionService"
    APP_VERSION: str = "1.0.0"
    ENV: str = "development"           # development | staging | production
    DEBUG: bool = False

    # ── API ───────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    PYTHON_PORT: int = 8000
    CORS_ORIGINS: list[str] = ["*"]   # Restrict in production

    # ── MongoDB ───────────────────────────────────────────────────────────
    MONGO_URI: str = "mongodb://localhost:27017/agentic"

    # ── MLflow ────────────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "ChurnPrediction"
    MLFLOW_MODEL_NAME: str = "ChurnModel"
    MLFLOW_MODEL_STAGE: str = "Production"     # Production | Staging | None

    # ── Model ─────────────────────────────────────────────────────────────
    MODEL_URI: Optional[str] = None    # Override: models:/ChurnModel/Production
    CHURN_WINDOW_DAYS: int = 30        # 30 | 60 | 90

    # ── Batch ─────────────────────────────────────────────────────────────
    BATCH_OUTPUT_DIR: str = "data/batch_results"
    BATCH_INPUT_PATH: str = "data/users.csv"

    # ── Logging ───────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"           # json | text

    # ── Risk thresholds ───────────────────────────────────────────────────
    RISK_LOW_THRESHOLD: float = 0.3
    RISK_HIGH_THRESHOLD: float = 0.7

    # ── Multi-agent quality gates ──────────────────────────────────────────
    MIN_ROC_AUC: float = 0.65
    MIN_PRECISION: float = 0.40
    MIN_RECALL: float = 0.40

    # ── Reports ───────────────────────────────────────────────────────────
    REPORTS_OUTPUT_DIR: str = "reports/output"
    ACTIVITY_CSV_PATH: str = "data/activity.csv"   # time-series format

    class Config:
        env_file = "../../.env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()


# Module-level convenience alias
settings = get_settings()
