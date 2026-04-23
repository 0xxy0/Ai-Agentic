# ══════════════════════════════════════════════════════════════════════════════
#  Python FastAPI ML Microservice Dockerfile — Production Multi-Stage Build
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: build dependencies ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install compile-time tools for native extensions (xgboost, scipy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: production runtime ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="mlops-team"
LABEL description="ChurnAI FastAPI ML Microservice — XGBoost + MLflow"

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy ALL application modules required at runtime
COPY app/        ./app/
COPY pipelines/  ./pipelines/
COPY tools/      ./tools/
COPY agents/     ./agents/

# Ensure batch output and report directories exist
RUN mkdir -p data/batch_results data/batch_results/reports reports/output mlflow

# ── Security: non-root user ───────────────────────────────────────────────────
RUN useradd -m -u 1001 mlservice && \
    chown -R mlservice:mlservice /app
USER mlservice

# ── Environment defaults — override in docker-compose or Kubernetes secrets ────
ENV APP_NAME="ChurnPredictionService" \
    APP_VERSION="1.0.0" \
    ENV="production" \
    DEBUG="false" \
    API_HOST="0.0.0.0" \
    API_PORT="8000" \
    LOG_LEVEL="INFO" \
    LOG_FORMAT="json" \
    MLFLOW_TRACKING_URI="http://mlflow:5000" \
    MLFLOW_MODEL_NAME="ChurnModel" \
    MLFLOW_MODEL_STAGE="Production" \
    CHURN_WINDOW_DAYS="30" \
    RISK_LOW_THRESHOLD="0.3" \
    RISK_HIGH_THRESHOLD="0.7" \
    BATCH_OUTPUT_DIR="data/batch_results" \
    ACTIVITY_CSV_PATH="data/activity.csv" \
    REPORTS_OUTPUT_DIR="reports/output"

EXPOSE 8000

# Docker HEALTHCHECK — Compose uses this for depends_on: service_healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# uvicorn with 2 workers — enough for a single-instance ML service
# (CPU-bound ML inference doesn't benefit from more workers than cores)
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info", \
     "--no-access-log"]
