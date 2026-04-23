# ChurnPredictionService v2 — Multi-Agent ML Backend

> **Production-ready Python ML backend** with a **5-agent pipeline** for customer churn prediction.  
> Built with **FastAPI · XGBoost · MLflow · ReportLab** on time-series (monthly/yearly) data.

---

## 🏗 System Architecture

```
CSV / SQLite
     │
     ▼
┌─────────────────┐
│ IngestionAgent  │  Load → Validate → Normalise to time-series format
└────────┬────────┘
         │  ts_df (user_id | year | month | txn_count | spend)
         ▼
┌─────────────────┐
│  FeatureAgent   │  Rolling averages · OLS trends · Activity gap · Log transforms
└────────┬────────┘
         │  feature_df (14 columns per user)
         ▼
┌─────────────────┐
│ ModelingAgent   │  Time-based split · XGBoost · class imbalance · MLflow logging
└────────┬────────┘      (training pipeline only)
         │
         ▼
┌─────────────────┐
│PredictionAgent  │  Batch score · churn_score [0,1] · risk_level LOW/MEDIUM/HIGH
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ValidationAgent  │  ROC-AUC · Precision · Recall · Confusion matrix · Quality gates
└────────┬────────┘
         │
         ▼
    PDF Report  /  Dashboard JSON  /  Scored CSV
```

---

## 📁 Project Structure

```
ml-service/
│
├── agents/
│   ├── ingestion_agent.py     # Phase 1: load + normalise
│   ├── feature_agent.py       # Phase 2: 14 time-series features
│   ├── modeling_agent.py      # Phase 3: train XGBoost + MLflow
│   ├── prediction_agent.py    # Phase 4: batch scoring
│   └── validation_agent.py    # Phase 5: evaluation + quality gates
│
├── tools/
│   ├── data_tools.py          # CSV/SQLite loading, time-series normalisation
│   ├── feature_tools.py       # Rolling stats, OLS trends, outlier capping
│   └── model_tools.py         # MLflow helpers, XGBoost defaults, metrics
│
├── pipelines/
│   ├── training_pipeline.py   # Agents 1–3–4–5 + optional PDF
│   └── batch_pipeline.py      # Agents 1–2–4 (no retraining) + CSV/PDF/JSON
│
├── app/
│   ├── api/
│   │   ├── predict.py         # POST /predict, POST /predict/batch
│   │   ├── train.py           # POST /train
│   │   ├── health.py          # GET /health
│   │   └── report.py          # GET /report/dashboard, POST /report/generate
│   ├── core/
│   │   ├── config.py          # Pydantic Settings (all env vars)
│   │   └── logger.py          # Structured JSON logger
│   └── main.py                # App factory, CORS, lifespan
│
├── reports/
│   └── pdf_generator.py       # ReportLab PDF + dashboard JSON builder
│
├── data/
│   ├── generate_activity_data.py   # Monthly time-series data generator
│   └── generate_sample_data.py     # Legacy flat data generator
│
├── tests/
│   ├── test_predict.py        # Existing FastAPI endpoint tests
│   └── test_agents.py         # Agent pipeline integration tests
│
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚡ Quick Start

### 1. Setup

```bash
cd ml-service
python -m venv venv && venv\Scripts\activate    # Windows
pip install -r requirements.txt
cp .env.example .env
```

### 2. Start MLflow server

```bash
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow/mlflow.db \
  --default-artifact-root mlflow/artifacts
```

### 3. Generate time-series activity data

```bash
python data/generate_activity_data.py --users 500 --months 18 --output data/activity.csv
```

### 4. Run training pipeline (agents 1→2→3→4→5)

```bash
python pipelines/training_pipeline.py --csv data/activity.csv --window 30 --pdf
```

Promote the new model version to **Production** at `http://localhost:5000`.

### 5. Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Run daily batch scoring

```bash
python pipelines/batch_pipeline.py --csv data/activity.csv --pdf --json
```

### 7. Run tests

```bash
pytest tests/ -v
```

---

## 📡 API Reference

### `GET /health`
```json
{ "status": "ok", "version": "1.0.0", "uptime_seconds": 42.1 }
```

### `POST /predict`  — single user
```json
// Request
{ "user_id": "usr_001", "txn_7d": 3, "txn_30d": 10, "txn_90d": 28,
  "recency_days": 5, "frequency": 45, "monetary": 1250.00 }

// Response
{ "user_id": "usr_001", "churn_score": 0.142, "risk_level": "LOW" }
```

### `POST /predict/batch`  — up to 1000 users
```json
// Response
{ "predictions": [...], "total": 100, "high_risk_count": 23,
  "medium_risk_count": 41, "low_risk_count": 36 }
```

### `GET /report/dashboard`
Returns the latest batch-pipeline dashboard JSON (frontend-ready).

### `POST /report/generate`
Triggers the batch pipeline in the background and generates PDF + JSON.

### `GET /report/pdf/latest`
Downloads the most recently generated PDF report.

### `GET /forecast`
Retrieve quarterly forecast predictions from MongoDB. Optional filter: `?month=Q3 2025`.

### `POST /forecast/refresh`
Manually refresh the forecast database from the raw predictions CSV.

---

## 📊 Time-Series Feature Matrix (14 features)

| Feature | Description |
|---|---|
| `txn_last_month` | Transactions in most recent month |
| `txn_3_month_avg` | 3-month rolling average |
| `txn_6_month_avg` | 6-month rolling average |
| `txn_trend` | OLS slope of txn series (↓ = churn signal) |
| `spend_last_month` | Spend in most recent month |
| `spend_3_month_avg` | 3-month rolling avg spend |
| `spend_6_month_avg` | 6-month rolling avg spend |
| `spend_trend` | OLS slope of spend series |
| `total_txn` | Lifetime transaction count |
| `total_spend` | Lifetime spend (IQR-capped) |
| `log_total_spend` | log1p-transformed spend |
| `activity_gap` | Months since last active (capped 24) |
| `n_active_months` | Months with any recorded activity |
| `spend_per_txn` | Average spend per transaction |

---

## 📄 PDF Report Sections

1. **Executive Summary** — total users, churn window
2. **Risk Distribution** — HIGH / MEDIUM / LOW counts + percentages
3. **Model Validation Metrics** — ROC-AUC, Accuracy, Precision, Recall, Confusion Matrix
4. **High-Risk Users** — top 20 scored users table
5. **Trend Summary** — narrative insight paragraph

---

## 🧠 Agent Contracts

Each agent reads from and writes to a **shared context dict**.
No agent has side-effects on another agent's responsibility.

```
IngestionAgent   → adds: ts_df, n_users, n_rows
FeatureAgent     → adds: feature_df, feature_columns, reference_period
ModelingAgent    → adds: model, run_id, train_metrics, train_end_month/year
PredictionAgent  → adds: scored_df, high_risk_users, summary_stats
ValidationAgent  → adds: validation_report, quality_gate_pass, quality_gate_notes
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |
| `MLFLOW_MODEL_NAME` | `ChurnModel` | Registry model name |
| `MLFLOW_MODEL_STAGE` | `Production` | Stage to load at startup |
| `CHURN_WINDOW_DAYS` | `30` | Churn window: 30/60/90 |
| `MIN_ROC_AUC` | `0.65` | Quality gate threshold |
| `MIN_PRECISION` | `0.40` | Quality gate threshold |
| `MIN_RECALL` | `0.40` | Quality gate threshold |
| `ACTIVITY_CSV_PATH` | `data/activity.csv` | Default time-series input |
| `REPORTS_OUTPUT_DIR` | `reports/output` | PDF output directory |

---

## 🐳 Docker

```bash
docker build -t churn-service:latest .
docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  churn-service:latest
```

---

## 🔗 Node.js Integration

```javascript
// Single prediction
const res = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ user_id: 'u1', txn_7d: 2, txn_30d: 8,
    txn_90d: 22, recency_days: 12, frequency: 30, monetary: 800 })
});
const { churn_score, risk_level } = await res.json();

// Trigger batch report
await fetch('http://localhost:8000/report/generate', { method: 'POST' });

// Fetch dashboard
const dashboard = await fetch('http://localhost:8000/report/dashboard').then(r => r.json());
```

---

## 🗓 Cron Schedule

```cron
# Daily batch scoring at 02:00
0 2 * * * cd /app && python pipelines/batch_pipeline.py --pdf --json

# Weekly model retraining on Sunday at 03:00
0 3 * * 0 cd /app && python pipelines/training_pipeline.py --pdf
```
