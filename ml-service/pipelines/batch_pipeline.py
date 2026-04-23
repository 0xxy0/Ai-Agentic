"""
Batch Pipeline — Ingestion → Feature → Prediction → Report

Runs full batch scoring for all users and generates a PDF/JSON report.
Designed for daily cron execution — does NOT retrain the model.

Usage:
    python pipelines/batch_pipeline.py --csv data/activity.csv

Cron (daily at 2 AM):
    0 2 * * * cd /app && python pipelines/batch_pipeline.py --pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── PYTHONPATH ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from agents.feature_agent import FeatureAgent
from agents.ingestion_agent import IngestionAgent
from agents.prediction_agent import PredictionAgent
from app.core.config import settings
from app.core.logger import get_logger
from reports.pdf_generator import build_dashboard_json, generate_pdf_report

logger = get_logger("batch_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Agent Churn Batch Scoring Pipeline")
    parser.add_argument("--csv",    type=str, default=settings.BATCH_INPUT_PATH)
    parser.add_argument("--output", type=str, default=settings.BATCH_OUTPUT_DIR,
                        help="Directory for CSV + PDF output")
    parser.add_argument("--pdf",    action="store_true", help="Generate PDF report")
    parser.add_argument("--json",   action="store_true", help="Generate dashboard JSON")
    parser.add_argument("--window", type=int, choices=[30, 60, 90],
                        default=settings.CHURN_WINDOW_DAYS)
    return parser.parse_args()


def run_batch_pipeline(
    csv_path: str,
    output_dir: str = "data/batch_results",
    churn_window_days: int = 30,
    generate_pdf: bool = True,
    generate_json: bool = False,
) -> dict:
    """
    Execute the batch scoring pipeline.

    Does NOT train — loads the Production model from MLflow registry.

    Args:
        csv_path: Path to the user activity CSV.
        output_dir: Directory for output artefacts.
        churn_window_days: Churn window for label context (reporting only).
        generate_pdf: If True, generate a PDF report.
        generate_json: If True, save a dashboard JSON file.

    Returns:
        Final pipeline context.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    context: dict = {
        "csv_path":          csv_path,
        "churn_window_days": churn_window_days,
    }

    pipeline = [
        ("Ingestion",  IngestionAgent()),
        ("Feature",    FeatureAgent()),
        ("Prediction", PredictionAgent()),
    ]

    for name, agent in pipeline:
        logger.info(f"Running {name}Agent")
        context = agent.run(context)
        logger.info(f"{name}Agent ✅ done")

    # ── Save scored CSV ───────────────────────────────────────────────────────
    scored_df: pd.DataFrame = context.get("scored_df", pd.DataFrame())
    if not scored_df.empty:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        csv_out = output_dir_path / f"churn_scores_{ts}.csv"
        scored_df.to_csv(csv_out, index=False)
        context["scored_csv_path"] = str(csv_out)
        logger.info("Scored CSV saved", extra={"path": str(csv_out)})

    # ── Optional PDF ──────────────────────────────────────────────────────────
    if generate_pdf:
        pdf_path = generate_pdf_report(context, output_dir=str(output_dir_path / "reports"))
        context["pdf_path"] = str(pdf_path)

    # ── Optional JSON ─────────────────────────────────────────────────────────
    if generate_json:
        dashboard = build_dashboard_json(context)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        json_path = output_dir_path / f"dashboard_{ts}.json"
        json_path.write_text(json.dumps(dashboard, indent=2))
        context["json_path"] = str(json_path)
        logger.info("Dashboard JSON saved", extra={"path": str(json_path)})

    return context


def main() -> None:
    args = parse_args()

    logger.info(
        "Batch pipeline started",
        extra={"csv_path": args.csv, "output_dir": args.output},
    )

    try:
        ctx = run_batch_pipeline(
            csv_path=args.csv,
            output_dir=args.output,
            churn_window_days=args.window,
            generate_pdf=args.pdf,
            generate_json=args.json,
        )

        stats = ctx.get("summary_stats", {})
        print(f"\n{'='*60}")
        print(f"  ✅  Batch scoring complete")
        print(f"  Total users  : {stats.get('total_users', 'N/A')}")
        print(f"  High risk    : {stats.get('high_risk_count', 'N/A')}")
        print(f"  Medium risk  : {stats.get('medium_risk_count', 'N/A')}")
        print(f"  Low risk     : {stats.get('low_risk_count', 'N/A')}")
        print(f"  Avg score    : {stats.get('avg_churn_score', 'N/A')}")
        if ctx.get("scored_csv_path"):
            print(f"  Output CSV   : {ctx['scored_csv_path']}")
        if ctx.get("pdf_path"):
            print(f"  PDF report   : {ctx['pdf_path']}")
        if ctx.get("json_path"):
            print(f"  Dashboard JSON: {ctx['json_path']}")
        print(f"{'='*60}\n")

    except Exception as exc:
        logger.error("Batch pipeline failed", extra={"error": str(exc)})
        print(f"\n❌ Batch pipeline failed: {exc}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
