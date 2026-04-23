"""
Training Pipeline — Ingestion → Feature → Modeling → Validation → Report

Orchestrates the multi-agent training flow from the command line.

Usage:
    python pipelines/training_pipeline.py --csv data/activity.csv --window 30

Designed for:
    * One-time training runs
    * Scheduled retraining (cron / Airflow)
    * CI/CD model refreshes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── PYTHONPATH ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.feature_agent import FeatureAgent
from agents.ingestion_agent import IngestionAgent
from agents.modeling_agent import ModelingAgent
from agents.validation_agent import ValidationAgent
from app.core.config import settings
from app.core.logger import get_logger
from reports.pdf_generator import generate_pdf_report

logger = get_logger("training_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Agent Churn Training Pipeline")
    parser.add_argument("--csv",    type=str, default=settings.BATCH_INPUT_PATH,
                        help="Path to the activity CSV")
    parser.add_argument("--window", type=int, choices=[30, 60, 90],
                        default=settings.CHURN_WINDOW_DAYS,
                        help="Churn definition window (days)")
    parser.add_argument("--no-register", action="store_true",
                        help="Skip MLflow model registration")
    parser.add_argument("--pdf",    action="store_true",
                        help="Generate a PDF report after training")
    return parser.parse_args()


def run_training_pipeline(
    csv_path: str,
    churn_window_days: int = 30,
    register_model: bool = True,
    generate_pdf: bool = False,
) -> dict:
    """
    Execute the full training pipeline.

    Returns:
        Final pipeline context dict.
    """
    context: dict = {
        "csv_path":            csv_path,
        "churn_window_days":   churn_window_days,
        "register_model":      register_model,
    }

    pipeline = [
        ("Ingestion",   IngestionAgent()),
        ("Feature",     FeatureAgent()),
        ("Modeling",    ModelingAgent()),
        ("Validation",  ValidationAgent()),
    ]

    for name, agent in pipeline:
        logger.info(f"Running {name}Agent")
        context = agent.run(context)
        logger.info(f"{name}Agent ✅ done")

    if generate_pdf:
        pdf_path = generate_pdf_report(context, output_dir="reports/output")
        context["pdf_path"] = str(pdf_path)
        logger.info("PDF report written", extra={"pdf_path": str(pdf_path)})

    return context


def main() -> None:
    args = parse_args()

    logger.info(
        "Training pipeline started",
        extra={
            "csv_path":    args.csv,
            "window":      args.window,
            "register":    not args.no_register,
        },
    )

    try:
        ctx = run_training_pipeline(
            csv_path=args.csv,
            churn_window_days=args.window,
            register_model=not args.no_register,
            generate_pdf=args.pdf,
        )

        metrics = ctx.get("train_metrics", {})
        print(f"\n{'='*60}")
        print(f"  ✅  Training complete")
        print(f"  Run ID      : {ctx.get('run_id', 'N/A')}")
        print(f"  ROC-AUC     : {metrics.get('roc_auc', 'N/A')}")
        print(f"  Precision   : {metrics.get('precision', 'N/A')}")
        print(f"  Recall      : {metrics.get('recall', 'N/A')}")
        print(f"  Gate passed : {ctx.get('quality_gate_pass', 'N/A')}")
        if ctx.get("pdf_path"):
            print(f"  PDF report  : {ctx['pdf_path']}")
        print(f"{'='*60}\n")

    except Exception as exc:
        logger.error("Training pipeline failed", extra={"error": str(exc)})
        print(f"\n❌ Pipeline failed: {exc}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
