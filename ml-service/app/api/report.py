"""
FastAPI endpoints for the multi-agent churn system.

New endpoints added on top of the existing /predict:
    GET  /report/dashboard — returns the latest dashboard JSON
    POST /report/generate  — triggers batch pipeline + PDF generation
    GET  /pipeline/status  — health of all agents
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/report", tags=["Reports & Pipeline"])


@router.get("/dashboard", summary="Latest dashboard JSON (frontend-ready)")
def get_dashboard_json() -> JSONResponse:
    """
    Return the most recent dashboard JSON produced by the batch pipeline.

    Node.js compatible — pure JSON.

    Raises:
        404: If no dashboard JSON has been generated yet.
    """
    output_dir = Path(settings.BATCH_OUTPUT_DIR)
    json_files = sorted(output_dir.glob("dashboard_*.json"), reverse=True)

    if not json_files:
        raise HTTPException(
            status_code=404,
            detail="No dashboard JSON found. Run the batch pipeline first.",
        )

    latest = json_files[0]
    data   = json.loads(latest.read_text())
    return JSONResponse(content=data)


@router.post("/generate", summary="Trigger batch pipeline + PDF report")
def generate_report(background_tasks: BackgroundTasks) -> dict[str, str]:
    """
    Asynchronously triggers the batch scoring pipeline and generates a PDF.

    Returns immediately with a confirmation message. Check logs or
    ``GET /report/dashboard`` for results.
    """
    def _run():
        import sys
        sys.path.insert(0, ".")
        from pipelines.batch_pipeline import run_batch_pipeline
        try:
            run_batch_pipeline(
                csv_path=settings.ACTIVITY_CSV_PATH,
                output_dir=settings.BATCH_OUTPUT_DIR,
                churn_window_days=settings.CHURN_WINDOW_DAYS,
                generate_pdf=True,
                generate_json=True,
            )
            logger.info("Background batch pipeline completed")
        except Exception as exc:
            logger.error("Background batch pipeline failed", extra={"error": str(exc)})

    background_tasks.add_task(_run)
    return {
        "status": "accepted",
        "message": "Batch pipeline started in background. Check GET /report/dashboard for results.",
    }


@router.get("/pdf/latest", summary="Download latest PDF report")
def download_latest_pdf() -> FileResponse:
    """Return the most recently generated PDF report as a file download."""
    reports_dir = Path(settings.REPORTS_OUTPUT_DIR)
    pdfs = sorted(reports_dir.glob("churn_report_*.pdf"), reverse=True)

    if not pdfs:
        raise HTTPException(
            status_code=404,
            detail="No PDF reports found. Run the training or batch pipeline with --pdf.",
        )

    return FileResponse(
        path=str(pdfs[0]),
        media_type="application/pdf",
        filename=pdfs[0].name,
    )


@router.get("/csv/latest", summary="Download latest raw predictions CSV")
def download_latest_csv() -> FileResponse:
    """Return the most recently generated predictions CSV as a file download."""
    output_dir = Path(settings.BATCH_OUTPUT_DIR)
    csv_files = sorted(output_dir.glob("churn_scores_*.csv"), reverse=True)

    if not csv_files:
        raise HTTPException(
            status_code=404,
            detail="No CSV prediction files found. Run the batch pipeline first.",
        )

    return FileResponse(
        path=str(csv_files[0]),
        media_type="text/csv",
        filename=csv_files[0].name,
    )
