"""
PDF Report Generator.

Produces a professional churn analysis PDF report using ReportLab.
The report includes:
    * Executive summary (total users, churn distribution)
    * Risk breakdown table
    * High-risk user list
    * Model validation metrics
    * Trend summary

Usage::

    from reports.pdf_generator import generate_pdf_report
    path = generate_pdf_report(context, output_dir="reports/output")

"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.logger import get_logger

logger = get_logger(__name__)


def _try_import_reportlab() -> bool:
    """Check if reportlab is installed."""
    try:
        import reportlab  # noqa: F401
        return True
    except ImportError:
        return False


def generate_pdf_report(
    context: dict[str, Any],
    output_dir: str | Path = "reports/output",
) -> Path:
    """
    Generate a PDF churn analysis report from the pipeline context.

    Args:
        context: Completed pipeline context containing ``scored_df``,
                 ``summary_stats``, ``validation_report``, and
                 ``quality_gate_notes``.
        output_dir: Directory where the PDF will be written.

    Returns:
        :class:`pathlib.Path` to the generated PDF file.

    Raises:
        ImportError: If ``reportlab`` is not installed.
    """
    if not _try_import_reportlab():
        raise ImportError(
            "reportlab is not installed. Run: pip install reportlab"
        )

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    import pandas as pd

    # ── Output path ───────────────────────────────────────────────────────────
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp  = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    pdf_path   = output_dir / f"churn_report_{timestamp}.pdf"

    # ── Collect context data ───────────────────────────────────────────────────
    summary: dict[str, Any]    = context.get("summary_stats", {})
    val_report: dict[str, Any] = context.get("validation_report", {})
    eval_m: dict[str, Any]     = val_report.get("evaluation_metrics", {})
    scored_df: pd.DataFrame    = context.get("scored_df", pd.DataFrame())
    gate_notes: list[str]      = context.get("quality_gate_notes", [])
    gate_pass: bool            = context.get("quality_gate_pass", False)
    run_id: str                = context.get("run_id", "N/A")
    n_users: int               = context.get("n_users", summary.get("total_users", 0))

    # ── Document setup ─────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        spaceAfter=6,
        textColor=colors.HexColor("#1a1a2e"),
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=12,
        spaceAfter=4,
        textColor=colors.HexColor("#16213e"),
    )
    normal = styles["Normal"]
    body   = ParagraphStyle("Body", parent=normal, fontSize=10, leading=14)

    # ── Table styles ──────────────────────────────────────────────────────────
    def _table_style(header_color: str = "#0f3460") -> TableStyle:
        return TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor(header_color)),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0), 10),
            ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#f5f5f5"), colors.white]),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("TOPPADDING",  (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])

    story: list = []

    # ══ PAGE 1: EXECUTIVE SUMMARY ════════════════════════════════════════════

    story.append(Paragraph("Churn Prediction Analysis Report", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"| MLflow Run: <font color='#0f3460'>{run_id}</font>",
        body,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f3460")))
    story.append(Spacer(1, 8))

    # ── Executive Summary ─────────────────────────────────────────────────────
    story.append(Paragraph("1. Executive Summary", heading_style))

    summary_data = [
        ["Metric", "Value"],
        ["Total Users Analysed",   str(n_users)],
        ["High Risk Users",        str(summary.get("high_risk_count",   "—"))],
        ["Medium Risk Users",      str(summary.get("medium_risk_count", "—"))],
        ["Low Risk Users",         str(summary.get("low_risk_count",    "—"))],
        ["Average Churn Score",    str(summary.get("avg_churn_score",   "—"))],
        ["Max Churn Score",        str(summary.get("max_churn_score",   "—"))],
        ["Churn Window (days)",    str(val_report.get("churn_window_days", "30"))],
    ]
    summary_table = Table(summary_data, colWidths=[90 * mm, 80 * mm])
    summary_table.setStyle(_table_style())
    story.append(summary_table)
    story.append(Spacer(1, 12))

    # ── Churn Distribution ────────────────────────────────────────────────────
    story.append(Paragraph("2. Risk Distribution", heading_style))

    total = max(n_users, 1)
    high   = summary.get("high_risk_count",   0)
    medium = summary.get("medium_risk_count", 0)
    low    = summary.get("low_risk_count",    0)

    dist_data = [
        ["Risk Level", "User Count", "Percentage"],
        ["🔴 HIGH",   str(high),   f"{high   / total * 100:.1f}%"],
        ["🟡 MEDIUM", str(medium), f"{medium / total * 100:.1f}%"],
        ["🟢 LOW",    str(low),    f"{low    / total * 100:.1f}%"],
    ]
    dist_table = Table(dist_data, colWidths=[60 * mm, 60 * mm, 50 * mm])
    dist_table.setStyle(_table_style("#e94560"))
    story.append(dist_table)
    story.append(Spacer(1, 12))

    # ── Model Validation ──────────────────────────────────────────────────────
    story.append(Paragraph("3. Model Validation Metrics", heading_style))

    gate_color = "#27ae60" if gate_pass else "#e74c3c"
    gate_label = "✅ PASSED" if gate_pass else "❌ FAILED"
    story.append(Paragraph(
        f"Quality Gate: <font color='{gate_color}'><b>{gate_label}</b></font>",
        body,
    ))
    story.append(Spacer(1, 4))

    metrics_data = [
        ["Metric", "Value"],
        ["ROC-AUC",   str(eval_m.get("roc_auc",   "—"))],
        ["Accuracy",  str(eval_m.get("accuracy",  "—"))],
        ["Precision", str(eval_m.get("precision", "—"))],
        ["Recall",    str(eval_m.get("recall",    "—"))],
    ]

    # Confusion matrix rows
    cm = eval_m.get("confusion_matrix")
    if cm and len(cm) == 2:
        metrics_data += [
            ["True Negatives",  str(cm[0][0])],
            ["False Positives", str(cm[0][1])],
            ["False Negatives", str(cm[1][0])],
            ["True Positives",  str(cm[1][1])],
        ]

    metrics_table = Table(metrics_data, colWidths=[90 * mm, 80 * mm])
    metrics_table.setStyle(_table_style("#0f3460"))
    story.append(metrics_table)
    story.append(Spacer(1, 8))

    # Gate notes
    for note in gate_notes:
        story.append(Paragraph(f"• {note}", body))

    story.append(Spacer(1, 12))

    # ── High-Risk Users ───────────────────────────────────────────────────────
    story.append(Paragraph("4. High-Risk Users (Top 20)", heading_style))

    if not scored_df.empty:
        high_df = (
            scored_df[scored_df["risk_level"] == "HIGH"]
            .sort_values("churn_score", ascending=False)
            .head(20)
        )
        if not high_df.empty:
            hr_data = [["User ID", "Churn Score", "Risk Level"]] + [
                [str(row["user_id"]), f"{row['churn_score']:.4f}", row["risk_level"]]
                for _, row in high_df.iterrows()
            ]
            hr_table = Table(hr_data, colWidths=[80 * mm, 50 * mm, 40 * mm])
            hr_table.setStyle(_table_style("#e94560"))
            story.append(hr_table)
        else:
            story.append(Paragraph("No high-risk users identified.", body))
    else:
        story.append(Paragraph("Scored DataFrame not available.", body))

    story.append(Spacer(1, 12))

    # ── Trend Summary ─────────────────────────────────────────────────────────
    story.append(Paragraph("5. Trend Summary", heading_style))
    story.append(Paragraph(
        f"The model identified <b>{high}</b> high-risk users out of "
        f"<b>{n_users}</b> total users ({high/total*100:.1f}% churn rate). "
        f"The average predicted churn score is "
        f"<b>{summary.get('avg_churn_score', 0):.4f}</b>.",
        body,
    ))

    # ── Build PDF ─────────────────────────────────────────────────────────────
    doc.build(story)
    logger.info("PDF report generated", extra={"path": str(pdf_path)})
    return pdf_path


def build_dashboard_json(context: dict[str, Any]) -> dict[str, Any]:
    """
    Build a frontend-ready JSON payload summarising the pipeline output.

    This is the optional dashboard data source — can be served directly
    from a FastAPI endpoint or saved to disk for a Streamlit/React frontend.

    Args:
        context: Completed pipeline context.

    Returns:
        Serialisable dict suitable for ``JSONResponse``.
    """
    import pandas as pd

    scored_df: pd.DataFrame = context.get("scored_df", pd.DataFrame())
    summary                 = context.get("summary_stats", {})
    val_report              = context.get("validation_report", {})

    high_risk = []
    if not scored_df.empty:
        high_risk = (
            scored_df[scored_df["risk_level"] == "HIGH"]
            .sort_values("churn_score", ascending=False)
            .head(50)
            .to_dict(orient="records")
        )

    # ── Trend Calculation ─────────────────────────────────────────────────────
    trends = []
    ts_df: pd.DataFrame = context.get("ts_df", pd.DataFrame())
    if not ts_df.empty and "month" in ts_df.columns:
        # Map month numbers to short names for frontend
        month_map = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        # Group by month and count transactions as 'activity'
        trend_agg = ts_df.groupby("month").size().reset_index(name="activity")
        # Ensure month is numeric for sorting before mapping to names
        trend_agg = trend_agg.sort_values("month")
        trend_agg["month"] = trend_agg["month"].map(month_map)
        trends = trend_agg.to_dict(orient="records")

    return {
        "summary":          summary,
        "validation":       val_report.get("evaluation_metrics", {}),
        "quality_gate": {
            "passed": context.get("quality_gate_pass", False),
            "notes":  context.get("quality_gate_notes", []),
        },
        "high_risk_users":  high_risk,
        "trends":           trends,
        "run_id":           context.get("run_id"),
        "generated_at":     datetime.now(tz=timezone.utc).isoformat(),
    }
