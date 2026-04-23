"""
Batch scoring pipeline.

Loads all users from a CSV/DB, generates features, predicts churn probability,
and writes results to a timestamped output file.

Run daily via cron:
    0 2 * * * cd /app && python pipelines/batch_scoring.py

Or with arguments:
    python pipelines/batch_scoring.py --input data/users.csv --output data/batch_results/
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Ensure project root is on PYTHONPATH ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app.core.config import settings
from app.core.logger import get_logger
from app.features.builder import build_feature_dataframe
from app.models.loader import load_model
from app.models.predictor import predict_batch

logger = get_logger("batch_scoring")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Churn Batch Scoring Pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default=settings.BATCH_INPUT_PATH,
        help="Path to user activity CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=settings.BATCH_OUTPUT_DIR,
        help="Directory to write scored output CSV",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Number of rows per batch chunk (for memory efficiency)",
    )
    return parser.parse_args()


def run_batch_scoring(
    input_path: str | Path,
    output_dir: str | Path,
    chunk_size: int = 5000,
) -> Path:
    """
    Execute the full batch scoring pipeline.

    Args:
        input_path: Path to user activity CSV.
        output_dir: Directory to write output.
        chunk_size: Rows per processing chunk.

    Returns:
        Path to the written output file.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"churn_scores_{timestamp}.csv"

    # Pre-load model into cache before chunking
    load_model()

    total_rows = 0
    first_chunk = True

    logger.info(
        "Batch scoring started",
        extra={"input": str(input_path), "output": str(output_path), "chunk_size": chunk_size},
    )

    for chunk_idx, chunk_df in enumerate(
        pd.read_csv(input_path, chunksize=chunk_size)
    ):
        logger.info(
            "Processing chunk",
            extra={"chunk": chunk_idx, "rows": len(chunk_df)},
        )

        records = chunk_df.to_dict(orient="records")
        feature_df = build_feature_dataframe(records)
        predictions = predict_batch(feature_df)

        # Build result DataFrame
        result_df = pd.DataFrame({
            "user_id": chunk_df.get("user_id", range(total_rows, total_rows + len(chunk_df))),
            "churn_score": [p["churn_score"] for p in predictions],
            "risk_level":  [p["risk_level"]  for p in predictions],
            "scored_at":   datetime.now(tz=timezone.utc).isoformat(),
        })

        result_df.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        total_rows += len(chunk_df)
        first_chunk = False

    logger.info(
        "Batch scoring completed",
        extra={"total_rows": total_rows, "output": str(output_path)},
    )
    return output_path


def main() -> None:
    args = parse_args()

    try:
        output_path = run_batch_scoring(
            input_path=args.input,
            output_dir=args.output,
            chunk_size=args.chunk_size,
        )
        print(f"\n✅ Batch scoring complete.")
        print(f"   Output: {output_path}\n")

    except Exception as exc:
        logger.error("Batch scoring pipeline failed", extra={"error": str(exc)})
        print(f"\n❌ Batch scoring failed: {exc}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
