"""Optional streaming module placeholder.

This module is intentionally left as a placeholder.
Kafka-based real-time streaming is NOT required to run the system.

To activate in the future:
1. Install: pip install kafka-python
2. Implement KafkaConsumer / KafkaProducer logic here
3. Wire into a separate worker process (not the FastAPI app)

The batch scoring pipeline (pipelines/batch_scoring.py) is the
recommended approach for production churn scoring.
"""

from app.core.logger import get_logger

logger = get_logger(__name__)


def start_kafka_consumer() -> None:  # pragma: no cover
    """
    PLACEHOLDER: Start a Kafka consumer for real-time event ingestion.

    Not implemented — streaming is optional and not required for system operation.
    """
    logger.warning(
        "Kafka streaming consumer is not implemented. "
        "Use the batch scoring pipeline for production workloads."
    )
    raise NotImplementedError(
        "Kafka streaming is a future optional feature. "
        "See module docstring for activation steps."
    )
