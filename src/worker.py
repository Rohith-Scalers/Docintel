from __future__ import annotations

import asyncio

from celery import Celery

from src.config import QueueConfig
from src.models import ExtractionSchema
from src.observability import get_logger, setup

logger = get_logger(__name__)

_queue_cfg = QueueConfig()

app = Celery(
    "docintel",
    broker=_queue_cfg.broker_url,
    backend=_queue_cfg.result_backend,
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_time_limit=_queue_cfg.task_time_limit,
    task_soft_time_limit=_queue_cfg.task_soft_time_limit,
    worker_prefetch_multiplier=1,
)


@app.on_after_configure.connect
def _setup_observability(sender, **_kwargs) -> None:
    """Initialise observability stack when the Celery worker starts."""
    from src.config import ObservabilityConfig
    setup(ObservabilityConfig())


@app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    name="docintel.process_document",
)
def process_document_task(
    self,
    source_path: str,
    extraction_schema_dict: dict | None = None,
    targets: list[str] | None = None,
) -> str:
    """Submit a document for extraction and multi-database storage.

    This task is the sole entry point for document processing. It wraps
    the async ExtractionPipeline.run() in asyncio.run() for Celery compatibility.

    Args:
        source_path: absolute path to the document file on the worker filesystem.
        extraction_schema_dict: serialised ExtractionSchema dict, or None.
        targets: list of storage adapter names to write to. None = all adapters.

    Returns:
        str: document_id UUID of the processed Document.
    """
    from src.extraction.pipeline import ExtractionPipeline

    schema: ExtractionSchema | None = None
    if extraction_schema_dict:
        schema = ExtractionSchema(**extraction_schema_dict)

    try:
        pipeline = ExtractionPipeline.from_env()
        doc = asyncio.run(pipeline.run(source_path, schema, targets))
        logger.info(
            "task_complete",
            document_id=doc.document_id,
            status=doc.processing_status,
        )
        return doc.document_id
    except Exception as exc:
        logger.error("task_failed", source_path=source_path, error=str(exc))
        raise self.retry(exc=exc)


def submit(
    source_path: str,
    extraction_schema: ExtractionSchema | None = None,
    targets: list[str] | None = None,
) -> str:
    """Submit a document processing task and block until complete.

    Convenience wrapper for synchronous callers. For fire-and-forget use
    process_document_task.delay() directly.

    Args:
        source_path: absolute path to the document.
        extraction_schema: optional field extraction schema.
        targets: optional list of target adapter names.

    Returns:
        str: document_id of the processed document.
    """
    schema_dict = extraction_schema.model_dump() if extraction_schema else None
    result = process_document_task.delay(source_path, schema_dict, targets)
    return result.get(timeout=_queue_cfg.task_time_limit)
