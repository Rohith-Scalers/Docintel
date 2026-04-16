from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Generator

import structlog
from prometheus_client import Counter, Histogram, start_http_server

from src.config import ObservabilityConfig

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

DOCS_PROCESSED = Counter(
    "docintel_documents_processed_total",
    "Total documents processed by final status.",
    ["status"],
)

VLM_LATENCY = Histogram(
    "docintel_vlm_latency_seconds",
    "VLM call latency in seconds per region type.",
    ["region_type"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
)

WRITE_FAILURES = Counter(
    "docintel_storage_write_failures_total",
    "Storage write failures per adapter.",
    ["adapter"],
)

CONFIDENCE_HIST = Histogram(
    "docintel_chunk_confidence",
    "Distribution of per-chunk confidence scores.",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

CORRECTIONS = Counter(
    "docintel_correction_passes_total",
    "VLM correction passes triggered per region type and pass number.",
    ["region_type", "pass_number"],
)

LAYOUT_REGIONS = Counter(
    "docintel_layout_regions_total",
    "Detected layout regions per type.",
    ["region_type"],
)

PIPELINE_LATENCY = Histogram(
    "docintel_pipeline_latency_seconds",
    "End-to-end document pipeline latency in seconds.",
    buckets=[5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0],
)

DEDUP_HITS = Counter(
    "docintel_dedup_hits_total",
    "Documents skipped due to content hash deduplication.",
)

# ---------------------------------------------------------------------------
# OpenTelemetry tracer (optional — no-op if otel_endpoint is not configured)
# ---------------------------------------------------------------------------

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


def _setup_tracing(config: ObservabilityConfig) -> None:
    """Initialise OpenTelemetry tracing if an OTLP endpoint is configured.

    Returns:
        None
    """
    if not _OTEL_AVAILABLE or not config.otel_endpoint:
        return

    resource = Resource.create({"service.name": config.service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=config.otel_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


def get_tracer(name: str = "docintel"):
    """Return an OpenTelemetry tracer. Returns a no-op tracer if OTEL is unavailable.

    Returns:
        opentelemetry.trace.Tracer: active tracer instance.
    """
    if _OTEL_AVAILABLE:
        return trace.get_tracer(name)

    class _NoOpTracer:
        @contextmanager
        def start_as_current_span(self, name: str, **_kwargs):
            yield _NoOpSpan()

    class _NoOpSpan:
        def set_attribute(self, *_args, **_kwargs):
            pass

    return _NoOpTracer()


# ---------------------------------------------------------------------------
# structlog
# ---------------------------------------------------------------------------


def _setup_logging(config: ObservabilityConfig) -> None:
    """Configure structlog with JSON rendering and the specified log level.

    Returns:
        None
    """
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", level=level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def setup(config: ObservabilityConfig) -> None:
    """Initialise all observability layers: logging, metrics server, tracing.

    Must be called once at application startup (Celery worker init or pipeline
    entry point). Idempotent — safe to call multiple times.

    Returns:
        None
    """
    _setup_logging(config)
    try:
        start_http_server(config.prometheus_port)
    except OSError:
        pass
    _setup_tracing(config)


def get_logger(name: str = "docintel") -> structlog.BoundLogger:
    """Return a structlog bound logger with the given name.

    Returns:
        structlog.BoundLogger: configured logger instance.
    """
    return structlog.get_logger(name)


# ---------------------------------------------------------------------------
# Convenience timing context manager
# ---------------------------------------------------------------------------


@contextmanager
def timed(histogram: Histogram, labels: list[str]) -> Generator[None, None, None]:
    """Context manager that records elapsed seconds into a Prometheus Histogram.

    Usage:
        with timed(VLM_LATENCY, ["table"]):
            response = await vlm_client.extract(region, schema)

    Returns:
        Generator[None, None, None]
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        histogram.labels(*labels).observe(elapsed)
