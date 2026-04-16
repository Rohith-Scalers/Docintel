from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VLMConfig(BaseSettings):
    """Configuration for the DeepSeek-OCR-2 (OpenAI-compatible) VLM endpoint.

    All values are loaded from environment variables prefixed with VLM_.
    api_key is required and must be set in .env.
    """

    model_config = SettingsConfigDict(env_prefix="VLM_", env_file=".env", extra="ignore")

    endpoint: str = "http://localhost:8000/v1/chat/completions"
    model: str = "deepseek-ocr-2"
    api_key: str = ""
    timeout_s: int = 60
    max_retries: int = 3
    concurrency_limit: int = 8
    overview_max_tokens: int = 256
    extraction_max_tokens: int = 1024
    correction_max_passes: int = 2
    prompt_config_path: str = "configs/vlm.yaml"


class EmbeddingConfig(BaseSettings):
    """Configuration for the text embedding endpoint (OpenAI-compatible).

    dimensions must match the vector index size configured in the vector
    database. For bge-m3 the correct value is 1024.
    """

    model_config = SettingsConfigDict(env_prefix="EMBED_", env_file=".env", extra="ignore")

    endpoint: str = "http://localhost:8001/v1/embeddings"
    model: str = "BAAI/bge-m3"
    api_key: str = ""
    dimensions: int = 1024


class VisualEmbeddingConfig(BaseSettings):
    """Configuration for a multimodal visual embedding endpoint (e.g. ColPali).

    When enabled, FIGURE chunks are embedded by sending the cropped image to
    this endpoint instead of embedding the text overview. Set enabled=true
    and configure endpoint/model to activate. dimensions=128 matches ColPali.
    """

    model_config = SettingsConfigDict(
        env_prefix="VISUAL_EMBED_", env_file=".env", extra="ignore"
    )

    enabled: bool = False
    endpoint: str | None = None
    model: str | None = None
    api_key: str = ""
    dimensions: int = 128


class DatabaseConfig(BaseSettings):
    """Connection strings and storage paths for all database backends.

    Only the adapters with non-None URIs are activated at pipeline startup.
    image_store_path is the local directory where region crop files are saved.
    image_base_url, if set, is prepended to cropped_image_path when building
    HTTP-accessible URLs (e.g. for nginx static serving or S3 pre-signed URLs).
    """

    model_config = SettingsConfigDict(env_prefix="DB_", env_file=".env", extra="ignore")

    mongodb_uri: str | None = None
    postgres_dsn: str | None = None
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None
    vector_backend: Literal["pgvector", "qdrant"] = "pgvector"
    qdrant_host: str | None = None
    qdrant_port: int = 6333
    qdrant_collection: str = "docintel_chunks"
    image_store_path: str = "/data/images"
    image_base_url: str | None = None
    postgres_pool_min: int = 2
    postgres_pool_max: int = 10


class QueueConfig(BaseSettings):
    """Celery + Redis broker configuration.

    broker_url and result_backend both point to Redis by default. The
    celery-worker container uses these values to connect at startup.
    task_time_limit is a hard per-task ceiling in seconds.
    """

    model_config = SettingsConfigDict(env_prefix="QUEUE_", env_file=".env", extra="ignore")

    broker_url: str = "redis://redis:6379/0"
    result_backend: str = "redis://redis:6379/1"
    task_time_limit: int = 600
    task_soft_time_limit: int = 540


class ObservabilityConfig(BaseSettings):
    """Logging, metrics, and tracing configuration.

    prometheus_port: HTTP port for /metrics endpoint (scraped by Prometheus).
    otel_endpoint: OTLP gRPC endpoint for distributed tracing. Leave empty
        to disable tracing export.
    log_level: structlog minimum level (DEBUG / INFO / WARNING / ERROR).
    """

    model_config = SettingsConfigDict(env_prefix="OBS_", env_file=".env", extra="ignore")

    prometheus_port: int = 9090
    otel_endpoint: str | None = None
    log_level: str = "INFO"
    service_name: str = "docintel"


class LayoutConfig(BaseSettings):
    """Configuration for DocLayout-YOLO layout detection.

    model_path: path to the .pt weights file. Download from:
        https://huggingface.co/opendatalab/DocLayout-YOLO
    conf_threshold: minimum detection confidence to accept a region.
    iou_threshold: NMS IoU threshold.
    device: 'cpu', 'cuda', or 'cuda:0' etc.
    """

    model_config = SettingsConfigDict(
        env_prefix="LAYOUT_", env_file=".env", extra="ignore"
    )

    model_path: str = "configs/doclayout_yolo.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cpu"
    input_size: int = 1024


class AppConfig(BaseSettings):
    """Aggregated application configuration loaded at startup."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    vlm: VLMConfig = Field(default_factory=VLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    visual_embedding: VisualEmbeddingConfig = Field(
        default_factory=VisualEmbeddingConfig
    )
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    layout: LayoutConfig = Field(default_factory=LayoutConfig)
