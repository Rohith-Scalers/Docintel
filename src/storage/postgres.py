"""Async PostgreSQL storage adapter using asyncpg with optional pgvector."""

from __future__ import annotations

import json
import logging
from typing import Any

import asyncpg

from src.config import DatabaseConfig, EmbeddingConfig
from src.models import CURRENT_SCHEMA_VERSION, Document, DocumentChunk, RawRegion
from src.storage.base import StorageAdapter

logger = logging.getLogger(__name__)

_CREATE_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    document_id      TEXT PRIMARY KEY,
    source_path      TEXT,
    content_hash     TEXT UNIQUE,
    format           TEXT,
    total_pages      INTEGER,
    processing_status TEXT,
    schema_version   INTEGER DEFAULT 2,
    metadata         JSONB DEFAULT '{}'
)
"""

_CREATE_REGIONS = """
CREATE TABLE IF NOT EXISTS regions (
    region_id            TEXT PRIMARY KEY,
    document_id          TEXT REFERENCES documents(document_id),
    page_number          INTEGER,
    region_index         INTEGER,
    region_type          TEXT,
    bbox                 JSONB,
    cropped_image_path   TEXT,
    content_hash         TEXT,
    detector_confidence  FLOAT,
    schema_version       INTEGER DEFAULT 2,
    metadata             JSONB DEFAULT '{}'
)
"""

_CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id             TEXT PRIMARY KEY,
    document_id          TEXT REFERENCES documents(document_id),
    page_number          INTEGER,
    chunk_index          INTEGER,
    reading_order_index  INTEGER,
    content_type         TEXT,
    raw_text             TEXT DEFAULT '',
    overview             TEXT DEFAULT '',
    table_data           JSONB,
    image_description    TEXT,
    formula_latex        TEXT,
    structured_data      JSONB DEFAULT '{}',
    entities             JSONB DEFAULT '[]',
    cropped_image_path   TEXT DEFAULT '',
    confidence_score     FLOAT DEFAULT 0.0,
    correction_applied   BOOLEAN DEFAULT FALSE,
    page_break_context   TEXT,
    caption              TEXT,
    schema_version       INTEGER DEFAULT 2,
    metadata             JSONB DEFAULT '{}'
)
"""

_GIN_STRUCTURED_DATA = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_structured_data "
    "ON chunks USING GIN (structured_data)"
)

_GIN_TABLE_DATA = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_table_data "
    "ON chunks USING GIN (table_data) WHERE table_data IS NOT NULL"
)

_IDX_CHUNKS_DOC = "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id)"
_IDX_REGIONS_DOC = "CREATE INDEX IF NOT EXISTS idx_regions_document_id ON regions (document_id)"


class PostgresAdapter(StorageAdapter):
    """PostgreSQL storage adapter using asyncpg connection pooling.

    pgvector support is detected at runtime. When available, an VECTOR column
    is added to the chunks table for embedding storage. The embedding column
    is never populated by this adapter; VectorStore handles that separately.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        embedding_config: EmbeddingConfig | None = None,
    ) -> None:
        self._pool = pool
        self._embedding_config = embedding_config

    @classmethod
    async def from_config(
        cls,
        config: DatabaseConfig,
        embedding_config: EmbeddingConfig | None = None,
    ) -> "PostgresAdapter":
        """Create a PostgresAdapter from config, opening a connection pool.

        Returns:
            Configured PostgresAdapter with tables created.
        """
        pool = await asyncpg.create_pool(
            dsn=config.postgres_dsn,
            min_size=config.postgres_pool_min,
            max_size=config.postgres_pool_max,
        )
        adapter = cls(pool, embedding_config)
        await adapter.create_tables()
        return adapter

    async def create_tables(self) -> None:
        """Create all required tables and indexes idempotently.

        Attempts to enable pgvector and add an embedding VECTOR column when
        the extension is available. Failures are logged but do not abort.

        Returns:
            None
        """
        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_DOCUMENTS)
            await conn.execute(_CREATE_REGIONS)
            await conn.execute(_CREATE_CHUNKS)
            await conn.execute(_GIN_STRUCTURED_DATA)
            await conn.execute(_GIN_TABLE_DATA)
            await conn.execute(_IDX_CHUNKS_DOC)
            await conn.execute(_IDX_REGIONS_DOC)

            if self._embedding_config is not None:
                dims = self._embedding_config.dimensions
                try:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    await conn.execute(
                        f"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS "
                        f"embedding VECTOR({dims})"
                    )
                except Exception:
                    logger.warning(
                        "pgvector extension not available; embedding column skipped",
                        exc_info=True,
                    )

    async def write_document(self, document: Document) -> None:
        """Persist a Document with all its chunks and regions using bulk insert.

        Uses executemany for batched chunk and region inserts. All writes are
        executed within a single connection for consistent error handling.

        Returns:
            None
        """
        async with self._pool.acquire() as conn:
            doc_dict = document.model_dump(exclude={"chunks", "regions"})
            await conn.execute(
                """
                INSERT INTO documents
                    (document_id, source_path, content_hash, format, total_pages,
                     processing_status, schema_version, metadata)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                ON CONFLICT (document_id) DO UPDATE SET
                    source_path=EXCLUDED.source_path,
                    content_hash=EXCLUDED.content_hash,
                    format=EXCLUDED.format,
                    total_pages=EXCLUDED.total_pages,
                    processing_status=EXCLUDED.processing_status,
                    schema_version=EXCLUDED.schema_version,
                    metadata=EXCLUDED.metadata
                """,
                doc_dict["document_id"],
                doc_dict["source_path"],
                doc_dict["content_hash"],
                doc_dict["format"],
                doc_dict["total_pages"],
                doc_dict["processing_status"],
                doc_dict["schema_version"],
                json.dumps(doc_dict["metadata"]),
            )

            if document.chunks:
                chunk_rows = [
                    _chunk_to_row(c) for c in document.chunks
                ]
                await conn.executemany(
                    """
                    INSERT INTO chunks
                        (chunk_id, document_id, page_number, chunk_index,
                         reading_order_index, content_type, raw_text, overview,
                         table_data, image_description, formula_latex, structured_data,
                         entities, cropped_image_path, confidence_score,
                         correction_applied, page_break_context, caption,
                         schema_version, metadata)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,
                            $16,$17,$18,$19,$20)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        reading_order_index=EXCLUDED.reading_order_index,
                        raw_text=EXCLUDED.raw_text,
                        overview=EXCLUDED.overview,
                        table_data=EXCLUDED.table_data,
                        image_description=EXCLUDED.image_description,
                        formula_latex=EXCLUDED.formula_latex,
                        structured_data=EXCLUDED.structured_data,
                        entities=EXCLUDED.entities,
                        cropped_image_path=EXCLUDED.cropped_image_path,
                        confidence_score=EXCLUDED.confidence_score,
                        correction_applied=EXCLUDED.correction_applied,
                        page_break_context=EXCLUDED.page_break_context,
                        caption=EXCLUDED.caption,
                        metadata=EXCLUDED.metadata
                    """,
                    chunk_rows,
                )

            if document.regions:
                region_rows = [_region_to_row(r) for r in document.regions]
                await conn.executemany(
                    """
                    INSERT INTO regions
                        (region_id, document_id, page_number, region_index,
                         region_type, bbox, cropped_image_path, content_hash,
                         detector_confidence, schema_version, metadata)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                    ON CONFLICT (region_id) DO UPDATE SET
                        detector_confidence=EXCLUDED.detector_confidence,
                        bbox=EXCLUDED.bbox,
                        cropped_image_path=EXCLUDED.cropped_image_path,
                        metadata=EXCLUDED.metadata
                    """,
                    region_rows,
                )

    async def find_by_hash(self, content_hash: str) -> Document | None:
        """Retrieve a full Document by sha256 content hash.

        Returns:
            Document with all chunks and regions, or None if not found.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM documents WHERE content_hash = $1", content_hash
            )
            if row is None:
                return None

            document_id = row["document_id"]
            chunk_rows = await conn.fetch(
                "SELECT * FROM chunks WHERE document_id = $1 "
                "ORDER BY reading_order_index",
                document_id,
            )
            region_rows = await conn.fetch(
                "SELECT * FROM regions WHERE document_id = $1 ORDER BY region_index",
                document_id,
            )

        chunks = [self._adapt_chunk(_row_to_chunk_dict(r)) for r in chunk_rows]
        regions = [RawRegion(**_row_to_region_dict(r)) for r in region_rows]
        return Document(**_row_to_doc_dict(row), chunks=chunks, regions=regions)

    async def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """Retrieve a single chunk by its UUID.

        Returns:
            DocumentChunk if found, None otherwise.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM chunks WHERE chunk_id = $1", chunk_id
            )
        if row is None:
            return None
        return self._adapt_chunk(_row_to_chunk_dict(row))

    async def close(self) -> None:
        """Close the asyncpg connection pool.

        Returns:
            None
        """
        await self._pool.close()


def _chunk_to_row(c: DocumentChunk) -> tuple[Any, ...]:
    """Serialize a DocumentChunk to a positional tuple for asyncpg executemany.

    Returns:
        Tuple of 20 values matching the chunks INSERT parameter order.
    """
    return (
        c.chunk_id,
        c.document_id,
        c.page_number,
        c.chunk_index,
        c.reading_order_index,
        c.content_type.value,
        c.raw_text,
        c.overview,
        json.dumps(c.table_data.model_dump()) if c.table_data else None,
        c.image_description,
        c.formula_latex,
        json.dumps(c.structured_data),
        json.dumps(c.entities),
        c.cropped_image_path,
        c.confidence_score,
        c.correction_applied,
        c.page_break_context,
        c.caption,
        c.schema_version,
        json.dumps(c.metadata),
    )


def _region_to_row(r: RawRegion) -> tuple[Any, ...]:
    """Serialize a RawRegion to a positional tuple for asyncpg executemany.

    Returns:
        Tuple of 11 values matching the regions INSERT parameter order.
    """
    return (
        r.region_id,
        r.document_id,
        r.page_number,
        r.region_index,
        r.region_type.value,
        json.dumps(r.bbox.model_dump()),
        r.cropped_image_path,
        r.content_hash,
        r.detector_confidence,
        r.schema_version,
        json.dumps(r.metadata),
    )


def _row_to_doc_dict(row: asyncpg.Record) -> dict:
    """Convert an asyncpg documents row to a dict suitable for Document(**...).

    Returns:
        Dict with JSON fields decoded.
    """
    d = dict(row)
    d["metadata"] = json.loads(d["metadata"]) if isinstance(d["metadata"], str) else d["metadata"] or {}
    return d


def _row_to_chunk_dict(row: asyncpg.Record) -> dict:
    """Convert an asyncpg chunks row to a dict suitable for _adapt_chunk().

    Returns:
        Dict with JSONB fields decoded and content_type as string.
    """
    d = dict(row)
    for field in ("structured_data", "entities", "metadata"):
        if isinstance(d.get(field), str):
            d[field] = json.loads(d[field])
        elif d.get(field) is None:
            d[field] = {} if field != "entities" else []
    if isinstance(d.get("table_data"), str):
        d["table_data"] = json.loads(d["table_data"])
    return d


def _row_to_region_dict(row: asyncpg.Record) -> dict:
    """Convert an asyncpg regions row to a dict suitable for RawRegion(**...).

    Returns:
        Dict with JSONB fields decoded.
    """
    d = dict(row)
    if isinstance(d.get("bbox"), str):
        d["bbox"] = json.loads(d["bbox"])
    if isinstance(d.get("metadata"), str):
        d["metadata"] = json.loads(d["metadata"])
    elif d.get("metadata") is None:
        d["metadata"] = {}
    return d
