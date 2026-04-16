"""Vector similarity retrieval over chunk embeddings."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

import httpx

from src.config import DatabaseConfig, EmbeddingConfig
from src.models import DocumentChunk, RetrievalResult
from src.storage.base import StorageAdapter
from src.storage.postgres import _row_to_chunk_dict
from src.storage.vector import _chunk_id_to_uint64

logger = logging.getLogger(__name__)


class VectorRetrieval:
    """Semantic similarity retrieval over chunk embeddings.

    Embeds query_text using the configured text embedding endpoint, then
    performs approximate nearest-neighbour search. Supports hybrid search
    (ANN + payload metadata filter) on Qdrant; pgvector uses ORDER BY cosine.
    """

    def __init__(
        self,
        config: DatabaseConfig,
        embedding_config: EmbeddingConfig,
        adapter: StorageAdapter,
        pool: Any | None = None,
    ) -> None:
        """Initialise with database config, embedding config, storage adapter, and pool.

        adapter: storage adapter instance for _adapt_chunk on pgvector results.
        pool: asyncpg pool; required when vector_backend is 'pgvector'.
        """
        self._config = config
        self._embedding_config = embedding_config
        self._adapter = adapter
        self._pool = pool
        self._qdrant_client: Any | None = None

    async def _embed_query(self, query_text: str) -> list[float]:
        """Embed a query string using the configured text embedding endpoint.

        Returns:
            List of floats representing the query embedding.
        """
        payload = {
            "model": self._embedding_config.model,
            "input": query_text,
            "encoding_format": "float",
        }
        headers: dict[str, str] = {}
        if self._embedding_config.api_key:
            headers["Authorization"] = f"Bearer {self._embedding_config.api_key}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self._embedding_config.endpoint,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]

    async def _get_qdrant_client(self) -> Any:
        """Return the Qdrant async client, creating it on first call.

        Returns:
            AsyncQdrantClient instance.
        """
        if self._qdrant_client is not None:
            return self._qdrant_client
        from qdrant_client import AsyncQdrantClient

        self._qdrant_client = AsyncQdrantClient(
            host=self._config.qdrant_host,
            port=self._config.qdrant_port,
        )
        return self._qdrant_client

    async def search(
        self,
        query_text: str,
        top_k: int = 20,
        metadata_filter: dict | None = None,
        content_types: list[str] | None = None,
        document_id: str | None = None,
        min_score: float | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> RetrievalResult:
        """Embed query_text and search for semantically similar chunks.

        metadata_filter: Qdrant filter dict or pgvector WHERE clause dict.
        min_score: minimum cosine similarity threshold (0.0–1.0).
        cursor: Qdrant next_page_offset or pgvector OFFSET integer (base64 encoded).

        Returns:
            RetrievalResult with up to limit chunks sorted by similarity score,
            with metadata["similarity_score"] set on each chunk.
        """
        vector = await self._embed_query(query_text)

        if self._config.vector_backend == "qdrant":
            return await self._search_qdrant(
                vector,
                top_k,
                metadata_filter,
                content_types,
                document_id,
                min_score,
                limit,
                cursor,
            )
        return await self._search_pgvector(
            vector,
            top_k,
            metadata_filter,
            content_types,
            document_id,
            min_score,
            limit,
            cursor,
        )

    async def _search_qdrant(
        self,
        vector: list[float],
        top_k: int,
        metadata_filter: dict | None,
        content_types: list[str] | None,
        document_id: str | None,
        min_score: float | None,
        limit: int,
        cursor: str | None,
    ) -> RetrievalResult:
        """Execute ANN search in Qdrant with optional payload filter.

        Returns:
            RetrievalResult with scored chunks.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        client = await self._get_qdrant_client()

        must_conditions: list[Any] = []
        if metadata_filter:
            for key, value in metadata_filter.items():
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        if document_id:
            must_conditions.append(
                FieldCondition(key="document_id", match=MatchValue(value=document_id))
            )
        if content_types:
            must_conditions.append(
                FieldCondition(key="content_type", match=MatchAny(any=content_types))
            )

        qdrant_filter = Filter(must=must_conditions) if must_conditions else None

        offset: int | None = None
        if cursor:
            raw = base64.b64decode(cursor).decode("utf-8")
            offset = int(raw) if raw.isdigit() else None

        results = await client.search(
            collection_name=self._config.qdrant_collection,
            query_vector=vector,
            query_filter=qdrant_filter,
            limit=limit,
            offset=offset,
            score_threshold=min_score,
            with_payload=True,
        )

        chunks: list[DocumentChunk] = []
        for hit in results:
            payload = hit.payload or {}
            chunk_id = payload.get("chunk_id", "")
            if not chunk_id:
                continue
            chunk_dict = {
                "chunk_id": chunk_id,
                "document_id": payload.get("document_id", ""),
                "page_number": payload.get("page_number", 0),
                "chunk_index": payload.get("chunk_index", 0),
                "reading_order_index": payload.get("reading_order_index", 0),
                "content_type": payload.get("content_type", "text"),
                "overview": payload.get("overview", ""),
                "cropped_image_path": payload.get("cropped_image_path", ""),
                "confidence_score": payload.get("confidence_score", 0.0),
                "schema_version": payload.get("schema_version", 2),
                "metadata": {"similarity_score": hit.score},
            }
            chunk = self._adapter._adapt_chunk(chunk_dict)
            chunk.metadata["similarity_score"] = hit.score
            chunks.append(chunk)

        next_cursor: str | None = None
        if len(results) == limit:
            new_offset = (offset or 0) + limit
            next_cursor = base64.b64encode(str(new_offset).encode("utf-8")).decode("utf-8")

        return RetrievalResult(chunks=chunks, total=len(chunks), next_cursor=next_cursor)

    async def _search_pgvector(
        self,
        vector: list[float],
        top_k: int,
        metadata_filter: dict | None,
        content_types: list[str] | None,
        document_id: str | None,
        min_score: float | None,
        limit: int,
        cursor: str | None,
    ) -> RetrievalResult:
        """Execute cosine similarity search using pgvector ORDER BY.

        Returns:
            RetrievalResult with scored chunks.
        """
        if self._pool is None:
            raise RuntimeError("asyncpg pool required for pgvector backend")

        conditions: list[str] = ["embedding IS NOT NULL"]
        params: list[Any] = []
        param_idx = 1

        vector_str = "[" + ",".join(str(v) for v in vector) + "]"
        params.append(vector_str)
        param_idx += 1

        if document_id:
            conditions.append(f"document_id = ${param_idx}")
            params.append(document_id)
            param_idx += 1

        if content_types:
            placeholders = ", ".join(f"${param_idx + i}" for i in range(len(content_types)))
            conditions.append(f"content_type IN ({placeholders})")
            params.extend(content_types)
            param_idx += len(content_types)

        if min_score is not None:
            conditions.append(f"1 - (embedding <=> $1::vector) >= ${param_idx}")
            params.append(min_score)
            param_idx += 1

        offset = 0
        if cursor:
            offset = int(base64.b64decode(cursor).decode("utf-8"))

        where_clause = "WHERE " + " AND ".join(conditions)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT *, 1 - (embedding <=> $1::vector) AS similarity_score
                FROM chunks
                {where_clause}
                ORDER BY embedding <=> $1::vector
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
                """,
                *params,
                limit,
                offset,
            )

        chunks: list[DocumentChunk] = []
        for row in rows:
            score = row["similarity_score"]
            raw = _row_to_chunk_dict(row)
            chunk = self._adapter._adapt_chunk(raw)
            chunk.metadata["similarity_score"] = float(score)
            chunks.append(chunk)

        next_cursor: str | None = None
        if len(chunks) == limit:
            next_cursor = base64.b64encode(
                str(offset + limit).encode("utf-8")
            ).decode("utf-8")

        return RetrievalResult(chunks=chunks, total=len(chunks), next_cursor=next_cursor)
