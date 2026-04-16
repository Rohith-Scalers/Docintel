"""Vector storage adapter supporting pgvector and Qdrant backends."""

from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from src.config import DatabaseConfig, EmbeddingConfig, VisualEmbeddingConfig
from src.models import DocumentChunk, RegionType

logger = logging.getLogger(__name__)

_VECTOR_PAYLOAD_FIELDS = (
    "chunk_id",
    "document_id",
    "content_type",
    "cropped_image_path",
    "overview",
    "page_number",
    "confidence_score",
)


class VectorStore:
    """Stores chunk embeddings in either pgvector (Postgres) or Qdrant.

    Backend is selected by DatabaseConfig.vector_backend.

    Embedding source per chunk type:
      FIGURE with VisualEmbeddingConfig.enabled: send base64 crop image to visual endpoint
      All others (and FIGURE fallback): embed overview text (or image_description if richer)

    Vector payload stored alongside embedding:
      {chunk_id, document_id, content_type, cropped_image_path, overview, page_number, confidence_score}
    """

    def __init__(
        self,
        config: DatabaseConfig,
        embedding_config: EmbeddingConfig,
        visual_config: VisualEmbeddingConfig,
        pool: Any | None = None,
    ) -> None:
        self._config = config
        self._embedding_config = embedding_config
        self._visual_config = visual_config
        self._pool = pool
        self._qdrant_client: Any | None = None

    @classmethod
    def from_config(
        cls,
        config: DatabaseConfig,
        embedding_config: EmbeddingConfig,
        visual_config: VisualEmbeddingConfig,
        pool: Any | None = None,
    ) -> "VectorStore":
        """Construct a VectorStore from configuration objects.

        pool: asyncpg connection pool; required when vector_backend is 'pgvector'.

        Returns:
            Configured VectorStore instance.
        """
        return cls(config, embedding_config, visual_config, pool)

    async def _get_qdrant_client(self) -> Any:
        """Return the Qdrant async client, creating it on first call.

        Creates the target collection with cosine distance if it does not exist.

        Returns:
            AsyncQdrantClient instance.
        """
        if self._qdrant_client is not None:
            return self._qdrant_client

        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams

        client = AsyncQdrantClient(
            host=self._config.qdrant_host,
            port=self._config.qdrant_port,
        )
        dims = self._embedding_config.dimensions
        collection = self._config.qdrant_collection

        existing = await client.get_collections()
        names = {c.name for c in existing.collections}
        if collection not in names:
            await client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
            )

        self._qdrant_client = client
        return client

    async def embed_chunk(self, chunk: DocumentChunk) -> list[float]:
        """Compute the embedding vector for a chunk.

        For FIGURE chunks with visual embedding enabled, sends the base64-encoded
        cropped image to the visual embedding endpoint. All other chunk types
        (and FIGURE fallback when visual is disabled or no image path) use the
        text embedding endpoint with the overview or image_description field.

        Returns:
            List of floats representing the embedding vector.
        """
        use_visual = (
            chunk.content_type == RegionType.FIGURE
            and self._visual_config.enabled
            and self._visual_config.endpoint
            and chunk.cropped_image_path
        )

        async with httpx.AsyncClient(timeout=60.0) as client:
            if use_visual:
                with open(chunk.cropped_image_path, "rb") as fh:
                    image_b64 = base64.b64encode(fh.read()).decode("utf-8")
                payload = {
                    "model": self._visual_config.model,
                    "input": image_b64,
                    "encoding_format": "float",
                }
                headers = {}
                if self._visual_config.api_key:
                    headers["Authorization"] = f"Bearer {self._visual_config.api_key}"
                response = await client.post(
                    self._visual_config.endpoint,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()["data"][0]["embedding"]

            text = chunk.overview
            if chunk.content_type == RegionType.FIGURE and chunk.image_description:
                text = chunk.image_description or chunk.overview

            payload = {
                "model": self._embedding_config.model,
                "input": text,
                "encoding_format": "float",
            }
            headers = {}
            if self._embedding_config.api_key:
                headers["Authorization"] = f"Bearer {self._embedding_config.api_key}"
            response = await client.post(
                self._embedding_config.endpoint,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]

    async def upsert(self, chunk: DocumentChunk) -> None:
        """Embed the chunk and store its vector in the configured backend.

        Returns:
            None
        """
        vector = await self.embed_chunk(chunk)
        payload = {field: getattr(chunk, field, None) for field in _VECTOR_PAYLOAD_FIELDS}
        payload["content_type"] = chunk.content_type.value

        if self._config.vector_backend == "qdrant":
            await self._upsert_qdrant(chunk.chunk_id, vector, payload)
        else:
            await self._upsert_pgvector(chunk.chunk_id, vector)

    async def _upsert_qdrant(
        self, chunk_id: str, vector: list[float], payload: dict
    ) -> None:
        """Insert or update a vector point in Qdrant.

        Returns:
            None
        """
        from qdrant_client.models import PointStruct

        client = await self._get_qdrant_client()
        point = PointStruct(id=_chunk_id_to_uint64(chunk_id), vector=vector, payload=payload)
        await client.upsert(
            collection_name=self._config.qdrant_collection,
            points=[point],
        )

    async def _upsert_pgvector(self, chunk_id: str, vector: list[float]) -> None:
        """Update the embedding column of an existing chunk row in PostgreSQL.

        Returns:
            None
        """
        if self._pool is None:
            raise RuntimeError("asyncpg pool is required for pgvector backend")

        vector_str = "[" + ",".join(str(v) for v in vector) + "]"
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE chunks SET embedding = $1::vector WHERE chunk_id = $2",
                vector_str,
                chunk_id,
            )


def _chunk_id_to_uint64(chunk_id: str) -> int:
    """Derive a deterministic uint64 Qdrant point ID from a UUID string.

    Takes the first 16 hex characters of the UUID (stripping dashes) and
    interprets them as a big-endian unsigned 64-bit integer.

    Returns:
        Integer in [0, 2^64) suitable as a Qdrant point ID.
    """
    hex_str = chunk_id.replace("-", "")[:16]
    return int(hex_str, 16)
