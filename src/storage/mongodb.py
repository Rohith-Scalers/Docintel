"""Async MongoDB storage adapter using Motor."""

from __future__ import annotations

import logging
from typing import Any

import motor.motor_asyncio

from src.config import DatabaseConfig
from src.models import CURRENT_SCHEMA_VERSION, Document, DocumentChunk, RawRegion
from src.storage.base import StorageAdapter

logger = logging.getLogger(__name__)

_DB_NAME = "docintel"


class MongoAdapter(StorageAdapter):
    """MongoDB storage adapter backed by Motor (async).

    Collections used:
      - documents: top-level document metadata (no embedded chunks/regions)
      - chunks: each DocumentChunk stored as a flat document
      - regions: each RawRegion stored as a flat document

    Unique indexes on content_hash and document_id prevent duplicate ingestion.
    """

    def __init__(self, client: motor.motor_asyncio.AsyncIOMotorClient) -> None:
        self._client = client
        db = client[_DB_NAME]
        self._documents = db["documents"]
        self._chunks = db["chunks"]
        self._regions = db["regions"]
        self._indexes_created = False

    @classmethod
    def from_config(cls, config: DatabaseConfig) -> "MongoAdapter":
        """Construct a MongoAdapter from a DatabaseConfig.

        Returns:
            Configured MongoAdapter instance.
        """
        client: motor.motor_asyncio.AsyncIOMotorClient = (
            motor.motor_asyncio.AsyncIOMotorClient(config.mongodb_uri)
        )
        return cls(client)

    async def _ensure_indexes(self) -> None:
        """Create all required indexes idempotently on first write.

        Returns:
            None
        """
        if self._indexes_created:
            return

        await self._documents.create_index("content_hash", unique=True)
        await self._documents.create_index("document_id", unique=True)

        await self._chunks.create_index("document_id")
        await self._chunks.create_index("chunk_id", unique=True)
        await self._chunks.create_index("content_type")
        await self._chunks.create_index(
            [("overview", "text")], sparse=True
        )

        await self._regions.create_index("document_id")
        await self._regions.create_index("region_id", unique=True)

        self._indexes_created = True

    async def write_document(self, document: Document) -> None:
        """Persist a Document and all its chunks and regions to MongoDB.

        The document metadata is stored without embedded chunks or regions.
        Each chunk and region is written individually to its own collection.

        Returns:
            None
        """
        await self._ensure_indexes()

        doc_dict = document.model_dump(exclude={"chunks", "regions"})
        await self._documents.update_one(
            {"document_id": document.document_id},
            {"$set": doc_dict},
            upsert=True,
        )

        if document.chunks:
            chunk_ops = [
                {
                    "filter": {"chunk_id": chunk.chunk_id},
                    "update": {"$set": chunk.model_dump()},
                    "upsert": True,
                }
                for chunk in document.chunks
            ]
            for op in chunk_ops:
                await self._chunks.update_one(
                    op["filter"], op["update"], upsert=op["upsert"]
                )

        if document.regions:
            for region in document.regions:
                await self._regions.update_one(
                    {"region_id": region.region_id},
                    {"$set": region.model_dump()},
                    upsert=True,
                )

    async def find_by_hash(self, content_hash: str) -> Document | None:
        """Retrieve a full Document by its sha256 content hash.

        Returns:
            Document with all chunks and regions populated, or None if not found.
        """
        doc_raw: dict[str, Any] | None = await self._documents.find_one(
            {"content_hash": content_hash}, {"_id": 0}
        )
        if doc_raw is None:
            return None

        document_id = doc_raw["document_id"]

        chunk_cursor = self._chunks.find({"document_id": document_id}, {"_id": 0})
        chunks: list[DocumentChunk] = []
        async for raw in chunk_cursor:
            chunks.append(self._adapt_chunk(raw))

        region_cursor = self._regions.find({"document_id": document_id}, {"_id": 0})
        regions: list[RawRegion] = []
        async for raw in region_cursor:
            regions.append(RawRegion(**raw))

        return Document(**doc_raw, chunks=chunks, regions=regions)

    async def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """Retrieve a single chunk by its UUID.

        Returns:
            DocumentChunk if found, None otherwise.
        """
        raw: dict[str, Any] | None = await self._chunks.find_one(
            {"chunk_id": chunk_id}, {"_id": 0}
        )
        if raw is None:
            return None
        return self._adapt_chunk(raw)

    async def close(self) -> None:
        """Close the Motor client connection.

        Returns:
            None
        """
        self._client.close()
