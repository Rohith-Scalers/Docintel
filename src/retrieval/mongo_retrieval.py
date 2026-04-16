"""MongoDB retrieval interface with keyset pagination."""

from __future__ import annotations

import base64
import logging
from typing import Any

from src.models import DocumentChunk, RetrievalResult
from src.storage.base import StorageAdapter

logger = logging.getLogger(__name__)


class MongoRetrieval:
    """Retrieval interface for MongoDB chunks collection.

    Uses keyset pagination on _id for O(1) page traversal regardless
    of collection size.
    """

    def __init__(self, adapter: StorageAdapter, collection: Any) -> None:
        """Initialise with a storage adapter (for _adapt_chunk) and raw Motor collection.

        adapter: MongoAdapter instance; used for schema migration on read.
        collection: motor AsyncIOMotorCollection for the chunks collection.
        """
        self._adapter = adapter
        self._collection = collection

    async def query(
        self,
        filters: dict | None = None,
        full_text: str | None = None,
        content_types: list[str] | None = None,
        document_id: str | None = None,
        min_confidence: float | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> RetrievalResult:
        """Query chunks with optional filters and full-text search.

        filters: arbitrary MongoDB filter dict merged with built-in filters.
        full_text: if provided, adds $text search on overview field.
        cursor: base64-encoded last _id from previous page.

        Returns:
            RetrievalResult with up to limit chunks, total count, next_cursor.
        """
        query_filter: dict[str, Any] = {}

        if filters:
            query_filter.update(filters)

        if full_text:
            query_filter["$text"] = {"$search": full_text}

        if content_types:
            query_filter["content_type"] = {"$in": content_types}

        if document_id:
            query_filter["document_id"] = document_id

        if min_confidence is not None:
            query_filter["confidence_score"] = {"$gte": min_confidence}

        if cursor:
            from bson import ObjectId
            last_id = ObjectId(base64.b64decode(cursor).decode("utf-8"))
            query_filter["_id"] = {"$gt": last_id}

        total = await self._collection.count_documents(
            {k: v for k, v in query_filter.items() if k != "_id"}
        )

        raw_cursor = self._collection.find(query_filter, {"_id": 1}).limit(limit + 1)

        docs: list[dict] = []
        last_oid = None
        async for doc in raw_cursor:
            last_oid = doc["_id"]
            docs.append(doc)

        has_more = len(docs) > limit
        if has_more:
            docs = docs[:limit]
            last_oid = docs[-1]["_id"]

        chunk_ids = [d["_id"] for d in docs]
        chunks: list[DocumentChunk] = []

        full_cursor = self._collection.find(
            {"_id": {"$in": chunk_ids}}, {"_id": 0}
        )
        id_to_chunk: dict[str, DocumentChunk] = {}
        async for raw in full_cursor:
            chunk = self._adapter._adapt_chunk(raw)
            id_to_chunk[chunk.chunk_id] = chunk

        for doc in docs:
            raw_doc = await self._collection.find_one(
                {"_id": doc["_id"]}, {"_id": 0}
            )
            if raw_doc:
                chunks.append(self._adapter._adapt_chunk(raw_doc))

        next_cursor: str | None = None
        if has_more and last_oid is not None:
            next_cursor = base64.b64encode(str(last_oid).encode("utf-8")).decode("utf-8")

        return RetrievalResult(chunks=chunks, total=total, next_cursor=next_cursor)
