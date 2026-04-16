"""PostgreSQL retrieval interface with keyset pagination."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

import asyncpg

from src.models import DocumentChunk, RetrievalResult
from src.storage.base import StorageAdapter
from src.storage.postgres import _row_to_chunk_dict

logger = logging.getLogger(__name__)


class PostgresRetrieval:
    """Retrieval interface for PostgreSQL chunks table.

    Uses UUID keyset pagination on chunk_id to avoid OFFSET degradation
    on large tables. Supports JSONB path queries on structured_data.
    """

    def __init__(self, adapter: StorageAdapter, pool: asyncpg.Pool) -> None:
        """Initialise with a storage adapter (for _adapt_chunk) and a connection pool.

        adapter: PostgresAdapter instance; used for schema migration on read.
        pool: asyncpg connection pool targeting the docintel database.
        """
        self._adapter = adapter
        self._pool = pool

    async def query(
        self,
        sql_filter: str | None = None,
        jsonb_path: str | None = None,
        content_types: list[str] | None = None,
        document_id: str | None = None,
        min_confidence: float | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> RetrievalResult:
        """Query chunks table with optional SQL filter and JSONB path expression.

        sql_filter: raw SQL WHERE clause fragment (e.g. "confidence_score > 0.7").
        jsonb_path: PostgreSQL jsonb_path_exists expression on structured_data.
        cursor: base64-encoded last chunk_id UUID from previous page.

        Returns:
            RetrievalResult with up to limit chunks, total count, next_cursor.
        """
        conditions: list[str] = []
        params: list[Any] = []
        param_idx = 1

        if document_id:
            conditions.append(f"document_id = ${param_idx}")
            params.append(document_id)
            param_idx += 1

        if content_types:
            placeholders = ", ".join(f"${param_idx + i}" for i in range(len(content_types)))
            conditions.append(f"content_type IN ({placeholders})")
            params.extend(content_types)
            param_idx += len(content_types)

        if min_confidence is not None:
            conditions.append(f"confidence_score >= ${param_idx}")
            params.append(min_confidence)
            param_idx += 1

        if jsonb_path:
            conditions.append(
                f"jsonb_path_exists(structured_data, ${param_idx}::jsonpath)"
            )
            params.append(jsonb_path)
            param_idx += 1

        if sql_filter:
            conditions.append(f"({sql_filter})")

        if cursor:
            last_id = base64.b64decode(cursor).decode("utf-8")
            conditions.append(f"chunk_id > ${param_idx}")
            params.append(last_id)
            param_idx += 1

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        count_conditions = [c for c in conditions if "chunk_id >" not in c]
        count_where = ("WHERE " + " AND ".join(count_conditions)) if count_conditions else ""
        count_params = params[: len(count_conditions)]

        async with self._pool.acquire() as conn:
            total: int = await conn.fetchval(
                f"SELECT COUNT(*) FROM chunks {count_where}", *count_params
            )

            rows = await conn.fetch(
                f"SELECT * FROM chunks {where_clause} "
                f"ORDER BY chunk_id LIMIT ${param_idx}",
                *params,
                limit + 1,
            )

        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]

        chunks = [
            self._adapter._adapt_chunk(_row_to_chunk_dict(r)) for r in rows
        ]

        next_cursor: str | None = None
        if has_more and chunks:
            next_cursor = base64.b64encode(
                chunks[-1].chunk_id.encode("utf-8")
            ).decode("utf-8")

        return RetrievalResult(chunks=chunks, total=total, next_cursor=next_cursor)
