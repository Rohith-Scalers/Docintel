"""Neo4j retrieval interface with entity traversal and Cypher query support."""

from __future__ import annotations

import base64
import logging
from typing import Any

from src.models import DocumentChunk, RetrievalResult
from src.storage.base import StorageAdapter
from src.storage.neo4j_store import _node_to_chunk_dict

logger = logging.getLogger(__name__)


class Neo4jRetrieval:
    """Retrieval interface for Neo4j Chunk nodes.

    Supports entity-based traversal (start from an Entity node, follow MENTIONS
    edges to Chunks) and direct Cypher queries. Uses SKIP/LIMIT pagination
    (Neo4j does not support keyset pagination — documented limitation).
    """

    def __init__(self, adapter: StorageAdapter, driver: Any) -> None:
        """Initialise with a storage adapter (for _adapt_chunk) and a Neo4j async driver.

        adapter: Neo4jAdapter instance; used for schema migration on read.
        driver: neo4j.AsyncDriver pointing at the docintel database.
        """
        self._adapter = adapter
        self._driver = driver

    async def traverse(
        self,
        start_entity: str | None = None,
        entity_type: str | None = None,
        depth: int = 1,
        cypher: str | None = None,
        document_id: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> RetrievalResult:
        """Traverse the graph from an entity or run a custom Cypher query.

        start_entity: entity text value to start traversal from.
        entity_type: optional entity type filter (ORG, PERSON, etc.).
        depth: relationship traversal depth from entity to chunks.
        cypher: if provided, execute this custom Cypher RETURN chunk query.
            Must return nodes with label Chunk as variable 'c'.
        cursor: base64-encoded integer SKIP offset.

        Returns:
            RetrievalResult with up to limit chunks, total count, next_cursor.
        """
        skip = 0
        if cursor:
            skip = int(base64.b64decode(cursor).decode("utf-8"))

        async with self._driver.session() as session:
            if cypher:
                chunks, total = await self._run_custom_cypher(
                    session, cypher, document_id, limit, skip
                )
            elif start_entity:
                chunks, total = await self._traverse_entity(
                    session,
                    start_entity,
                    entity_type,
                    depth,
                    document_id,
                    limit,
                    skip,
                )
            else:
                chunks, total = await self._fetch_all_chunks(
                    session, document_id, limit, skip
                )

        has_more = (skip + limit) < total
        next_cursor: str | None = None
        if has_more:
            next_cursor = base64.b64encode(str(skip + limit).encode("utf-8")).decode("utf-8")

        return RetrievalResult(chunks=chunks, total=total, next_cursor=next_cursor)

    async def _traverse_entity(
        self,
        session: Any,
        start_entity: str,
        entity_type: str | None,
        depth: int,
        document_id: str | None,
        limit: int,
        skip: int,
    ) -> tuple[list[DocumentChunk], int]:
        """Follow MENTIONS edges from an Entity node to Chunk nodes.

        Returns:
            Tuple of (chunks list, total count).
        """
        type_clause = "AND e.type = $entity_type " if entity_type else ""
        doc_clause = "AND c.document_id = $document_id " if document_id else ""
        rel_pattern = "-[:MENTIONS*1..{depth}]-".format(depth=depth)

        count_result = await session.run(
            f"""
            MATCH (e:Entity {{text: $text}}) {type_clause}
            MATCH (c:Chunk){rel_pattern}(e) {doc_clause}
            RETURN count(DISTINCT c) AS total
            """,
            text=start_entity,
            entity_type=entity_type,
            document_id=document_id,
        )
        count_record = await count_result.single()
        total = count_record["total"] if count_record else 0

        result = await session.run(
            f"""
            MATCH (e:Entity {{text: $text}}) {type_clause}
            MATCH (c:Chunk){rel_pattern}(e) {doc_clause}
            RETURN DISTINCT c ORDER BY c.reading_order_index
            SKIP $skip LIMIT $limit
            """,
            text=start_entity,
            entity_type=entity_type,
            document_id=document_id,
            skip=skip,
            limit=limit,
        )

        chunks: list[DocumentChunk] = []
        async for record in result:
            raw = _node_to_chunk_dict(dict(record["c"]))
            chunks.append(self._adapter._adapt_chunk(raw))

        return chunks, total

    async def _run_custom_cypher(
        self,
        session: Any,
        cypher: str,
        document_id: str | None,
        limit: int,
        skip: int,
    ) -> tuple[list[DocumentChunk], int]:
        """Execute a caller-supplied Cypher query returning Chunk nodes as 'c'.

        Returns:
            Tuple of (chunks list, total count).
        """
        result = await session.run(cypher, document_id=document_id, skip=skip, limit=limit)
        chunks: list[DocumentChunk] = []
        async for record in result:
            raw = _node_to_chunk_dict(dict(record["c"]))
            chunks.append(self._adapter._adapt_chunk(raw))

        return chunks, len(chunks) + skip

    async def _fetch_all_chunks(
        self,
        session: Any,
        document_id: str | None,
        limit: int,
        skip: int,
    ) -> tuple[list[DocumentChunk], int]:
        """Retrieve all Chunk nodes, optionally filtered by document_id.

        Returns:
            Tuple of (chunks list, total count).
        """
        doc_clause = "WHERE c.document_id = $document_id" if document_id else ""

        count_result = await session.run(
            f"MATCH (c:Chunk) {doc_clause} RETURN count(c) AS total",
            document_id=document_id,
        )
        count_record = await count_result.single()
        total = count_record["total"] if count_record else 0

        result = await session.run(
            f"MATCH (c:Chunk) {doc_clause} "
            "RETURN c ORDER BY c.reading_order_index "
            "SKIP $skip LIMIT $limit",
            document_id=document_id,
            skip=skip,
            limit=limit,
        )

        chunks: list[DocumentChunk] = []
        async for record in result:
            raw = _node_to_chunk_dict(dict(record["c"]))
            chunks.append(self._adapter._adapt_chunk(raw))

        return chunks, total
