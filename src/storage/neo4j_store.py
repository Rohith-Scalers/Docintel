"""Async Neo4j storage adapter using the official neo4j Python driver."""

from __future__ import annotations

import json
import logging
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver

from src.config import DatabaseConfig
from src.models import (
    Document,
    DocumentChunk,
    RawRegion,
    RegionType,
    TableData,
    BoundingBox,
)
from src.storage.base import StorageAdapter

logger = logging.getLogger(__name__)


class Neo4jAdapter(StorageAdapter):
    """Neo4j storage adapter using the official async Python driver.

    Graph schema:
      Nodes:  Document, Chunk, Region, Entity, TableRow
      Edges:  HAS_CHUNK, NEXT_CHUNK, SOURCE_REGION, MENTIONS, HAS_ROW, HAS_REGION

    Entity nodes are MERGEd on (text, type) to deduplicate across documents.
    NEXT_CHUNK edges form a linked list ordered by reading_order_index.
    """

    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver

    @classmethod
    def from_config(cls, config: DatabaseConfig) -> "Neo4jAdapter":
        """Construct a Neo4jAdapter from a DatabaseConfig.

        Returns:
            Configured Neo4jAdapter instance.
        """
        driver = AsyncGraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password),
        )
        return cls(driver)

    async def _ensure_constraints(self, session: Any) -> None:
        """Create uniqueness constraints idempotently.

        Returns:
            None
        """
        await session.run(
            "CREATE CONSTRAINT doc_content_hash IF NOT EXISTS "
            "FOR (d:Document) REQUIRE d.content_hash IS UNIQUE"
        )
        await session.run(
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
        )
        await session.run(
            "CREATE CONSTRAINT region_id IF NOT EXISTS "
            "FOR (r:Region) REQUIRE r.region_id IS UNIQUE"
        )

    async def write_document(self, document: Document) -> None:
        """Persist a Document and all related nodes and relationships.

        Execution order:
          1. Document node (MERGE on content_hash)
          2. Chunk nodes with HAS_CHUNK edges from Document
          3. NEXT_CHUNK chain ordered by reading_order_index
          4. Region nodes with HAS_REGION edges from Document
          5. Entity nodes (MERGE dedup) with MENTIONS edges from Chunks
          6. TableRow nodes with HAS_ROW edges from TABLE Chunks

        Returns:
            None
        """
        async with self._driver.session() as session:
            await self._ensure_constraints(session)
            await session.execute_write(self._write_tx, document)

    @staticmethod
    async def _write_tx(tx: Any, document: Document) -> None:
        """Transactional write for all document graph nodes.

        Returns:
            None
        """
        doc_props = {
            "document_id": document.document_id,
            "source_path": document.source_path,
            "content_hash": document.content_hash,
            "format": document.format,
            "total_pages": document.total_pages,
            "processing_status": document.processing_status,
            "schema_version": document.schema_version,
            "metadata": json.dumps(document.metadata),
        }
        await tx.run(
            "MERGE (d:Document {content_hash: $content_hash}) SET d += $props",
            content_hash=document.content_hash,
            props=doc_props,
        )

        sorted_chunks = sorted(document.chunks, key=lambda c: c.reading_order_index)
        for chunk in sorted_chunks:
            chunk_props = _chunk_to_props(chunk)
            await tx.run(
                """
                MATCH (d:Document {document_id: $doc_id})
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c += $props
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                doc_id=document.document_id,
                chunk_id=chunk.chunk_id,
                props=chunk_props,
            )

        for i in range(len(sorted_chunks) - 1):
            await tx.run(
                """
                MATCH (a:Chunk {chunk_id: $a_id})
                MATCH (b:Chunk {chunk_id: $b_id})
                MERGE (a)-[:NEXT_CHUNK]->(b)
                """,
                a_id=sorted_chunks[i].chunk_id,
                b_id=sorted_chunks[i + 1].chunk_id,
            )

        for region in document.regions:
            region_props = _region_to_props(region)
            await tx.run(
                """
                MATCH (d:Document {document_id: $doc_id})
                MERGE (r:Region {region_id: $region_id})
                SET r += $props
                MERGE (d)-[:HAS_REGION]->(r)
                """,
                doc_id=document.document_id,
                region_id=region.region_id,
                props=region_props,
            )

        for chunk in document.chunks:
            for entity in chunk.entities:
                entity_text = entity.get("text", "")
                entity_type = entity.get("type", "UNKNOWN")
                if not entity_text:
                    continue
                await tx.run(
                    """
                    MERGE (e:Entity {text: $text, type: $type})
                    WITH e
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    text=entity_text,
                    type=entity_type,
                    chunk_id=chunk.chunk_id,
                )

            if chunk.content_type == RegionType.TABLE and chunk.table_data:
                for row_index, row_values in enumerate(chunk.table_data.rows):
                    await tx.run(
                        """
                        MATCH (c:Chunk {chunk_id: $chunk_id})
                        CREATE (tr:TableRow {
                            row_index: $row_index,
                            values: $values
                        })
                        MERGE (c)-[:HAS_ROW]->(tr)
                        """,
                        chunk_id=chunk.chunk_id,
                        row_index=row_index,
                        values=json.dumps(row_values),
                    )

    async def find_by_hash(self, content_hash: str) -> Document | None:
        """Retrieve a full Document by its sha256 content hash.

        Returns:
            Document with all chunks and regions, or None if not found.
        """
        async with self._driver.session() as session:
            doc_result = await session.run(
                "MATCH (d:Document {content_hash: $hash}) RETURN d",
                hash=content_hash,
            )
            doc_record = await doc_result.single()
            if doc_record is None:
                return None

            doc_node = dict(doc_record["d"])
            doc_node["metadata"] = json.loads(doc_node.get("metadata", "{}"))

            chunks_result = await session.run(
                """
                MATCH (d:Document {content_hash: $hash})-[:HAS_CHUNK]->(c:Chunk)
                RETURN c ORDER BY c.reading_order_index
                """,
                hash=content_hash,
            )
            chunks: list[DocumentChunk] = []
            async for record in chunks_result:
                raw = _node_to_chunk_dict(dict(record["c"]))
                chunks.append(self._adapt_chunk(raw))

            regions_result = await session.run(
                """
                MATCH (d:Document {content_hash: $hash})-[:HAS_REGION]->(r:Region)
                RETURN r ORDER BY r.region_index
                """,
                hash=content_hash,
            )
            regions: list[RawRegion] = []
            async for record in regions_result:
                raw = _node_to_region_dict(dict(record["r"]))
                regions.append(RawRegion(**raw))

        return Document(**doc_node, chunks=chunks, regions=regions)

    async def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """Retrieve a single chunk by its UUID.

        Returns:
            DocumentChunk if found, None otherwise.
        """
        async with self._driver.session() as session:
            result = await session.run(
                "MATCH (c:Chunk {chunk_id: $chunk_id}) RETURN c",
                chunk_id=chunk_id,
            )
            record = await result.single()
            if record is None:
                return None
            raw = _node_to_chunk_dict(dict(record["c"]))
            return self._adapt_chunk(raw)

    async def close(self) -> None:
        """Close the Neo4j async driver.

        Returns:
            None
        """
        await self._driver.close()


def _chunk_to_props(chunk: DocumentChunk) -> dict:
    """Serialize a DocumentChunk to a flat Neo4j property dict.

    Returns:
        Dict of Neo4j-compatible property values.
    """
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "page_number": chunk.page_number,
        "chunk_index": chunk.chunk_index,
        "reading_order_index": chunk.reading_order_index,
        "content_type": chunk.content_type.value,
        "raw_text": chunk.raw_text,
        "overview": chunk.overview,
        "table_data": json.dumps(chunk.table_data.model_dump()) if chunk.table_data else None,
        "image_description": chunk.image_description,
        "formula_latex": chunk.formula_latex,
        "structured_data": json.dumps(chunk.structured_data),
        "entities": json.dumps(chunk.entities),
        "cropped_image_path": chunk.cropped_image_path,
        "confidence_score": chunk.confidence_score,
        "correction_applied": chunk.correction_applied,
        "page_break_context": chunk.page_break_context,
        "caption": chunk.caption,
        "schema_version": chunk.schema_version,
        "metadata": json.dumps(chunk.metadata),
    }


def _region_to_props(region: RawRegion) -> dict:
    """Serialize a RawRegion to a flat Neo4j property dict.

    Returns:
        Dict of Neo4j-compatible property values.
    """
    return {
        "region_id": region.region_id,
        "document_id": region.document_id,
        "page_number": region.page_number,
        "region_index": region.region_index,
        "region_type": region.region_type.value,
        "bbox": json.dumps(region.bbox.model_dump()),
        "cropped_image_path": region.cropped_image_path,
        "content_hash": region.content_hash,
        "detector_confidence": region.detector_confidence,
        "schema_version": region.schema_version,
        "metadata": json.dumps(region.metadata),
    }


def _node_to_chunk_dict(node: dict) -> dict:
    """Convert a Neo4j Chunk node property dict to a DocumentChunk-compatible dict.

    Returns:
        Dict with JSON string fields decoded.
    """
    for field in ("structured_data", "entities", "metadata"):
        val = node.get(field)
        if isinstance(val, str):
            node[field] = json.loads(val)
        elif val is None:
            node[field] = {} if field != "entities" else []
    if isinstance(node.get("table_data"), str):
        node["table_data"] = json.loads(node["table_data"])
    return node


def _node_to_region_dict(node: dict) -> dict:
    """Convert a Neo4j Region node property dict to a RawRegion-compatible dict.

    Returns:
        Dict with JSON string fields decoded.
    """
    if isinstance(node.get("bbox"), str):
        node["bbox"] = json.loads(node["bbox"])
    if isinstance(node.get("metadata"), str):
        node["metadata"] = json.loads(node["metadata"])
    elif node.get("metadata") is None:
        node["metadata"] = {}
    return node
