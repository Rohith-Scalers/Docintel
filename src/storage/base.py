"""Abstract base class for all storage adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import CURRENT_SCHEMA_VERSION, Document, DocumentChunk


class UnsupportedSchemaVersion(Exception):
    """Raised when a stored document has a schema_version below minimum supported."""


class StorageAdapter(ABC):
    """Abstract interface that all storage adapters must implement.

    Schema versioning: every adapter stores schema_version on write and
    calls _adapt_chunk() on read to upgrade older records transparently.
    SUPPORTED_SCHEMA_VERSIONS defines which versions can be upgraded.
    """

    SUPPORTED_SCHEMA_VERSIONS: set[int] = {1, 2}

    @abstractmethod
    async def write_document(self, document: Document) -> None:
        """Persist a Document and all its chunks and regions.

        Returns:
            None
        """

    @abstractmethod
    async def find_by_hash(self, content_hash: str) -> Document | None:
        """Look up an existing document by its source file sha256 hash.

        Returns:
            Document if found, None otherwise.
        """

    @abstractmethod
    async def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """Retrieve a single chunk by its UUID.

        Returns:
            DocumentChunk if found, None otherwise.
        """

    @abstractmethod
    async def close(self) -> None:
        """Close all connections held by this adapter.

        Returns:
            None
        """

    def _adapt_chunk(self, raw: dict) -> DocumentChunk:
        """Upgrade a raw dict from any supported schema_version to the current version.

        v1 → v2 changes:
          - rename image_path → cropped_image_path
          - add entities = []
          - add reading_order_index = chunk_index
          - add correction_applied = False

        Returns:
            DocumentChunk at CURRENT_SCHEMA_VERSION.
        Raises:
            UnsupportedSchemaVersion: if raw schema_version is not in SUPPORTED_SCHEMA_VERSIONS.
        """
        v = raw.get("schema_version", 1)
        if v not in self.SUPPORTED_SCHEMA_VERSIONS:
            raise UnsupportedSchemaVersion(f"schema_version={v} is not supported")
        if v == 1:
            raw["cropped_image_path"] = raw.pop("image_path", "")
            raw["entities"] = raw.get("entities", [])
            raw["reading_order_index"] = raw.get(
                "reading_order_index", raw.get("chunk_index", 0)
            )
            raw["correction_applied"] = raw.get("correction_applied", False)
            raw["schema_version"] = CURRENT_SCHEMA_VERSION
        return DocumentChunk(**raw)
