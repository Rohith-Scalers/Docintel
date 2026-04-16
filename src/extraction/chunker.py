"""Group RawRegion objects from layout detection into DocumentChunk objects."""

from __future__ import annotations

import logging
from collections import defaultdict
from uuid import uuid4

from src.models import (
    CURRENT_SCHEMA_VERSION,
    DocumentChunk,
    RawRegion,
    RegionType,
)

logger = logging.getLogger(__name__)

# Region types that must always occupy their own chunk (never merged).
_ISOLATED_TYPES: frozenset[RegionType] = frozenset(
    {RegionType.FIGURE, RegionType.TABLE, RegionType.FORMULA}
)

# Sentence-ending punctuation characters; absence triggers page-break bridging.
_SENTENCE_ENDINGS: frozenset[str] = frozenset({".", "!", "?"})

# Region types treated as body text that may be merged within a page.
_MERGEABLE_TYPES: frozenset[RegionType] = frozenset({RegionType.TEXT})


class Chunker:
    """Groups ordered RawRegion objects into DocumentChunk objects.

    TEXT regions on the same page are merged into a single chunk up to
    max_text_chars characters. Isolated content types (FIGURE, TABLE,
    FORMULA) always produce their own chunk. CAPTION text is attached to the
    nearest preceding FIGURE or TABLE chunk on the same page rather than
    generating a standalone chunk. HEADER and FOOTER chunks carry
    metadata_only=True in their metadata. PAGE-break bridging sets
    page_break_context on the last TEXT chunk of page N and the first TEXT
    chunk of page N+1 when the page N text does not end with sentence-ending
    punctuation.
    """

    def __init__(self, max_text_chars: int = 1500) -> None:
        self._max_text_chars = max_text_chars

    def chunk_regions(
        self,
        regions: list[RawRegion],
        document_id: str,
    ) -> list[DocumentChunk]:
        """Group ordered RawRegion objects into DocumentChunk objects.

        Processing order: regions are processed page by page, each page in
        ascending region_index order (reading order from XY-cut). TEXT regions
        within a page are merged up to max_text_chars. Non-text content types
        always get their own chunk. Captions are attached to the nearest
        preceding FIGURE or TABLE chunk on the same page instead of creating a
        standalone chunk. Page-break context strings are added to TEXT chunks
        that straddle page boundaries.

        Returns:
            list[DocumentChunk]: chunks in reading order with 0-based
            chunk_index assigned sequentially across all pages.
        """
        sorted_regions = sorted(regions, key=lambda r: (r.page_number, r.region_index))

        pages: dict[int, list[RawRegion]] = defaultdict(list)
        for region in sorted_regions:
            pages[region.page_number].append(region)

        all_chunks: list[DocumentChunk] = []
        reading_order_counter = 0

        for page_number in sorted(pages.keys()):
            page_regions = pages[page_number]
            page_chunks, reading_order_counter = self._process_page(
                page_regions, document_id, reading_order_counter
            )
            all_chunks.extend(page_chunks)

        self._apply_page_break_context(all_chunks)

        for idx, chunk in enumerate(all_chunks):
            object.__setattr__(chunk, "chunk_index", idx) if chunk.model_config.get(
                "frozen"
            ) else setattr(chunk, "chunk_index", idx)

        return all_chunks

    def _process_page(
        self,
        regions: list[RawRegion],
        document_id: str,
        reading_order_counter: int,
    ) -> tuple[list[DocumentChunk], int]:
        """Process all regions on a single page and return the resulting chunks.

        Returns:
            tuple[list[DocumentChunk], int]: (page_chunks, updated_counter).
        """
        chunks: list[DocumentChunk] = []
        # Accumulator for merging consecutive TEXT regions.
        text_buffer: list[RawRegion] = []

        def flush_text_buffer() -> None:
            nonlocal reading_order_counter
            if not text_buffer:
                return
            chunk = self._merge_text_regions(
                text_buffer, document_id, reading_order_counter
            )
            chunks.append(chunk)
            reading_order_counter += 1
            text_buffer.clear()

        for region in regions:
            rtype = region.region_type

            if rtype == RegionType.CAPTION:
                flush_text_buffer()
                self._attach_caption(region, chunks)
                continue

            if rtype in (RegionType.HEADER, RegionType.FOOTER):
                flush_text_buffer()
                chunk = self._make_metadata_chunk(
                    region, document_id, reading_order_counter
                )
                chunks.append(chunk)
                reading_order_counter += 1
                continue

            if rtype in _ISOLATED_TYPES:
                flush_text_buffer()
                chunk = self._make_isolated_chunk(
                    region, document_id, reading_order_counter
                )
                chunks.append(chunk)
                reading_order_counter += 1
                continue

            if rtype in _MERGEABLE_TYPES:
                region_text = region.metadata.get("raw_text", "")
                if text_buffer:
                    current_text = "".join(
                        r.metadata.get("raw_text", "") for r in text_buffer
                    )
                    if len(current_text) + len(region_text) > self._max_text_chars:
                        flush_text_buffer()
                text_buffer.append(region)
                continue

            # Unknown or future region type: emit as isolated chunk.
            logger.debug("Unknown region type %r; emitting as isolated chunk.", rtype)
            flush_text_buffer()
            chunk = self._make_isolated_chunk(
                region, document_id, reading_order_counter
            )
            chunks.append(chunk)
            reading_order_counter += 1

        flush_text_buffer()
        return chunks, reading_order_counter

    def _merge_text_regions(
        self,
        regions: list[RawRegion],
        document_id: str,
        reading_order_index: int,
    ) -> DocumentChunk:
        """Merge one or more consecutive TEXT regions into a single DocumentChunk.

        Returns:
            DocumentChunk: merged text chunk with combined raw_text.
        """
        combined_text = " ".join(
            r.metadata.get("raw_text", "") for r in regions
        ).strip()
        first = regions[0]
        return DocumentChunk(
            chunk_id=str(uuid4()),
            document_id=document_id,
            page_number=first.page_number,
            chunk_index=0,
            reading_order_index=reading_order_index,
            content_type=RegionType.TEXT,
            raw_text=combined_text,
            cropped_image_path=first.cropped_image_path,
            schema_version=CURRENT_SCHEMA_VERSION,
        )

    def _make_isolated_chunk(
        self,
        region: RawRegion,
        document_id: str,
        reading_order_index: int,
    ) -> DocumentChunk:
        """Create a standalone chunk for an isolated region type.

        Returns:
            DocumentChunk: chunk wrapping the single region.
        """
        return DocumentChunk(
            chunk_id=str(uuid4()),
            document_id=document_id,
            page_number=region.page_number,
            chunk_index=0,
            reading_order_index=reading_order_index,
            content_type=region.region_type,
            raw_text=region.metadata.get("raw_text", ""),
            cropped_image_path=region.cropped_image_path,
            schema_version=CURRENT_SCHEMA_VERSION,
        )

    def _make_metadata_chunk(
        self,
        region: RawRegion,
        document_id: str,
        reading_order_index: int,
    ) -> DocumentChunk:
        """Create a metadata-only chunk for HEADER or FOOTER regions.

        Returns:
            DocumentChunk: chunk with metadata_only=True flag in metadata.
        """
        return DocumentChunk(
            chunk_id=str(uuid4()),
            document_id=document_id,
            page_number=region.page_number,
            chunk_index=0,
            reading_order_index=reading_order_index,
            content_type=region.region_type,
            raw_text=region.metadata.get("raw_text", ""),
            cropped_image_path=region.cropped_image_path,
            metadata={"metadata_only": True},
            schema_version=CURRENT_SCHEMA_VERSION,
        )

    @staticmethod
    def _attach_caption(region: RawRegion, chunks: list[DocumentChunk]) -> None:
        """Attach a CAPTION region's text to the nearest preceding FIGURE or TABLE chunk.

        Searches backwards through the chunks already produced on the same page.
        If no eligible anchor chunk is found the caption is silently dropped;
        callers may optionally persist the region via region.metadata.

        Returns:
            None
        """
        caption_text = region.metadata.get("raw_text", "").strip()
        if not caption_text:
            return

        for chunk in reversed(chunks):
            if chunk.page_number != region.page_number:
                break
            if chunk.content_type in (RegionType.FIGURE, RegionType.TABLE):
                existing = chunk.caption or ""
                separator = " " if existing else ""
                chunk.caption = existing + separator + caption_text
                return

        logger.debug(
            "No FIGURE or TABLE anchor found for caption on page %d; caption dropped.",
            region.page_number,
        )

    @staticmethod
    def _apply_page_break_context(chunks: list[DocumentChunk]) -> None:
        """Set page_break_context on TEXT chunks that straddle page boundaries.

        For each consecutive pair of pages, if the last TEXT chunk on page N
        does not end with sentence-ending punctuation (. ! ?) the context
        strings are written onto both the closing chunk (page N) and the
        opening chunk (page N+1).

        Returns:
            None
        """
        # Index TEXT chunks by page number for efficient lookup.
        text_by_page: dict[int, list[DocumentChunk]] = defaultdict(list)
        for chunk in chunks:
            if chunk.content_type == RegionType.TEXT and not chunk.metadata.get(
                "metadata_only"
            ):
                text_by_page[chunk.page_number].append(chunk)

        for page_n, page_chunks in text_by_page.items():
            page_n1_chunks = text_by_page.get(page_n + 1)
            if not page_n1_chunks:
                continue

            last_chunk = page_chunks[-1]
            first_next_chunk = page_n1_chunks[0]

            closing_text = last_chunk.raw_text.rstrip()
            if not closing_text:
                continue
            if closing_text[-1] in _SENTENCE_ENDINGS:
                continue

            last_50 = closing_text[-50:]
            first_50 = first_next_chunk.raw_text[:50] if first_next_chunk.raw_text else ""

            last_chunk.page_break_context = (
                f"[...continues on page {page_n + 1}: {first_50}]"
            )
            first_next_chunk.page_break_context = (
                f"[...continued from page {page_n}: {last_50}]"
            )
