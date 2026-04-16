from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

CURRENT_SCHEMA_VERSION: int = 2


class RegionType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    FORMULA = "formula"
    HEADER = "header"
    FOOTER = "footer"
    CAPTION = "caption"


class BoundingBox(BaseModel):
    """Pixel coordinates of a detected region on a page image.

    All values are in pixels relative to the rasterised page image dimensions.
    """

    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float
    page_height: float

    @property
    def width(self) -> float:
        """Region width in pixels."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Region height in pixels."""
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        """Region area in pixels squared."""
        return self.width * self.height

    @property
    def cx(self) -> float:
        """Horizontal centre of region."""
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        """Vertical centre of region."""
        return (self.y0 + self.y1) / 2


class RawRegion(BaseModel):
    """A single layout element detected by DocLayout-YOLO on one page.

    Each RawRegion maps one-to-one with a cropped image file saved to
    IMAGE_STORE_PATH. region_index reflects reading order assigned by the
    XY-cut algorithm, not raw bbox position.
    """

    region_id: str
    document_id: str
    page_number: int
    region_index: int
    region_type: RegionType
    bbox: BoundingBox
    cropped_image_path: str
    content_hash: str
    detector_confidence: float
    metadata: dict = Field(default_factory=dict)
    schema_version: int = CURRENT_SCHEMA_VERSION


class TableCell(BaseModel):
    """A single cell within an extracted table."""

    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    value: str
    is_header: bool = False


class TableData(BaseModel):
    """Structured representation of an extracted table.

    Stores data in three complementary forms: a normalized 2-D grid for
    simple iteration, a flat cell list preserving span metadata, and the
    raw markdown string as returned by DeepSeek-OCR-2.
    """

    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    cells: list[TableCell] = Field(default_factory=list)
    markdown: str = ""
    continued_from_page: int | None = None
    merge_confidence: float | None = None


class DocumentChunk(BaseModel):
    """A logical content unit extracted from one or more adjacent regions.

    overview is always populated and serves as the primary embedding source.
    For FIGURE chunks image_description provides a dense visual caption.
    correction_applied records whether the validator triggered a re-extraction pass.
    """

    chunk_id: str
    document_id: str
    page_number: int
    chunk_index: int
    reading_order_index: int
    content_type: RegionType
    raw_text: str = ""
    overview: str = ""
    table_data: TableData | None = None
    image_description: str | None = None
    formula_latex: str | None = None
    structured_data: dict = Field(default_factory=dict)
    entities: list[dict] = Field(default_factory=list)
    cropped_image_path: str = ""
    confidence_score: float = 0.0
    correction_applied: bool = False
    page_break_context: str | None = None
    caption: str | None = None
    metadata: dict = Field(default_factory=dict)
    schema_version: int = CURRENT_SCHEMA_VERSION


class Document(BaseModel):
    """Top-level document produced by the extraction pipeline.

    content_hash (sha256 of the source file bytes) is the deduplication key
    checked at pipeline entry before any compute is spent. regions preserves
    the raw layout detection output independently of chunking decisions.
    """

    document_id: str
    source_path: str
    content_hash: str
    format: str
    total_pages: int
    processing_status: Literal[
        "pending", "processing", "complete", "review_required"
    ] = "pending"
    chunks: list[DocumentChunk] = Field(default_factory=list)
    regions: list[RawRegion] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    schema_version: int = CURRENT_SCHEMA_VERSION


class RetrievalResult(BaseModel):
    """Paginated result returned by all retrieval interfaces.

    next_cursor is None when no further pages exist. Its encoding is
    adapter-specific and opaque to callers — pass it back verbatim.
    """

    chunks: list[DocumentChunk]
    total: int
    next_cursor: str | None = None


class ExtractionSchema(BaseModel):
    """User-supplied extraction schema passed to the VLM for each chunk.

    fields maps output key names to plain-English descriptions used in the
    extraction prompt. validation_rules are evaluated post-extraction.
    """

    fields: dict[str, str] = Field(default_factory=dict)
    extract_entities: bool = True
    entity_types: list[str] = Field(
        default_factory=lambda: [
            "ORG", "PERSON", "DATE", "MONEY", "LOCATION", "PRODUCT"
        ]
    )
    custom_validation_hints: dict[str, str] = Field(default_factory=dict)
