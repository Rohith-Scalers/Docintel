"""Abstract base class for all document format ingestors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PageResult:
    """Single rasterised page produced by an ingestor.

    image: full-page BGR numpy array at the configured DPI.
    native_text: text extracted directly (e.g. PyMuPDF digital PDF).
        Empty string for scanned/image-only pages.
    page_number: 1-indexed page number.
    estimated_dpi: rasterisation DPI used.
    quality_tier: 'standard' or 'high_detail' — set by router.
    is_scanned: True when no native text was extractable.
    """

    page_number: int
    image: np.ndarray
    native_text: str = ""
    estimated_dpi: int = 300
    quality_tier: str = "standard"
    is_scanned: bool = False
    metadata: dict = field(default_factory=dict)


class IngestorBase(ABC):
    """Base class for all document format ingestors.

    Subclasses implement ingest() to return a list of PageResult objects,
    one per page (or logical page equivalent for non-paged formats).
    """

    @abstractmethod
    def ingest(self, source_path: str) -> list[PageResult]:
        """Load document and return one PageResult per page.

        Returns:
            list[PageResult]: ordered list of page results, 1-indexed page_number.
        """

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """List of lowercase file extensions this ingestor handles (e.g. ['.pdf']).

        Returns:
            list[str]: supported file extensions including the dot prefix.
        """
