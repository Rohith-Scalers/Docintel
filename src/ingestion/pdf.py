"""PDF ingestor using PyMuPDF (fitz).

Rasterises each page to a BGR numpy array and detects whether the page
is digitally native or a scanned image based on extractable text volume.
"""
from __future__ import annotations

import logging

import fitz
import numpy as np
from PIL import Image

from src.ingestion.base import IngestorBase, PageResult

logger = logging.getLogger(__name__)

_STANDARD_DPI: int = 300
_HIGH_DETAIL_DPI: int = 400
_SCANNED_TEXT_THRESHOLD: int = 50


class PdfIngestor(IngestorBase):
    """Ingestor for PDF documents using PyMuPDF.

    Opens the PDF, iterates over all pages, extracts native text where
    available, rasterises each page to a BGR numpy array at the configured
    DPI, and marks pages as scanned when insufficient native text is found.
    """

    @property
    def supported_extensions(self) -> list[str]:
        """File extensions handled by this ingestor.

        Returns:
            list[str]: ['.pdf']
        """
        return [".pdf"]

    def ingest(self, source_path: str) -> list[PageResult]:
        """Load a PDF file and return one PageResult per page.

        Each page is rasterised at 300 DPI (standard) or 400 DPI
        (high_detail, set by upstream router). Scanned detection uses the
        character count of the native text layer; fewer than 50 characters
        indicates an image-only page.

        Returns:
            list[PageResult]: ordered list of page results, 1-indexed page_number.
        """
        results: list[PageResult] = []
        doc = fitz.open(source_path)

        try:
            for page_index in range(len(doc)):
                page = doc[page_index]
                native_text: str = page.get_text()
                is_scanned: bool = len(native_text.strip()) < _SCANNED_TEXT_THRESHOLD
                dpi: int = _STANDARD_DPI
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)

                pil_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                bgr_array = np.array(pil_image)[:, :, ::-1].copy()

                metadata = {
                    "page_label": page.get_label(),
                    "rotation": page.rotation,
                }

                results.append(
                    PageResult(
                        page_number=page_index + 1,
                        image=bgr_array,
                        native_text=native_text,
                        estimated_dpi=dpi,
                        quality_tier="standard",
                        is_scanned=is_scanned,
                        metadata=metadata,
                    )
                )
        finally:
            doc.close()

        return results
