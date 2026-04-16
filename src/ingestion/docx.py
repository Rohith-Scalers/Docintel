"""DOCX ingestor.

Converts .docx / .doc files to a temporary PDF via docx2pdf, then
delegates all page rasterisation and text extraction to PdfIngestor.
"""
from __future__ import annotations

import logging
import os
import tempfile

import docx2pdf

from src.ingestion.base import IngestorBase, PageResult
from src.ingestion.pdf import PdfIngestor

logger = logging.getLogger(__name__)


class DocxIngestor(IngestorBase):
    """Ingestor for Microsoft Word documents (.docx / .doc).

    Converts the input file to a temporary PDF using docx2pdf (which
    delegates to LibreOffice or Microsoft Word depending on the platform),
    then hands off to PdfIngestor for per-page rasterisation.
    """

    @property
    def supported_extensions(self) -> list[str]:
        """File extensions handled by this ingestor.

        Returns:
            list[str]: ['.docx', '.doc']
        """
        return [".docx", ".doc"]

    def ingest(self, source_path: str) -> list[PageResult]:
        """Convert the Word document to PDF and return one PageResult per page.

        The intermediate PDF is written to a system temporary file and removed
        after processing regardless of success or failure.

        Returns:
            list[PageResult]: ordered list of page results, 1-indexed page_number.
        """
        tmp_fd, tmp_pdf_path = tempfile.mkstemp(suffix=".pdf")
        os.close(tmp_fd)

        try:
            docx2pdf.convert(source_path, tmp_pdf_path)
            pdf_ingestor = PdfIngestor()
            results = pdf_ingestor.ingest(tmp_pdf_path)
        finally:
            if os.path.exists(tmp_pdf_path):
                os.remove(tmp_pdf_path)

        return results
