from __future__ import annotations

import hashlib
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path

from src.ingestion.base import IngestorBase
from src.ingestion.pdf import PdfIngestor
from src.ingestion.docx import DocxIngestor
from src.ingestion.image import ImageIngestor
from src.ingestion.text import TextIngestor
from src.ingestion.excel import ExcelIngestor
from src.ingestion.email import EmailIngestor
from src.observability import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Extension → ingestor class mapping
# ---------------------------------------------------------------------------

_EXTENSION_MAP: dict[str, type[IngestorBase]] = {
    ".pdf": PdfIngestor,
    ".docx": DocxIngestor,
    ".doc": DocxIngestor,
    ".jpg": ImageIngestor,
    ".jpeg": ImageIngestor,
    ".png": ImageIngestor,
    ".tiff": ImageIngestor,
    ".tif": ImageIngestor,
    ".bmp": ImageIngestor,
    ".webp": ImageIngestor,
    ".txt": TextIngestor,
    ".html": TextIngestor,
    ".htm": TextIngestor,
    ".csv": TextIngestor,
    ".md": TextIngestor,
    ".xlsx": ExcelIngestor,
    ".xls": ExcelIngestor,
    ".xlsm": ExcelIngestor,
    ".eml": EmailIngestor,
    ".msg": EmailIngestor,
}

# MIME types that override extension detection when the extension is absent
_MIME_MAP: dict[str, type[IngestorBase]] = {
    "application/pdf": PdfIngestor,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocxIngestor,
    "application/msword": DocxIngestor,
    "image/jpeg": ImageIngestor,
    "image/png": ImageIngestor,
    "image/tiff": ImageIngestor,
    "image/bmp": ImageIngestor,
    "image/webp": ImageIngestor,
    "text/plain": TextIngestor,
    "text/html": TextIngestor,
    "text/csv": TextIngestor,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ExcelIngestor,
    "application/vnd.ms-excel": ExcelIngestor,
    "message/rfc822": EmailIngestor,
}

# Minimum characters of native text on a page to consider it digital-native
_SCANNED_TEXT_THRESHOLD = 50

# Text density ratio below which a PDF page is flagged as high_detail
_HIGH_DETAIL_DENSITY_THRESHOLD = 0.3


@dataclass
class RoutingDecision:
    """Output of the smart router for a single source document.

    ingestor_cls: the IngestorBase subclass to use for loading.
    format: lowercase format string (e.g. "pdf", "docx", "image").
    quality_tier: "standard" or "high_detail" — applied to every page.
    content_hash: sha256 hex of the source file bytes (dedup key).
    """

    ingestor_cls: type[IngestorBase]
    format: str
    quality_tier: str
    content_hash: str


def _sha256_file(path: str) -> str:
    """Compute the sha256 hex digest of a file without loading it fully into memory.

    Returns:
        str: lowercase hex sha256 digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_format(path: str) -> str:
    """Return a normalised format string from file extension or MIME sniff.

    Returns:
        str: lowercase format identifier (e.g. "pdf", "docx", "image").
    """
    ext = Path(path).suffix.lower()
    if ext in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"):
        return "image"
    if ext in (".xlsx", ".xls", ".xlsm"):
        return "excel"
    if ext in (".eml", ".msg"):
        return "email"
    if ext in (".txt", ".html", ".htm", ".csv", ".md"):
        return "text"
    return ext.lstrip(".") or "unknown"


def _score_pdf_quality(path: str) -> str:
    """Assess PDF quality tier by sampling text density from the first 5 pages.

    A PDF is considered "high_detail" (likely scanned or low-quality) when the
    average ratio of text characters to page area is below the threshold.

    Returns:
        str: "standard" or "high_detail".
    """
    try:
        import fitz  # type: ignore[import]

        doc = fitz.open(path)
        densities: list[float] = []
        for i, page in enumerate(doc):
            if i >= 5:
                break
            text_len = len(page.get_text().strip())
            area = page.rect.width * page.rect.height
            densities.append(text_len / max(area, 1))
        doc.close()
        avg_density = sum(densities) / max(len(densities), 1)
        return "high_detail" if avg_density < _HIGH_DETAIL_DENSITY_THRESHOLD else "standard"
    except Exception as exc:
        logger.warning("pdf_quality_score_failed", error=str(exc))
        return "standard"


class SmartRouter:
    """Selects an ingestor, determines quality tier, and computes the dedup hash.

    The router never loads or processes document content beyond what is needed
    for format detection and quality scoring. All heavy work is deferred to
    the ingestor and pipeline.
    """

    def route(self, source_path: str) -> RoutingDecision:
        """Analyse source_path and return a RoutingDecision.

        Format detection order:
        1. File extension lookup in _EXTENSION_MAP.
        2. MIME type sniff via mimetypes.guess_type() if extension unknown.
        3. Fallback to TextIngestor.

        Quality tier:
        - PDFs: sample text density across first 5 pages.
        - All other formats: "standard" by default.

        Returns:
            RoutingDecision with ingestor_cls, format, quality_tier, content_hash.
        Raises:
            FileNotFoundError: if source_path does not exist.
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source document not found: {source_path}")

        content_hash = _sha256_file(source_path)
        ext = Path(source_path).suffix.lower()
        ingestor_cls = _EXTENSION_MAP.get(ext)

        if ingestor_cls is None:
            mime, _ = mimetypes.guess_type(source_path)
            if mime:
                ingestor_cls = _MIME_MAP.get(mime)

        if ingestor_cls is None:
            logger.warning(
                "unknown_format_fallback",
                path=source_path,
                ext=ext,
            )
            ingestor_cls = TextIngestor

        fmt = _detect_format(source_path)

        quality_tier = "standard"
        if ingestor_cls is PdfIngestor:
            quality_tier = _score_pdf_quality(source_path)

        logger.info(
            "router_decision",
            path=source_path,
            format=fmt,
            ingestor=ingestor_cls.__name__,
            quality_tier=quality_tier,
        )

        return RoutingDecision(
            ingestor_cls=ingestor_cls,
            format=fmt,
            quality_tier=quality_tier,
            content_hash=content_hash,
        )
