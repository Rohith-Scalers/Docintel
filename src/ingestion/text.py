"""Plain text, HTML, CSV, and Markdown ingestor.

Renders text content to page images using PIL ImageDraw on an A4 canvas
at 300 DPI. HTML is stripped to plain text before rendering; CSV is
formatted as aligned rows.
"""
from __future__ import annotations

import csv
import html.parser
import io
import logging
import textwrap
from typing import IO

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.ingestion.base import IngestorBase, PageResult

logger = logging.getLogger(__name__)

MAX_CHARS_PER_PAGE: int = 3000
_PAGE_WIDTH_PX: int = 2480
_PAGE_HEIGHT_PX: int = 3508
_MARGIN_PX: int = 120
_LINE_HEIGHT_PX: int = 36
_WRAP_CHARS: int = 100
_FONT_SIZE: int = 24
_DPI: int = 300


class _HTMLTextStripper(html.parser.HTMLParser):
    """Minimal HTML parser that collects visible text nodes."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        """Return all collected text joined with whitespace.

        Returns:
            str: stripped plain text content.
        """
        return " ".join(self._parts)


def _strip_html(raw: str) -> str:
    """Remove HTML tags and return visible text.

    Returns:
        str: plain text with tags removed.
    """
    stripper = _HTMLTextStripper()
    stripper.feed(raw)
    return stripper.get_text()


def _csv_to_text(raw: str) -> str:
    """Convert CSV content to a human-readable line-per-row text representation.

    Returns:
        str: plain text where each row is formatted as comma-separated values.
    """
    reader = csv.reader(io.StringIO(raw))
    lines: list[str] = []
    for row in reader:
        lines.append(", ".join(row))
    return "\n".join(lines)


def _render_text_to_image(text_chunk: str) -> np.ndarray:
    """Render a text chunk onto a white A4 page image.

    Wraps text at _WRAP_CHARS characters per line and draws using the PIL
    default font. Returns a BGR numpy array.

    Returns:
        np.ndarray: BGR uint8 image of shape (_PAGE_HEIGHT_PX, _PAGE_WIDTH_PX, 3).
    """
    pil_image = Image.new("RGB", (_PAGE_WIDTH_PX, _PAGE_HEIGHT_PX), color=(255, 255, 255))
    draw = ImageDraw.Draw(pil_image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", _FONT_SIZE)
    except (IOError, OSError):
        font = ImageFont.load_default()

    wrapped_lines: list[str] = []
    for paragraph in text_chunk.splitlines():
        if paragraph.strip():
            wrapped_lines.extend(textwrap.wrap(paragraph, width=_WRAP_CHARS) or [""])
        else:
            wrapped_lines.append("")

    y_cursor = _MARGIN_PX
    for line in wrapped_lines:
        if y_cursor + _LINE_HEIGHT_PX > _PAGE_HEIGHT_PX - _MARGIN_PX:
            break
        draw.text((_MARGIN_PX, y_cursor), line, fill=(0, 0, 0), font=font)
        y_cursor += _LINE_HEIGHT_PX

    return np.array(pil_image)[:, :, ::-1].copy()


class TextIngestor(IngestorBase):
    """Ingestor for plain text, HTML, CSV, and Markdown files.

    Splits content into chunks of MAX_CHARS_PER_PAGE characters and renders
    each chunk as an A4 page image. native_text is populated with the raw
    chunk text so downstream OCR is skipped for these formats.
    """

    @property
    def supported_extensions(self) -> list[str]:
        """File extensions handled by this ingestor.

        Returns:
            list[str]: text-based file extensions.
        """
        return [".txt", ".html", ".htm", ".csv", ".md"]

    def ingest(self, source_path: str) -> list[PageResult]:
        """Load a text-based file and return one PageResult per logical page.

        HTML content is stripped of tags before rendering; CSV is converted to
        a readable row format. Long documents are paginated at MAX_CHARS_PER_PAGE.

        Returns:
            list[PageResult]: ordered list of page results, 1-indexed page_number.
        """
        with open(source_path, "r", encoding="utf-8", errors="replace") as fh:
            raw_content = fh.read()

        ext = source_path.rsplit(".", 1)[-1].lower()

        if ext in ("html", "htm"):
            text_content = _strip_html(raw_content)
        elif ext == "csv":
            text_content = _csv_to_text(raw_content)
        else:
            text_content = raw_content

        chunks: list[str] = [
            text_content[i: i + MAX_CHARS_PER_PAGE]
            for i in range(0, max(len(text_content), 1), MAX_CHARS_PER_PAGE)
        ]

        results: list[PageResult] = []
        for page_idx, chunk in enumerate(chunks):
            image = _render_text_to_image(chunk)
            results.append(
                PageResult(
                    page_number=page_idx + 1,
                    image=image,
                    native_text=chunk,
                    estimated_dpi=_DPI,
                    quality_tier="standard",
                    is_scanned=False,
                    metadata={},
                )
            )

        return results
