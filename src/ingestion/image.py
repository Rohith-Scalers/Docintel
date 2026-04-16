"""Single-image ingestor for common raster formats.

Loads JPEG, PNG, TIFF, BMP, and WEBP files as a single-page document.
Images are always treated as scanned content since no native text layer
is present.
"""
from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from src.ingestion.base import IngestorBase, PageResult

logger = logging.getLogger(__name__)

_DEFAULT_DPI: int = 150


class ImageIngestor(IngestorBase):
    """Ingestor for standalone raster image files.

    Converts the image to a BGR numpy array suitable for layout detection.
    DPI is read from the image's EXIF/metadata when available; otherwise
    defaults to 150 DPI. The single result always has page_number=1 and
    is_scanned=True.
    """

    @property
    def supported_extensions(self) -> list[str]:
        """File extensions handled by this ingestor.

        Returns:
            list[str]: supported raster image extensions.
        """
        return [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"]

    def ingest(self, source_path: str) -> list[PageResult]:
        """Load a raster image and return a single-page PageResult.

        The image is converted to RGB before being transformed to BGR numpy
        format. DPI is extracted from PIL image info when present.

        Returns:
            list[PageResult]: list containing exactly one PageResult.
        """
        pil_image = Image.open(source_path).convert("RGB")

        dpi_info = pil_image.info.get("dpi")
        if dpi_info and isinstance(dpi_info, (tuple, list)) and len(dpi_info) >= 1:
            try:
                estimated_dpi = int(float(dpi_info[0]))
                if estimated_dpi <= 0:
                    estimated_dpi = _DEFAULT_DPI
            except (ValueError, TypeError):
                estimated_dpi = _DEFAULT_DPI
        else:
            estimated_dpi = _DEFAULT_DPI

        bgr_array = np.array(pil_image)[:, :, ::-1].copy()

        return [
            PageResult(
                page_number=1,
                image=bgr_array,
                native_text="",
                estimated_dpi=estimated_dpi,
                quality_tier="standard",
                is_scanned=True,
                metadata={},
            )
        ]
