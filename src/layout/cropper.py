"""Region cropper: extract and persist individual layout regions as images.

Crops detected regions from the full page image, saves them to
IMAGE_STORE_PATH with a deterministic filename, computes a SHA-256
content hash, and returns the updated RawRegion.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from src.config import DatabaseConfig
from src.models import RawRegion

logger = logging.getLogger(__name__)


def crop_and_save(
    page_image: np.ndarray,
    region: RawRegion,
    config: DatabaseConfig,
    padding: int = 8,
) -> RawRegion:
    """Crop a region from a page image, save it to IMAGE_STORE_PATH, and
    return the updated RawRegion with cropped_image_path and content_hash set.

    A padding of `padding` pixels is added on all sides (clamped to image
    bounds) to provide context for the VLM.

    File naming:
        {IMAGE_STORE_PATH}/{document_id}/page_{n:03d}_region_{i:02d}_{type}.png

    Returns:
        RawRegion: updated with cropped_image_path and content_hash.
    """
    h, w = page_image.shape[:2]

    x0 = max(0, int(region.bbox.x0) - padding)
    y0 = max(0, int(region.bbox.y0) - padding)
    x1 = min(w, int(region.bbox.x1) + padding)
    y1 = min(h, int(region.bbox.y1) + padding)

    cropped = page_image[y0:y1, x0:x1]

    output_dir = Path(config.image_store_path) / region.document_id
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"page_{region.page_number:03d}_"
        f"region_{region.region_index:02d}_"
        f"{region.region_type.value}.png"
    )
    output_path = output_dir / filename

    success, png_bytes_array = cv2.imencode(".png", cropped)
    if not success:
        logger.error(
            "cv2.imencode failed for region %s (doc %s page %d).",
            region.region_id,
            region.document_id,
            region.page_number,
        )
        return region

    png_bytes: bytes = png_bytes_array.tobytes()

    with open(output_path, "wb") as fh:
        fh.write(png_bytes)

    content_hash = hashlib.sha256(png_bytes).hexdigest()

    region.cropped_image_path = str(output_path)
    region.content_hash = content_hash

    return region
