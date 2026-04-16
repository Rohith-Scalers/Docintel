"""DocLayout-YOLO layout region detector.

Wraps an Ultralytics YOLO model trained on the DocLayout-YOLO dataset.
Maps class IDs to RegionType values and constructs RawRegion objects from
detected bounding boxes. Falls back to a single full-page TEXT region when
the model weights file is not present.
"""
from __future__ import annotations

import logging
import uuid

import numpy as np

from src.config import LayoutConfig
from src.models import BoundingBox, RawRegion, RegionType

logger = logging.getLogger(__name__)

CLASS_MAP: dict[int, RegionType] = {
    0: RegionType.TEXT,
    1: RegionType.TEXT,
    2: RegionType.TEXT,
    3: RegionType.TABLE,
    4: RegionType.FIGURE,
    5: RegionType.CAPTION,
    6: RegionType.CAPTION,
    7: RegionType.HEADER,
    8: RegionType.FOOTER,
    9: RegionType.TEXT,
    10: RegionType.FORMULA,
}


class LayoutDetector:
    """Wrapper around DocLayout-YOLO for document region detection.

    The YOLO model is loaded once at construction time. If the model weights
    file does not exist a warning is logged and the detector falls back to
    returning a single full-page TEXT region for every call to detect().
    """

    def __init__(self, config: LayoutConfig) -> None:
        """Initialise the layout detector.

        Attempts to load the YOLO model from config.model_path. Records
        whether the model loaded successfully for use in detect().

        Args:
            config: LayoutConfig instance with model_path, conf_threshold,
                iou_threshold, and device settings.
        """
        self._config = config
        self._model = None

        import os

        if not os.path.exists(config.model_path):
            logger.warning(
                "DocLayout-YOLO weights not found at '%s'. "
                "Falling back to full-page TEXT region for all pages.",
                config.model_path,
            )
            return

        try:
            from ultralytics import YOLO

            self._model = YOLO(config.model_path)
            self._model.to(config.device)
            logger.info(
                "DocLayout-YOLO loaded from '%s' on device '%s'.",
                config.model_path,
                config.device,
            )
        except Exception:
            logger.exception(
                "Failed to load DocLayout-YOLO from '%s'. Using fallback.",
                config.model_path,
            )

    def detect(
        self,
        page_image: np.ndarray,
        page_number: int,
        document_id: str,
    ) -> list[RawRegion]:
        """Run layout detection on a single page image.

        If the YOLO model is unavailable, returns a single full-page TEXT
        region covering the entire image. Otherwise, runs inference with the
        configured confidence and IoU thresholds and builds a RawRegion for
        each accepted detection.

        cropped_image_path and content_hash are left as empty strings;
        the cropper module fills these after cropping is performed.
        region_index is assigned in raw detection order and will be updated
        by the reading order module.

        Args:
            page_image: BGR numpy array of the preprocessed page.
            page_number: 1-indexed page number.
            document_id: parent document identifier.

        Returns:
            list[RawRegion]: detected regions in raw detection order.
        """
        h, w = page_image.shape[:2]

        if self._model is None:
            return [
                RawRegion(
                    region_id=str(uuid.uuid4()),
                    document_id=document_id,
                    page_number=page_number,
                    region_index=0,
                    region_type=RegionType.TEXT,
                    bbox=BoundingBox(
                        x0=0.0,
                        y0=0.0,
                        x1=float(w),
                        y1=float(h),
                        page_width=float(w),
                        page_height=float(h),
                    ),
                    cropped_image_path="",
                    content_hash="",
                    detector_confidence=1.0,
                )
            ]

        results = self._model.predict(
            source=page_image,
            conf=self._config.conf_threshold,
            iou=self._config.iou_threshold,
            imgsz=self._config.input_size,
            device=self._config.device,
            verbose=False,
        )

        regions: list[RawRegion] = []
        if not results:
            return regions

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return regions

        for idx, box in enumerate(boxes):
            x0, y0, x1, y1 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            region_type = CLASS_MAP.get(cls_id, RegionType.TEXT)

            regions.append(
                RawRegion(
                    region_id=str(uuid.uuid4()),
                    document_id=document_id,
                    page_number=page_number,
                    region_index=idx,
                    region_type=region_type,
                    bbox=BoundingBox(
                        x0=float(x0),
                        y0=float(y0),
                        x1=float(x1),
                        y1=float(y1),
                        page_width=float(w),
                        page_height=float(h),
                    ),
                    cropped_image_path="",
                    content_hash="",
                    detector_confidence=confidence,
                )
            )

        return regions
