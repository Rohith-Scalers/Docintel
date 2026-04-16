"""Image preprocessing pipeline for layout detection.

Applies deskew, bilateral denoising, CLAHE contrast normalisation, and a
resolution guard in sequence to prepare page images for DocLayout-YOLO.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_DESKEW_ANGLE_CLAMP: float = 5.0
_MIN_DPI_THRESHOLD: int = 150
_A4_HEIGHT_AT_300DPI: float = 3508.0
_BILATERAL_D: int = 9
_BILATERAL_SIGMA: int = 75


def _estimate_dpi_from_height(height: int) -> float:
    """Estimate the DPI of a page image by assuming A4 paper dimensions.

    Returns:
        float: estimated DPI based on A4 height at 300 DPI reference.
    """
    return height * 300.0 / _A4_HEIGHT_AT_300DPI


def _deskew(image: np.ndarray) -> np.ndarray:
    """Correct skew in a page image using Hough line detection.

    Converts to grayscale, applies binary threshold, detects lines via
    HoughLinesP, computes the dominant angle, and rotates the image to
    correct misalignment. The rotation angle is clamped to ±5° to avoid
    aggressive corrections on intentionally rotated content.

    Returns:
        np.ndarray: deskewed BGR image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=image.shape[1] // 4,
        maxLineGap=20,
    )

    if lines is None:
        return image

    angles: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if abs(angle) < _DESKEW_ANGLE_CLAMP:
                angles.append(angle)

    if not angles:
        return image

    median_angle = float(np.median(angles))
    rotation_angle = -median_angle

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def _bilateral_denoise(image: np.ndarray) -> np.ndarray:
    """Apply bilateral filtering to reduce noise while preserving edges.

    Returns:
        np.ndarray: denoised BGR image.
    """
    return cv2.bilateralFilter(
        image,
        d=_BILATERAL_D,
        sigmaColor=_BILATERAL_SIGMA,
        sigmaSpace=_BILATERAL_SIGMA,
    )


def _clahe_normalise(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast normalisation on the L channel of LAB colour space.

    Improves contrast on fax / photocopy scans without affecting hue or
    saturation.

    Returns:
        np.ndarray: contrast-normalised BGR image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_normalised = clahe.apply(l_channel)
    merged = cv2.merge([l_normalised, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _resolution_guard(image: np.ndarray) -> np.ndarray:
    """Upscale low-resolution images to ensure adequate detail for detection.

    Estimates DPI from image height assuming A4 dimensions. If the estimated
    DPI is below _MIN_DPI_THRESHOLD the image is upscaled 2x using bicubic
    interpolation.

    Returns:
        np.ndarray: image at sufficient resolution for layout detection.
    """
    estimated_dpi = _estimate_dpi_from_height(image.shape[0])
    if estimated_dpi < _MIN_DPI_THRESHOLD:
        logger.debug(
            "Estimated DPI %.1f below threshold %d — upscaling 2x.",
            estimated_dpi,
            _MIN_DPI_THRESHOLD,
        )
        image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return image


def preprocess_page(image: np.ndarray) -> np.ndarray:
    """Apply deskew, bilateral denoising, CLAHE contrast normalisation, and
    resolution guard to a page image before layout detection.

    Steps applied in order:
    1. Deskew: binary threshold + Hough line detection to find dominant angle;
       rotate by -angle (clamped to ±5°) to correct scanner misalignment.
    2. Bilateral denoise: cv2.bilateralFilter(d=9, sigmaColor=75, sigmaSpace=75)
       to smooth noise while preserving text stroke edges.
    3. CLAHE contrast normalisation on the L channel of LAB colour space to
       improve low-contrast fax / photocopy scans.
    4. Resolution guard: estimate DPI from image height (A4 assumption);
       if estimated DPI < 150, upscale 2x with cv2.INTER_CUBIC.

    Returns:
        np.ndarray: preprocessed BGR uint8 image.
    """
    image = _deskew(image)
    image = _bilateral_denoise(image)
    image = _clahe_normalise(image)
    image = _resolution_guard(image)
    return image
