"""Image preprocessing: make the flattened card more readable before OCR.

Readability improvements only — we never distort the geometry (that already happened
in the document detector) and we never sharpen a blurry image into a false positive;
the quality gate runs first and rejects unreadable input.
"""

import cv2
import numpy as np

from khmer_kyc_ocr.pipeline.warp import Image


def improve_readability(image: Image) -> Image:
    """Return a contrast-enhanced grayscale image ready for line detection and OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.bilateralFilter(gray, 5, 30, 30)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return np.asarray(enhanced, dtype=np.uint8)
