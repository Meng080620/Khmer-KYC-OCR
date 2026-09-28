import cv2

from khmer_kyc_ocr.pipeline.warp import Image


def short_side(image: Image) -> int:
    return int(min(image.shape[:2]))


def sharpness(image: Image) -> float:
    """Variance of the Laplacian: high for crisp edges, low for motion/focus blur.

    Measured on the flattened card, not the whole photo, so a blurred background
    (portrait mode, shallow depth of field) doesn't cause false rejects.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
