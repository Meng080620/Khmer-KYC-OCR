"""Image quality detection: blur, glare, darkness and resolution.

All measurements are taken on the flattened card, not the whole photo, so a blurred
or bright background (portrait mode, glossy table) doesn't cause false rejects.
"""

import cv2
import numpy as np

from khmer_kyc_ocr.pipeline.warp import Image
from khmer_kyc_ocr.schemas import QualityReport

_SATURATED_LEVEL = 245
_GLARE_BLOCK = 16


def short_side(image: Image) -> int:
    return int(min(image.shape[:2]))


def sharpness(image: Image) -> float:
    """Variance of the Laplacian: high for crisp edges, low for motion/focus blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(image: Image) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(np.mean(gray))


def glare_ratio(image: Image) -> float:
    """Fraction of the card that is specularly blown out.

    A glossy highlight saturates a *contiguous patch* of pixels. We measure the worst
    local patch rather than a global count, so a naturally light card doesn't read as
    glare, while a bright reflected streak does.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    saturated = (gray >= _SATURATED_LEVEL).astype(np.uint8)
    h, w = saturated.shape
    gh, gw = max(1, h // _GLARE_BLOCK), max(1, w // _GLARE_BLOCK)
    resized = cv2.resize(saturated, (gw, gh), interpolation=cv2.INTER_AREA)
    return float(np.max(resized))


class QualityDetector:
    """Runs the quality gates and reports a ``QualityReport``."""

    def __init__(
        self,
        *,
        min_short_side_px: int,
        min_sharpness: float,
        min_brightness: float,
        max_glare_ratio: float,
    ) -> None:
        self.min_short_side_px = min_short_side_px
        self.min_sharpness = min_sharpness
        self.min_brightness = min_brightness
        self.max_glare_ratio = max_glare_ratio

    def check(self, card: Image) -> QualityReport:
        side = short_side(card)
        sharp = sharpness(card)
        bright = brightness(card)
        glare = glare_ratio(card)

        is_blurry = sharp < self.min_sharpness
        is_dark = bright < self.min_brightness
        is_glared = glare > self.max_glare_ratio
        too_small = side < self.min_short_side_px

        return QualityReport(
            short_side_px=side,
            sharpness=sharp,
            is_blurry=is_blurry,
            brightness=bright,
            is_dark=is_dark,
            glare_ratio=glare,
            is_glared=is_glared,
            passed=not (is_blurry or is_dark or is_glared or too_small),
        )
