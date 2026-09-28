"""Where each field sits on the flattened card.

Coordinates are fractions of the card (0..1) so they survive a change of CARD_W/CARD_H.

UNCALIBRATED: these boxes are an approximation of the current Cambodian national ID front and
MUST be tuned against real samples with `spike/calibrate.py` before production. The version
string says so, and every API response carries it, so results from an uncalibrated template
can never be mistaken for calibrated ones.
"""

from dataclasses import dataclass

import numpy as np

from khmer_kyc_ocr.pipeline.warp import Image
from khmer_kyc_ocr.schemas import FieldName

TEMPLATE_VERSION = "kh-nid-front-v1-uncalibrated"


@dataclass(frozen=True, slots=True)
class Roi:
    x0: float
    y0: float
    x1: float
    y1: float

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            round(self.x0 * width),
            round(self.y0 * height),
            round(self.x1 * width),
            round(self.y1 * height),
        )


TEMPLATE_V1: dict[FieldName, Roi] = {
    "id_number": Roi(0.62, 0.03, 0.98, 0.14),
    "name_khmer": Roi(0.28, 0.16, 0.98, 0.27),
    "name_latin": Roi(0.28, 0.27, 0.98, 0.36),
    "date_of_birth": Roi(0.28, 0.36, 0.66, 0.46),
    "sex": Roi(0.66, 0.36, 0.82, 0.46),
    "expiry_date": Roi(0.28, 0.74, 0.98, 0.84),
}


def crop(card: Image, roi: Roi) -> Image:
    h, w = card.shape[:2]
    x0, y0, x1, y1 = roi.to_pixels(w, h)
    # Copy so the recognizer can never mutate or keep a view into another field's pixels.
    return np.ascontiguousarray(card[y0:y1, x0:x1]).copy()
