"""Face detection: locate the portrait area and extract it as a separate image.

Haar-cascade-free YuNet detection needs a real face, so unit tests here cover the parts
that don't need one (loading, no-face -> detected=False). Positive detection is verified
manually with real ID photos.
"""

import numpy as np

from khmer_kyc_ocr.pipeline.face import FaceDetector
from khmer_kyc_ocr.pipeline.warp import CARD_H, CARD_W


def test_detector_loads_bundled_cascade() -> None:
    assert FaceDetector() is not None


def test_no_face_on_blank_card() -> None:
    card = np.full((CARD_H, CARD_W, 3), 235, dtype=np.uint8)
    result = FaceDetector().detect(card)
    assert result.detected is False
    assert result.image_base64 is None


def test_no_face_on_text_card() -> None:
    from tests.synthetic import make_card

    result = FaceDetector().detect(make_card(seed=2))
    assert result.detected is False
