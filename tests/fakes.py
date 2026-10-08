"""Shared test doubles and card builders, so no real ID image ever lives in the repo.

The fake OCR provider derives the ID number from the *image it receives* (the fraction of
dark pixels in the top-left quadrant), and simulates 180°-upside-down text by returning no
lines when the marker is in the bottom-right. This keeps the provider stateless — every
value comes from the input image — so a concurrency test can prove requests never mix.
"""

import cv2
import numpy as np

from khmer_kyc_ocr.pipeline.ocr import OcrLine
from khmer_kyc_ocr.pipeline.warp import CARD_H, CARD_W, Image
from tests.synthetic import photograph

FAKE_LINES = [
    "ព្រះរាជាណាចក្រកម្ពុជា",
    "អត្តសញ្ញាណប័ណ្ណ",
    "គោត្តនាម និងនាម: សុខ ដារា",
    "ឈ្មោះជាឡាតាំង: SOK DARA",
    "ថ្ងៃខែឆ្នាំកំណើត: 01.02.1990",
    "ភេទ: ប្រុស",
    "សញ្ជាតិ: ខ្មែរ",
    "ទីកន្លែងកំណើត: ភ្នំពេញ",
    "អាសយដ្ឋាន: ភ្នំពេញ កម្ពុជា",
    "សុពលភាព: 05.03.2020 ដល់ 04.03.2030",
]


def id_for_digit(digit: int) -> str:
    return f"01084561{digit}"


def lines_for(digit: int, confidence: float = 0.98) -> list[OcrLine]:
    texts = [f"លេខអត្តសញ្ញាណប័ណ្ណ: {id_for_digit(digit)}", *FAKE_LINES]
    return [
        OcrLine(
            text=text, bbox=(10, 10 + i * 32, 500, 26), confidence=confidence, line_number=i + 1
        )
        for i, text in enumerate(texts)
    ]


class FakeOCRProvider:
    """Stateless provider that decodes the digit from the top-left quadrant's dark fraction.

    If the marker is in the bottom-right, the card is upside down, so it returns no lines
    (garbled text), matching how real OCR behaves on 180°-flipped input.
    """

    engine = "fake-ocr"
    model_revision: str | None = None

    def __init__(self, confidence_override: float | None = None) -> None:
        self.confidence_override = confidence_override

    def read_lines(self, image: Image) -> list[OcrLine]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape
        top_left = float(np.mean(gray[0 : h // 2, 0 : w // 2] < 128))
        bottom_right = float(np.mean(gray[h // 2 :, w // 2 :] < 128))
        if top_left > bottom_right:
            digit = min(9, round(top_left * 10))
        else:
            return []
        confidence = self.confidence_override if self.confidence_override is not None else 0.98
        return lines_for(digit, confidence=confidence)


def card_with_digit(digit: int) -> Image:
    """A flattened card whose top-left quadrant dark fraction encodes ``digit`` (0..9)."""
    card = np.full((CARD_H, CARD_W, 3), 235, dtype=np.uint8)
    if digit:
        qw, qh = CARD_W // 2, CARD_H // 2
        margin = 10
        height = qh - 2 * margin
        width = int((qw * qh) * digit / 10 / height)
        card[margin : margin + height, margin : margin + width] = 20
    return card


def card_photo(digit: int = 0, *, angle: float = 7.0) -> bytes:
    ok, buf = cv2.imencode(
        ".jpg", photograph(card_with_digit(digit), angle_deg=angle), [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    assert ok
    return buf.tobytes()
