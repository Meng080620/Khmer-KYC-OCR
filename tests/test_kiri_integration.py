"""Real model, full pipeline, synthetic card. Skipped unless kiri-ocr and its weights are present.

Only Latin/digit fields are asserted: rendering Khmer correctly needs Pillow built with raqm
(complex-script shaping), and badly shaped Khmer would test the renderer, not the OCR.
Real-card accuracy is measured by spike/bench.py, not here.
"""

import time
from datetime import date

import cv2
import numpy as np
import pytest
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from khmer_kyc_ocr.config import Settings
from khmer_kyc_ocr.pipeline.template import TEMPLATE_V1
from khmer_kyc_ocr.pipeline.warp import CARD_H, CARD_W
from khmer_kyc_ocr.schemas import FieldName, IdCardResult
from khmer_kyc_ocr.service import IdCardReader
from tests.synthetic import photograph

pytest.importorskip("kiri_ocr")
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"

TEXT: dict[FieldName, str] = {
    "id_number": "010845612",
    "name_latin": "SOK DARA",
    "date_of_birth": "01.02.1990",
    "expiry_date": "05.03.2020 - 04.03.2030",
}


@pytest.fixture(scope="module")
def reader() -> IdCardReader:
    from khmer_kyc_ocr.pipeline.kiri import KiriRecognizer

    try:
        recognizer = KiriRecognizer()
    except Exception as exc:  # no cached weights and no network: skip, don't fail
        pytest.skip(f"Kiri weights unavailable: {exc}")
    settings = Settings.model_validate({"api_token": "t", "min_confidence": 0.0})
    return IdCardReader(recognizer, settings, today=lambda: date(2026, 9, 28))


def render_card() -> bytes:
    try:
        font = ImageFont.truetype(FONT_PATH, 34)
    except OSError:
        pytest.skip("font not available")
    card = PILImage.new("RGB", (CARD_W, CARD_H), (236, 236, 230))
    draw = ImageDraw.Draw(card)
    for name, text in TEXT.items():
        x0, y0, _, _ = TEMPLATE_V1[name].to_pixels(CARD_W, CARD_H)
        draw.text((x0 + 8, y0 + 12), text, fill=(20, 20, 20), font=font)
    bgr = np.asarray(card)[:, :, ::-1].copy()
    ok, buf = cv2.imencode(".jpg", photograph(bgr, angle_deg=185.0), [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return buf.tobytes()


def test_reads_latin_fields_end_to_end_from_upside_down_photo(reader: IdCardReader) -> None:
    started = time.perf_counter()
    result = reader.read(render_card())
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert isinstance(result, IdCardResult), result
    assert result.fields["id_number"].value == "010845612"
    assert result.fields["name_latin"].value == "SOK DARA"
    assert result.fields["date_of_birth"].value == "1990-02-01"
    assert result.fields["expiry_date"].value == "2030-03-04"
    print(f"\nend-to-end latency: {elapsed_ms:.0f} ms")  # visible with -s
