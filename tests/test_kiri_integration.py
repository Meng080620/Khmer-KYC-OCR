"""Real model, full pipeline, synthetic card. Skipped unless kiri-ocr and its weights are present.

Only Latin/digit fields are asserted: rendering Khmer correctly needs Pillow built with raqm
(complex-script shaping), and badly shaped Khmer would test the renderer, not the OCR.
Real-card accuracy is measured by spike/bench.py, not here.
"""

import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pytest

from khmer_kyc_ocr.config import Settings
from khmer_kyc_ocr.pipeline.kiri import KiriRecognizer
from khmer_kyc_ocr.pipeline.ocr import LocalOCRProvider
from khmer_kyc_ocr.pipeline.warp import CARD_H, CARD_W
from khmer_kyc_ocr.schemas import KYCResult
from khmer_kyc_ocr.service import KYCPipeline
from tests.synthetic import photograph

pytest.importorskip("kiri_ocr")
FONT_PATHS = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)

TEXT_LINES = [
    "National Identity Card",
    "ID Number: 010845612",
    "Surname: SOK DARA",
    "Date of Birth: 01.02.1990",
    "Validity: 05.03.2020 - 04.03.2030",
]


@pytest.fixture(scope="module")
def pipeline() -> KYCPipeline:
    try:
        recognizer = KiriRecognizer()
    except Exception as exc:
        pytest.skip(f"Kiri weights unavailable: {exc}")
    provider = LocalOCRProvider(recognizer, model_revision=recognizer.model_revision)
    settings = Settings.model_validate(
        {"api_token": "t", "min_confidence": 0.0, "face_detection": False}
    )
    return KYCPipeline(provider, settings, today=lambda: date(2026, 9, 28))


def render_card() -> bytes:
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont

    font_path = next((p for p in FONT_PATHS if Path(p).exists()), None)
    if font_path is None:
        pytest.skip("no Latin TrueType font available")
    font = ImageFont.truetype(font_path, 40)
    card = PILImage.new("RGB", (CARD_W, CARD_H), (236, 236, 230))
    draw = ImageDraw.Draw(card)
    for i, text in enumerate(TEXT_LINES):
        draw.text((60, 50 + i * 110), text, fill=(20, 20, 20), font=font)
    bgr = np.asarray(card)[:, :, ::-1].copy()
    ok, buf = cv2.imencode(".jpg", photograph(bgr, angle_deg=185.0), [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return buf.tobytes()


def test_reads_latin_fields_end_to_end_from_upside_down_photo(pipeline: KYCPipeline) -> None:
    started = time.perf_counter()
    result = pipeline.read(render_card())
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert isinstance(result, KYCResult), result
    assert result.fields["id_number"].value == "010845612"
    assert result.fields["name_en"].value == "SOK DARA"
    assert result.fields["date_of_birth"].value == "1990-02-01"
    assert result.fields["issue_date"].value == "2020-03-05"
    assert result.fields["expiry_date"].value == "2030-03-04"
    assert result.fields["document_type"].value == "national_id"
    print(f"\nend-to-end latency: {elapsed_ms:.0f} ms")
