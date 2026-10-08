"""Quality gates: blur, glare, darkness and the combined pass/fail decision."""

import cv2
import numpy as np

from khmer_kyc_ocr.pipeline.quality import QualityDetector, brightness, glare_ratio, sharpness
from khmer_kyc_ocr.pipeline.warp import CARD_H, CARD_W
from tests.synthetic import make_card


def _detector(
    *,
    min_sharpness: float = 60.0,
    min_brightness: float = 50.0,
    max_glare_ratio: float = 0.25,
) -> QualityDetector:
    return QualityDetector(
        min_short_side_px=600,
        min_sharpness=min_sharpness,
        min_brightness=min_brightness,
        max_glare_ratio=max_glare_ratio,
    )


def test_clean_card_passes() -> None:
    report = _detector().check(make_card(seed=1))
    assert report.passed
    assert not report.is_blurry and not report.is_dark and not report.is_glared


def test_blurred_card_is_blurry() -> None:
    blurred = cv2.GaussianBlur(make_card(seed=1), (0, 0), 5)
    report = _detector(min_sharpness=60.0).check(np.asarray(blurred, dtype=np.uint8))
    assert report.is_blurry and not report.passed


def test_dark_card_is_dark() -> None:
    dark = np.zeros((CARD_H, CARD_W, 3), dtype=np.uint8)
    dark.fill(60)
    report = _detector(min_brightness=120.0).check(dark)
    assert report.is_dark and not report.passed


def test_glared_card_is_glared() -> None:
    card = make_card(seed=1)
    card[:, : CARD_W // 2] = 255
    report = _detector(max_glare_ratio=0.25).check(card)
    assert report.is_glared and not report.passed


def test_brightness_and_sharpness_metrics_are_sane() -> None:
    card = make_card(seed=1)
    assert 0.0 <= brightness(card) <= 255.0
    assert sharpness(card) > 0.0
    assert 0.0 <= glare_ratio(card) <= 1.0
