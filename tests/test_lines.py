"""Text-line detection must find whatever text is printed, independent of layout."""

import cv2
import numpy as np
import pytest

from khmer_kyc_ocr.pipeline.lines import detect_text_lines
from khmer_kyc_ocr.pipeline.warp import CARD_H, CARD_W


def _render(text_lines: list[str], font_scale: float = 1.2) -> np.ndarray:
    card = np.full((CARD_H, CARD_W, 3), 235, dtype=np.uint8)
    for i, text in enumerate(text_lines):
        y = 60 + i * 120
        cv2.putText(card, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), 3)
    return card


def test_detects_each_rendered_line() -> None:
    lines = ["FIRST LINE", "SECOND LINE", "THIRD LINE"]
    boxes = detect_text_lines(_render(lines))
    assert len(boxes) >= len(lines)
    ys = sorted({b[1] for b in boxes})
    assert ys == sorted(ys)


def test_empty_card_yields_no_lines() -> None:
    card = np.full((CARD_H, CARD_W, 3), 235, dtype=np.uint8)
    assert detect_text_lines(card) == []


def test_boxes_are_within_card_bounds() -> None:
    boxes = detect_text_lines(_render(["HELLO WORLD"]))
    for x, y, w, h in boxes:
        assert x >= 0 and x + w <= CARD_W
        assert y >= 0 and y + h <= CARD_H
        assert w > 0 and h > 0


def test_separates_two_spaced_lines() -> None:
    card = np.full((CARD_H, CARD_W, 3), 235, dtype=np.uint8)
    cv2.putText(card, "TOP", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
    cv2.putText(card, "BOTTOM", (40, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
    boxes = detect_text_lines(card)
    assert len(boxes) >= 2


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_finds_lines_on_rotated_cards(rotation: int) -> None:
    card = _render(["SOME TEXT"])
    if rotation in (90, 270):
        pytest.skip("90/270 are normalized away by document detection")
    rotated = np.ascontiguousarray(np.rot90(card, rotation // 90))
    assert detect_text_lines(rotated)
