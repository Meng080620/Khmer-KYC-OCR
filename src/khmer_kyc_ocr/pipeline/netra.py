"""Netra-OCR adapter (netra-ocr 1.1.0, model Darayut/khmer-text-recognition).

A Khmer + English text-line recogniser, read the same way as Kiri: one cropped line in,
its text out. Imports torch, so it is loaded lazily by api.py.

Things to know before choosing it over Kiri:
- It returns text only, no confidence. Every line it reads is given ``CONFIDENCE``, a
  fixed stand-in, so a low-confidence gate does nothing for it: validation (formats,
  check digits) is what catches a misread.
- The weights are MIT, but part of the training data (KhmerSynthetic1M) is licensed for
  research and academic use only. Decide whether that suits a commercial service.
- The weights (about 77 MB) are downloaded from the Hugging Face Hub on first use and
  cached; they are not pinned to a commit the way Kiri's are.
- :class:`NetraOCRProvider` also finds the lines with the package's YOLO detector, which
  needs ``pip install "netra-ocr[yolo]"``. That pulls in ultralytics, which is AGPL-3.0:
  another thing to weigh before a commercial service ships it.
"""

import threading
from typing import Any

import cv2
import numpy as np
from PIL import Image as PILImage

from khmer_kyc_ocr.pipeline.ocr import OcrLine
from khmer_kyc_ocr.pipeline.recognize import Recognition
from khmer_kyc_ocr.pipeline.warp import Image

MODEL_REPO = "Darayut/khmer-text-recognition"
# Not a measurement: the model gives none. High enough to pass the default gate.
CONFIDENCE = 0.9
# Fewer lines than this is not a card (a logo, a stray word).
_MIN_LINES_FOR_A_CARD = 4
# A line with no other line within this share of the photo does not belong to the card.
_ALONE_BEYOND = 0.3
_REGION_MARGIN = 0.06
_SAME_LINE_OVERLAP = 0.6
_MAX_GAP_IN_HEIGHTS = 1.2


class NetraRecognizer:
    model_revision: str | None = None

    def __init__(self, beam_width: int = 1) -> None:
        # Greedy decoding (width 1). Measured on a real card on an 8-core laptop CPU: 0.12s
        # a line, against 9s a line at the package's default of 3, for all but a handful of
        # the same characters. The model card puts the gain from beam search at about a
        # quarter of a point of error rate.
        from netra_ocr.recognition.recognize_text import _get_predictor

        self._predictor = _get_predictor()
        self._beam_width = beam_width
        # The predictor keeps no per-call state we know of, but it is not documented as
        # thread-safe and the API reads from several threads: one line at a time.
        self._lock = threading.Lock()

    def read_line(self, crop: Image) -> Recognition:
        if crop.size == 0:
            return Recognition("", 0.0)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        if float(np.mean(gray)) < 127:  # trained on dark text on a light background
            gray = 255 - gray
        with self._lock:
            text = str(self._predictor.predict(PILImage.fromarray(gray), self._beam_width))
        text = text.strip()
        return Recognition(text, CONFIDENCE if text else 0.0)


class NetraOCRProvider:
    """Finds the text lines with Netra's YOLO detector and reads each with its recogniser.

    Used instead of :class:`~khmer_kyc_ocr.pipeline.ocr.LocalOCRProvider` because that
    one finds lines by projection profiles, which works on a clean synthetic card and
    finds nothing on a photographed one: a real card's patterned background leaves ink
    in every row, so no row gap ever separates two lines.
    """

    engine = "netra-ocr"
    model_revision: str | None = None

    def __init__(self, recognizer: NetraRecognizer | None = None) -> None:
        from netra_ocr.detectors import get_detector

        self._detector: Any = get_detector("yolo")
        self._recognizer = recognizer or NetraRecognizer()
        self._lock = threading.Lock()

    def text_region(self, image: Image) -> tuple[int, int, int, int] | None:
        """Where the printed text is in a whole photo: (x0, y0, x1, y1), with a margin.

        A card whose edges cannot be found (light card, light table) is still where its
        text is. The lines found here are small and only used to locate it; the region is
        then cut from the full-size photo and read properly.
        """
        height, width = image.shape[:2]
        boxes = self._text_boxes(image)
        if len(boxes) < _MIN_LINES_FOR_A_CARD:
            return None
        xs0, ys0, xs1, ys1 = (np.array(side, dtype=np.float64) for side in zip(*boxes, strict=True))
        # Drop a line that stands alone, far from every other (print on the table, a
        # reflection). Measured to its nearest neighbour, not to the middle of the text: a
        # card that fills the photo has lines a long way from the middle that belong to it.
        cx, cy = (xs0 + xs1) / 2, (ys0 + ys1) / 2
        apart = np.hypot((cx[:, None] - cx[None, :]) / width, (cy[:, None] - cy[None, :]) / height)
        np.fill_diagonal(apart, np.inf)
        keep = apart.min(axis=1) < _ALONE_BEYOND
        if int(keep.sum()) < _MIN_LINES_FOR_A_CARD:
            return None
        x0, y0 = float(xs0[keep].min()), float(ys0[keep].min())
        x1, y1 = float(xs1[keep].max()), float(ys1[keep].max())
        mx, my = (x1 - x0) * _REGION_MARGIN, (y1 - y0) * _REGION_MARGIN
        return (
            max(0, round(x0 - mx)),
            max(0, round(y0 - my)),
            min(width, round(x1 + mx)),
            min(height, round(y1 + my)),
        )

    def _text_boxes(self, image: Image) -> list[tuple[int, int, int, int]]:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
        with self._lock:  # ultralytics models are not safe to call from several threads
            found = self._detector.detect(np.ascontiguousarray(bgr))
        return [line.bbox for line in found if line.label == "text"]

    def read_lines(self, image: Image) -> list[OcrLine]:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
        with self._lock:  # ultralytics models are not safe to call from several threads
            found = self._detector.detect(np.ascontiguousarray(bgr))
        boxes = _join_pieces_of_a_line([line.bbox for line in found if line.label == "text"])
        lines: list[OcrLine] = []
        for number, (x1, y1, x2, y2) in enumerate(boxes, 1):
            crop = np.ascontiguousarray(image[y1:y2, x1:x2])
            read = self._recognizer.read_line(crop)
            if not read.text:
                continue
            lines.append(
                OcrLine(
                    text=read.text,
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    confidence=read.confidence,
                    line_number=number,
                )
            )
        return lines


Box = tuple[int, int, int, int]  # x1, y1, x2, y2


def _join_pieces_of_a_line(boxes: list[Box]) -> list[Box]:
    """Boxes that sit side by side on one printed line, joined into one.

    The detector often returns a label and its value, or two words of a name, as separate
    boxes. The recogniser reads a whole line better than its pieces, and the extractor
    needs a label and its value together. Two boxes are one line when they overlap for
    most of their height and the gap between them is less than a character or two wide.
    """
    lines = [list(box) for box in boxes]
    # Until nothing joins any more: a value grows as its own pieces join, and only then
    # reaches back to its label.
    while True:
        joined: list[list[int]] = []
        for x1, y1, x2, y2 in sorted(lines, key=lambda box: ((box[1] + box[3]) / 2, box[0])):
            for line in joined:
                lx1, ly1, lx2, ly2 = line
                overlap = min(y2, ly2) - max(y1, ly1)
                shorter = min(y2 - y1, ly2 - ly1)
                gap = max(x1, lx1) - min(x2, lx2)
                if overlap > shorter * _SAME_LINE_OVERLAP and gap < shorter * _MAX_GAP_IN_HEIGHTS:
                    line[0], line[1] = min(lx1, x1), min(ly1, y1)
                    line[2], line[3] = max(lx2, x2), max(ly2, y2)
                    break
            else:
                joined.append([x1, y1, x2, y2])
        if len(joined) == len(lines):
            break
        lines = joined
    return sorted(((a, b, c, d) for a, b, c, d in joined), key=lambda box: (box[1], box[0]))
