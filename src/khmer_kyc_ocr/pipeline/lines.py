"""Layout-agnostic text-line detection.

Unlike the old fixed template, this finds *whatever* text lines are printed on the
flattened card, so the field extractor can work on any document layout (and any card
series) rather than assuming boxes. It runs fully in memory on the flattened image.

Implementation: binarise, then use horizontal and vertical projection profiles to
segment text bands into line boxes. Deterministic and offline — no extra model.
"""

import cv2
import numpy as np

from khmer_kyc_ocr.pipeline.warp import Image

MIN_LINE_HEIGHT = 14
MIN_LINE_WIDTH = 24
MAX_LINE_HEIGHT = 120
_GAP_MERGE_FRACTION = 0.8
_BORDER_INSET_FRACTION = 0.01
_LINE_PADDING = 6


def _binarize(gray: Image) -> Image:
    """Return a binary image where text is white on black, denoised."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return np.asarray(opened, dtype=np.uint8)


def _runs(mask: np.ndarray, *, min_run: int) -> list[tuple[int, int]]:
    """Contiguous [start, end] index ranges where ``mask`` is True, merged if the gap
    between them is small relative to the run height."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev <= max(1, int(min_run * _GAP_MERGE_FRACTION)):
            prev = i
            continue
        runs.append((start, prev))
        start = prev = i
    runs.append((start, prev))
    return runs


def detect_text_lines(image: Image) -> list[tuple[int, int, int, int]]:
    """Return (x, y, w, h) boxes for every detected text line, top-to-bottom then left-right."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = np.asarray(gray, dtype=np.uint8)
    h, w = gray.shape
    inset = max(2, int(min(h, w) * _BORDER_INSET_FRACTION))
    inner = gray[inset : h - inset, inset : w - inset]
    binary = _binarize(inner)

    row_ink = binary.sum(axis=1) > 0
    bands = _runs(row_ink, min_run=MIN_LINE_HEIGHT)

    lines: list[tuple[int, int, int, int]] = []
    for y0, y1 in bands:
        height = y1 - y0 + 1
        if not MIN_LINE_HEIGHT <= height <= MAX_LINE_HEIGHT:
            continue
        col_ink = binary[y0 : y1 + 1, :].sum(axis=0) > 0
        columns = _runs(col_ink, min_run=MIN_LINE_WIDTH)
        for x0, x1 in columns:
            width = x1 - x0 + 1
            if width < MIN_LINE_WIDTH:
                continue
            px0 = max(0, x0 + inset - _LINE_PADDING)
            py0 = max(0, y0 + inset - _LINE_PADDING)
            px1 = min(w, x1 + inset + 1 + _LINE_PADDING)
            py1 = min(h, y1 + inset + 1 + _LINE_PADDING)
            lines.append((px0, py0, px1 - px0, py1 - py0))
    return sorted(lines, key=lambda b: (b[1], b[0]))
