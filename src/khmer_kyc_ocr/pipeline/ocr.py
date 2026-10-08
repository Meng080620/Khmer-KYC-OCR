"""OCR provider abstraction.

The pipeline depends only on the :class:`OCRProvider` protocol, so the engine can be
swapped without touching extraction, validation or the API:

- :class:`LocalOCRProvider` — offline Kiri OCR (Khmer + English), the default.
- :class:`CloudOCRProvider` — a self-hosted or vendor OCR service over HTTPS, disabled
  by default because it sends identity data off the device/network boundary.
- ``AIExtractionProvider`` (in ``extract.py``) — a text-to-fields step that can run an
  LLM instead of the rule-based extractor; also disabled by default for the same reason.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from khmer_kyc_ocr.pipeline.lines import detect_text_lines
from khmer_kyc_ocr.pipeline.recognize import Recognizer
from khmer_kyc_ocr.pipeline.warp import Image


@dataclass(frozen=True, slots=True)
class OcrLine:
    """One recognised text line with its position on the flattened card."""

    text: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    det_confidence: float = 1.0
    line_number: int = 0


class OCRProvider(Protocol):
    """Recognises all text lines on a flattened document image. Thread-safe."""

    engine: str
    model_revision: str | None

    def read_lines(self, image: Image) -> list[OcrLine]: ...


class LocalOCRProvider:
    """Offline provider: OpenCV line detection + Kiri line recognition (CPU)."""

    engine = "kiri-ocr"

    def __init__(self, recognizer: Recognizer, model_revision: str | None = None) -> None:
        self._recognizer = recognizer
        self.model_revision = model_revision

    def read_lines(self, image: Image) -> list[OcrLine]:
        boxes = detect_text_lines(image)
        lines: list[OcrLine] = []
        for i, (x, y, w, h) in enumerate(boxes, 1):
            crop = np.ascontiguousarray(image[y : y + h, x : x + w]).copy()
            if crop.size == 0:
                continue
            recognition = self._recognizer.read_line(crop)
            if not recognition.text.strip():
                continue
            lines.append(
                OcrLine(
                    text=recognition.text.strip(),
                    bbox=(x, y, w, h),
                    confidence=min(max(recognition.confidence, 0.0), 1.0),
                    line_number=i,
                )
            )
        return lines


class CloudOCRProvider:
    """Sends the flattened card to a configurable OCR endpoint over HTTPS and returns lines.

    Disabled by default: sending an identity document to a third party is a privacy
    decision that must be made explicitly. When enabled, configure ``KYC_OCR_CLOUD_URL``
    and ``KYC_OCR_CLOUD_TOKEN``; the request is authenticated with a bearer token and the
    connection is TLS.
    """

    engine = "cloud-ocr"
    model_revision: str | None = None

    def __init__(self, url: str, token: str, timeout_s: float = 30.0) -> None:
        import httpx

        self._client = httpx.Client(
            base_url=url, headers={"Authorization": f"Bearer {token}"}, timeout=timeout_s
        )

    def read_lines(self, image: Image) -> list[OcrLine]:
        import cv2

        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return []
        response = self._client.post(
            "/lines", content=buf.tobytes(), headers={"Content-Type": "image/jpeg"}
        )
        response.raise_for_status()
        payload = response.json()
        return [
            OcrLine(
                text=str(item["text"]).strip(),
                bbox=tuple(int(v) for v in item["bbox"]),  # type: ignore[arg-type]
                confidence=float(item.get("confidence", 0.0)),
            )
            for item in payload.get("lines", [])
        ]
