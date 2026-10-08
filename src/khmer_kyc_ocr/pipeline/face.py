"""Face/portrait detection on the flattened identity document.

Uses OpenCV's YuNet face detector (a small ONNX model bundled with the package), so it
runs fully offline with no third-party service. The goal for the MVP is to locate the
portrait area and extract it as a separate image so the architecture is ready for
selfie-to-ID matching later. Detection here is *not* liveness and *not* identity
verification; it only says "a face-like region exists here".
"""

import base64
from importlib.resources import files

import cv2
import numpy as np

from khmer_kyc_ocr.pipeline.warp import Image
from khmer_kyc_ocr.schemas import FaceResult

_MODEL_NAME = "face_detection_yunet_2023mar.onnx"


def _default_model_path() -> str:
    return str(files("khmer_kyc_ocr").joinpath("data", _MODEL_NAME))


class FaceDetector:
    def __init__(self, model_path: str | None = None, score_threshold: float = 0.9) -> None:
        # The installed cv2 stub predates YuNet, so the attribute is unknown to mypy.
        self._detector = cv2.FaceDetectorYN_create(  # type: ignore[attr-defined]
            model_path or _default_model_path(), "", (320, 320), score_threshold, 0.3, 5000
        )

    def detect(self, card: Image) -> FaceResult:
        h, w = card.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(card)
        if faces is None or len(faces) == 0:
            return FaceResult(detected=False)

        best = max(faces, key=lambda f: float(f[2]) * float(f[3]))
        x, y, fw, fh = (round(float(v)) for v in best[:4])
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + fw), min(h, y + fh)
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            return FaceResult(detected=False)

        crop = np.ascontiguousarray(card[y0:y1, x0:x1]).copy()
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        image_b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
        return FaceResult(detected=True, bbox=(x0, y0, x1 - x0, y1 - y0), image_base64=image_b64)
