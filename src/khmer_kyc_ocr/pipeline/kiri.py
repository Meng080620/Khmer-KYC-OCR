"""Kiri OCR adapter (kiri-ocr 0.2.15). Imports torch, so it is loaded lazily by api.py.

Kiri's confidence is a heuristic (mean per-character max probability blended with the beam
decoder score), not a calibrated probability: perfectly read clean lines score ~0.44 with the
pinned weights. Tune KYC_OCR_MIN_CONFIDENCE from spike/bench.py output, never by intuition.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from kiri_ocr import OCR
from kiri_ocr.model import preprocess_pil
from PIL import Image as PILImage

from khmer_kyc_ocr.pipeline.recognize import Recognition
from khmer_kyc_ocr.pipeline.warp import Image

# Pinned to a commit so a model update upstream can't silently change results in production.
# Weights are fetched once at image build time (see Dockerfile) and read from the local
# Hugging Face cache afterwards; HF_HUB_OFFLINE=1 forbids any runtime download.
#
# Why 2026-01-30 and not the latest commit: from 2026-02-06 the repo's model.safetensors has a
# 384-wide decoder but model_meta.json omits the decoder dims, so kiri-ocr 0.2.15 builds a
# 256-wide model and exits on the shape mismatch; the repo's model.pt holds only optimizer
# state. Re-check when upgrading either the weights or kiri-ocr.
MODEL_REPO = "mrrtmob/kiri-ocr"
MODEL_REVISION = "7661ff6f27cb381cd5928f5ca80252ec135955f4"
# model_meta.json carries the input size; without it Kiri silently falls back to defaults.
MODEL_FILES = ("model.safetensors", "model_meta.json", "vocab.json")


def fetch_model(repo: str = MODEL_REPO, revision: str = MODEL_REVISION) -> Path:
    """Download (or read from cache) the pinned weights and return the local model path.

    Done here rather than by Kiri because Kiri's own loader swallows download errors with
    bare excepts and later fails with a misleading "file not found".
    """
    paths = [hf_hub_download(repo, name, revision=revision) for name in MODEL_FILES]
    return Path(paths[0])


class KiriRecognizer:
    # Exposed so the OCR provider can record the pinned weights on every result.
    model_revision: str | None = MODEL_REVISION

    def __init__(self, model_path: Path | None = None, device: str = "cpu") -> None:
        path = model_path or fetch_model()
        # "accurate" = beam search: slower than greedy CTC, better on stacked Khmer subscripts.
        # Do NOT switch to "decoder": Kiri implements it by mutating the shared cfg.BEAM,
        # which races when the API's threadpool calls read_line concurrently.
        self._ocr = OCR(model_path=str(path), device=device, decode_method="accurate")

    def read_line(self, crop: Image) -> Recognition:
        if crop.size == 0:
            return Recognition("", 0.0)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        if float(np.mean(gray)) < 127:  # Kiri expects dark text on a light background
            gray = 255 - gray
        tensor = preprocess_pil(self._ocr.cfg, PILImage.fromarray(gray))
        # Kiri runs its encoder outside inference mode; without this every call builds an
        # autograd graph (extra memory and latency) that is never used.
        with torch.inference_mode():
            text, confidence = self._ocr.recognize_region(tensor)
        return Recognition(str(text).strip(), float(confidence))
