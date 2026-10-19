"""Document detection: locate the identity document in a photo and flatten it.

Thin stage wrapper around the geometric detection in ``warp``, so the pipeline stages
are explicit and the underlying detector can be swapped (e.g. for a learned detector)
without touching the orchestration.
"""

import numpy as np

from khmer_kyc_ocr.pipeline import warp
from khmer_kyc_ocr.pipeline.warp import Frame, Image


class DocumentDetector:
    """Finds the largest quadrilateral document and flattens it to a fixed landscape card."""

    def find_card(self, image: Image) -> Image | None:
        card = warp.find_card(image)
        # Guarantee contiguous memory: downstream line detection slices and OCR copies it.
        return None if card is None else np.ascontiguousarray(card)

    def find_card_in_frame(self, image: Image, frame: Frame) -> Image | None:
        """The card from the caller's guide frame, when its edges were not found."""
        card = warp.find_card_in_frame(image, frame)
        return None if card is None else np.ascontiguousarray(card)
