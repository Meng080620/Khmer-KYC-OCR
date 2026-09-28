from dataclasses import dataclass
from typing import Protocol

from khmer_kyc_ocr.pipeline.warp import Image


@dataclass(frozen=True, slots=True)
class Recognition:
    text: str
    confidence: float  # 0..1


class Recognizer(Protocol):
    """Reads a single cropped text line. Implementations must be safe to call from many threads."""

    def read_line(self, crop: Image) -> Recognition: ...
