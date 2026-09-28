import cv2
import numpy as np

from khmer_kyc_ocr.pipeline.warp import Image


def decode_image(data: bytes) -> Image | None:
    """Decode JPEG/PNG/WebP bytes in memory. Never touches disk.

    IMREAD_COLOR applies the EXIF orientation tag, so phone photos come out the right way up.
    """
    if not data:
        return None
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return None if image is None else np.asarray(image, dtype=np.uint8)
