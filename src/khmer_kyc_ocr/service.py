"""Orchestrates one KYC read: decode, quality gate, flatten, OCR, extract, validate, face.

Stateless per call: every intermediate image is a local variable, so concurrent requests
cannot see each other's data. The only shared objects are the OCR provider and the face
detector, which must be thread-safe.

Stage order (mirrors the documented pipeline):

    Image -> QualityDetector -> DocumentDetector -> ImagePreprocessor
          -> OCR -> FieldExtractor -> Validator -> KYCResult
"""

import logging
import time
from collections.abc import Callable
from datetime import date

import numpy as np

from khmer_kyc_ocr.config import Settings
from khmer_kyc_ocr.pipeline.decode import decode_image
from khmer_kyc_ocr.pipeline.document import DocumentDetector
from khmer_kyc_ocr.pipeline.extract import FieldExtractor
from khmer_kyc_ocr.pipeline.face import FaceDetector
from khmer_kyc_ocr.pipeline.ocr import OCRProvider
from khmer_kyc_ocr.pipeline.preprocess import improve_readability
from khmer_kyc_ocr.pipeline.quality import QualityDetector
from khmer_kyc_ocr.pipeline.validate import Validator
from khmer_kyc_ocr.pipeline.warp import Image
from khmer_kyc_ocr.schemas import FieldName, FieldResult, KYCResult, RejectedImage

logger = logging.getLogger(__name__)

# Below this many valid fields, the card is probably upside-down (or unreadable), so we
# try the 180° flip. An upright ID card yields several valid fields; garbled upside-down
# text rarely yields more than one accidental match.
_MIN_VALID_FIELDS_FOR_ORIENTATION = 2


class KYCPipeline:
    """Runs the full document -> structured-KYC pipeline for one image."""

    def __init__(
        self,
        ocr: OCRProvider,
        settings: Settings,
        *,
        extractor: FieldExtractor | None = None,
        face_detector: FaceDetector | None = None,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._ocr = ocr
        self._settings = settings
        self._extractor = extractor or _default_extractor()
        self._face = self._build_face_detector(face_detector)
        self._validator = Validator(min_confidence=settings.min_confidence, today=today)
        self._quality = QualityDetector(
            min_short_side_px=settings.min_short_side_px,
            min_sharpness=settings.min_sharpness,
            min_brightness=settings.min_brightness,
            max_glare_ratio=settings.max_glare_ratio,
        )
        self._document = DocumentDetector()

    def _build_face_detector(self, injected: FaceDetector | None) -> FaceDetector | None:
        if not self._settings.face_detection:
            return None
        # A test-injected detector wins; otherwise build the default offline YuNet detector.
        return injected if injected is not None else FaceDetector()

    def read(self, data: bytes) -> KYCResult | RejectedImage:
        started = time.perf_counter()
        outcome = self._read(data)
        # Log the outcome and timing only: never image bytes or field values (personal data).
        logger.info(
            "kyc read outcome=%s ms=%.0f",
            outcome.reason if isinstance(outcome, RejectedImage) else "ok",
            (time.perf_counter() - started) * 1000,
        )
        return outcome

    def _read(self, data: bytes) -> KYCResult | RejectedImage:
        image = decode_image(data)
        if image is None:
            return RejectedImage(reason="undecodable", detail="Not a JPEG, PNG or WebP image.")

        if min(image.shape[:2]) < self._settings.min_short_side_px:
            min_side = self._settings.min_short_side_px
            return RejectedImage(
                reason="too_small",
                detail=f"Photo must be at least {min_side}px on its short side.",
            )

        card = self._document.find_card(image)
        if card is None:
            return RejectedImage(
                reason="card_not_found",
                detail="Place the whole card inside the frame on a plain, darker background.",
            )

        quality = self._quality.check(card)
        if quality.is_blurry:
            return RejectedImage(
                reason="blurry", detail="Hold the phone steady and retake the photo."
            )
        if quality.is_glared:
            return RejectedImage(reason="glare", detail="Move out of direct light to avoid glare.")
        if quality.is_dark:
            return RejectedImage(reason="dark", detail="The photo is too dark; add more light.")

        readable = improve_readability(card)
        fields, reasons = self._read_oriented(readable)

        face = self._face.detect(card) if self._face is not None else None

        return KYCResult(
            fields=fields,
            confidence={name: f.confidence for name, f in fields.items()},
            face=face,
            quality=quality,
            ocr_engine=self._ocr.engine,
            model_revision=self._ocr.model_revision,
            needs_review=bool(reasons),
            review_reasons=reasons,
        )

    def _read_oriented(self, readable: Image) -> tuple[dict[FieldName, FieldResult], list[str]]:
        """OCR + extract, resolving the 180° ambiguity with content.

        A rectangle is 180°-symmetric, so geometry can't tell upright from upside-down.
        We read the card, and if the result looks garbled (too few valid fields), we read
        the 180°-flipped card and keep whichever orientation produced more valid fields.
        """
        fields, reasons = self._extract_and_validate(readable)
        if self._valid_field_count(fields) >= _MIN_VALID_FIELDS_FOR_ORIENTATION:
            return fields, reasons

        flipped = np.ascontiguousarray(np.rot90(readable, 2))
        flipped_fields, flipped_reasons = self._extract_and_validate(flipped)
        if self._valid_field_count(flipped_fields) > self._valid_field_count(fields):
            return flipped_fields, flipped_reasons
        return fields, reasons

    def _extract_and_validate(
        self, readable: Image
    ) -> tuple[dict[FieldName, FieldResult], list[str]]:
        lines = self._ocr.read_lines(readable)
        extracted = self._extractor.extract(lines)
        return self._validator.validate(extracted)

    @staticmethod
    def _valid_field_count(fields: dict[FieldName, FieldResult]) -> int:
        return sum(1 for f in fields.values() if f.valid)


def _default_extractor() -> FieldExtractor:
    from khmer_kyc_ocr.pipeline.extract import RuleBasedFieldExtractor

    return RuleBasedFieldExtractor()
