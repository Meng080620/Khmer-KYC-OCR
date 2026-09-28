"""Orchestrates one ID-card read: decode, quality gate, flatten, orient, recognize, parse, decide.

Stateless per call: every intermediate image is a local variable, so concurrent requests
cannot see each other's data. The only shared object is the recognizer, which must be
thread-safe (see recognize.Recognizer).
"""

import logging
import time
from collections.abc import Callable
from datetime import date

import numpy as np

from khmer_kyc_ocr.config import Settings
from khmer_kyc_ocr.pipeline import parse
from khmer_kyc_ocr.pipeline.decode import decode_image
from khmer_kyc_ocr.pipeline.quality import sharpness, short_side
from khmer_kyc_ocr.pipeline.recognize import Recognition, Recognizer
from khmer_kyc_ocr.pipeline.template import TEMPLATE_V1, TEMPLATE_VERSION, crop
from khmer_kyc_ocr.pipeline.warp import Image, find_card
from khmer_kyc_ocr.schemas import FieldName, FieldResult, IdCardResult, RejectedImage

logger = logging.getLogger(__name__)

PARSERS: dict[FieldName, Callable[[str], parse.Parsed]] = {
    "id_number": parse.parse_id_number,
    "name_khmer": parse.parse_name_khmer,
    "name_latin": parse.parse_name_latin,
    "date_of_birth": parse.parse_date,
    "sex": parse.parse_sex,
    "expiry_date": parse.parse_expiry,
}


class IdCardReader:
    def __init__(
        self,
        recognizer: Recognizer,
        settings: Settings,
        *,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._recognizer = recognizer
        self._settings = settings
        self._today = today

    def read(self, data: bytes) -> IdCardResult | RejectedImage:
        started = time.perf_counter()
        outcome = self._read(data)
        # Log the outcome and timing only: never image bytes or field values (personal data).
        logger.info(
            "id-card read outcome=%s ms=%.0f",
            outcome.reason if isinstance(outcome, RejectedImage) else "ok",
            (time.perf_counter() - started) * 1000,
        )
        return outcome

    def _read(self, data: bytes) -> IdCardResult | RejectedImage:
        image = decode_image(data)
        if image is None:
            return RejectedImage(reason="undecodable", detail="Not a JPEG, PNG or WebP image.")
        min_side = self._settings.min_short_side_px
        if short_side(image) < min_side:
            return RejectedImage(
                reason="too_small", detail=f"Photo must be at least {min_side}px on its short side."
            )
        card = find_card(image)
        if card is None:
            return RejectedImage(
                reason="card_not_found",
                detail="Place the whole card inside the frame on a plain, darker background.",
            )
        if sharpness(card) < self._settings.min_sharpness:
            return RejectedImage(
                reason="blurry", detail="Hold the phone steady and retake the photo."
            )

        card, id_reading = self._orient(card)
        readings: dict[FieldName, Recognition] = {"id_number": id_reading}
        for name, roi in TEMPLATE_V1.items():
            if name != "id_number":
                readings[name] = self._recognizer.read_line(crop(card, roi))
        return self._decide(readings)

    def _orient(self, card: Image) -> tuple[Image, Recognition]:
        """Pick upright vs 180°-flipped by which orientation yields a valid ID number.

        Costs one extra recognizer call per request; cheaper and more reliable than a
        separate orientation classifier, because the ID number has a strict format.
        """
        roi = TEMPLATE_V1["id_number"]
        upright = self._recognizer.read_line(crop(card, roi))
        if parse.parse_id_number(upright.text).valid:
            return card, upright
        flipped_card: Image = np.ascontiguousarray(np.rot90(card, 2))
        flipped = self._recognizer.read_line(crop(flipped_card, roi))
        if parse.parse_id_number(flipped.text).valid or flipped.confidence > upright.confidence:
            return flipped_card, flipped
        return card, upright

    def _decide(self, readings: dict[FieldName, Recognition]) -> IdCardResult:
        fields: dict[FieldName, FieldResult] = {}
        reasons: list[str] = []
        for name, reading in readings.items():
            parsed = PARSERS[name](reading.text)
            confidence = min(max(reading.confidence, 0.0), 1.0)
            fields[name] = FieldResult(
                value=parsed.value, raw=reading.text, confidence=confidence, valid=parsed.valid
            )
            # Reasons name the field and the problem, never the value: they reach logs/tickets.
            if not parsed.valid:
                reasons.append(f"{name}: failed validation")
            elif confidence < self._settings.min_confidence:
                reasons.append(f"{name}: low confidence {confidence:.2f}")

        expiry = fields["expiry_date"].value
        if expiry is not None and date.fromisoformat(expiry) < self._today():
            reasons.append("expiry_date: card expired")

        return IdCardResult(
            template_version=TEMPLATE_VERSION,
            fields=fields,
            needs_review=bool(reasons),
            review_reasons=reasons,
        )
