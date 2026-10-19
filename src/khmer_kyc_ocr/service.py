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
from itertools import pairwise

import cv2
import numpy as np

from khmer_kyc_ocr.config import Settings
from khmer_kyc_ocr.pipeline.decode import decode_image
from khmer_kyc_ocr.pipeline.document import DocumentDetector
from khmer_kyc_ocr.pipeline.extract import FieldExtractor
from khmer_kyc_ocr.pipeline.face import FaceDetector
from khmer_kyc_ocr.pipeline.lines import detect_text_lines
from khmer_kyc_ocr.pipeline.mrz import (
    IdCardMrz,
    MrzField,
    PassportMrz,
    find_td1,
    find_td3,
    parse_td1,
    parse_td3,
)
from khmer_kyc_ocr.pipeline.ocr import OcrLine, OCRProvider
from khmer_kyc_ocr.pipeline.preprocess import improve_readability
from khmer_kyc_ocr.pipeline.quality import QualityDetector
from khmer_kyc_ocr.pipeline.validate import Validator
from khmer_kyc_ocr.pipeline.warp import CARD_W, Frame, Image, frame_region
from khmer_kyc_ocr.schemas import (
    FieldName,
    FieldResult,
    KYCResult,
    PassportFieldName,
    PassportResult,
    RejectedImage,
)

logger = logging.getLogger(__name__)

# Below this many valid fields, the card is probably upside-down (or unreadable), so we
# try the 180° flip. An upright ID card yields several valid fields; garbled upside-down
# text rarely yields more than one accidental match.
_MIN_VALID_FIELDS_FOR_ORIENTATION = 2

_FRAME_REASON = "card edges not found: read from where the text is or the guide frame"
# The widest a card found by its text is read at.
_READING_WIDTH = 1600
# Wide enough that each of the zone's 44 characters is some 25px across.
_PASSPORT_WIDTH = 1400


# A row this much of the page's width may be a zone line.
_LONG_ROW_FRACTION = 0.6
# Roughly how many times its height one piece of a long row is: about ten characters.
_PIECE_ASPECT = 7.0
_PIECE_PAD = 12


def _rows_of(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Line boxes merged into whole printed rows: (x, y, w, h), top to bottom."""
    rows: list[list[int]] = []
    for x, y, w, h in sorted(boxes, key=lambda box: (box[1], box[0])):
        if rows and abs((y + h / 2) - (rows[-1][1] + rows[-1][3]) / 2) < h / 2:
            row = rows[-1]
            row[0], row[1] = min(row[0], x), min(row[1], y)
            row[2], row[3] = max(row[2], x + w), max(row[3], y + h)
        else:
            rows.append([x, y, x + w, y + h])
    return [(x0, y0, x1 - x0, y1 - y0) for x0, y0, x1, y1 in rows]


def _gaps_near_even_splits(row: Image, pieces: int) -> list[int]:
    """Where to cut a row into ``pieces``: the emptiest column near each even split, so a
    cut falls between two characters and not through one."""
    ink = (255 - row.astype(np.int32)).sum(axis=0)
    width = row.shape[1]
    reach = max(2, width // (pieces * 4))
    cuts: list[int] = []
    for i in range(1, pieces):
        centre = round(width * i / pieces)
        lo, hi = max(1, centre - reach), min(width - 1, centre + reach)
        cuts.append(lo + int(np.argmin(ink[lo:hi])))
    return cuts


def _fit_width(image: Image, width: int) -> Image:
    scale = width / image.shape[1]
    if abs(scale - 1) < 0.05:
        return image
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)
    return np.ascontiguousarray(resized, dtype=np.uint8)


def _passport_fields(
    mrz: PassportMrz,
) -> tuple[dict[PassportFieldName, FieldResult], list[str]]:
    """Each field with ``valid`` from its check digit; a field the zone cannot confirm
    (the names, the nationality) stands as long as the line as a whole checks out."""
    read: dict[PassportFieldName, tuple[str | None, str, bool | None]] = {
        "document_number": _triple(mrz.document_number),
        "surname": _triple(mrz.surname),
        "given_names": _triple(mrz.given_names),
        "nationality": _triple(mrz.nationality),
        "date_of_birth": _triple(mrz.date_of_birth),
        "gender": _triple(mrz.gender),
        "expiry_date": _triple(mrz.expiry_date),
    }
    fields: dict[PassportFieldName, FieldResult] = {}
    reasons: list[str] = []
    for name, (value, raw, valid) in read.items():
        if value is None and not raw.strip("<"):
            continue  # nothing printed there, e.g. no given names
        ok = bool(value) and (valid if valid is not None else True)
        fields[name] = FieldResult(
            value=value if ok else None, raw=raw, confidence=mrz.confidence, valid=ok
        )
        if not ok:
            reasons.append(f"{name}: check digit or format does not agree")
    if not mrz.composite_valid:
        reasons.append("machine-readable zone: overall check digit does not agree")
    return fields, reasons


def _with_the_zone(
    fields: dict[FieldName, FieldResult], reasons: list[str], mrz: IdCardMrz, today: date
) -> tuple[dict[FieldName, FieldResult], list[str]]:
    """The printed fields with the card's machine-readable zone laid over them.

    A zone value whose check digit agrees is more trustworthy than the same value read
    from the printed text, so it replaces it and clears that field's complaint. The Latin
    name has no check digit at all; it is taken from the zone anyway, since the printed
    one is in small type over a pattern.
    """
    merged = dict(fields)
    settled: set[str] = set()

    def take(name: FieldName, field: MrzField, value: str | None) -> None:
        if value and field.valid:
            merged[name] = FieldResult(
                value=value, raw=field.raw, confidence=mrz.confidence, valid=True
            )
            settled.add(name)

    take("id_number", mrz.document_number, mrz.document_number.value)
    take("date_of_birth", mrz.date_of_birth, mrz.date_of_birth.value)
    take("expiry_date", mrz.expiry_date, mrz.expiry_date.value)
    # One letter in a fixed place: taken even when the overall check digit was not read.
    if mrz.gender.value:
        merged["gender"] = FieldResult(
            value=mrz.gender.value, raw=mrz.gender.raw, confidence=mrz.confidence, valid=True
        )
        settled.add("gender")
    name = " ".join(part for part in (mrz.surname.value, mrz.given_names.value) if part)
    # The third line is outside every check digit, the overall one included, so there is
    # nothing to wait for: it is still the clearest print of the name on the card.
    if name:
        merged["name_en"] = FieldResult(
            value=name,
            raw=f"{mrz.surname.raw}<<{mrz.given_names.raw}",
            confidence=mrz.confidence,
            valid=True,
        )
        settled.add("name_en")

    # Only what the zone settles is cleared: how well a field was read. A card that has
    # expired is still expired.
    kept = [
        reason
        for reason in reasons
        if not (
            reason.split(":")[0] in settled
            and ("failed validation" in reason or "low confidence" in reason)
        )
    ]
    expired = "expiry_date: card expired"
    if (
        "expiry_date" in settled
        and expired not in kept
        and date.fromisoformat(str(merged["expiry_date"].value)) < today
    ):
        kept.append(expired)
    if not mrz.composite_valid:
        kept.append("machine-readable zone: overall check digit does not agree")
    return merged, kept


def _triple(field: MrzField) -> tuple[str | None, str, bool | None]:
    return field.value, field.raw, field.valid


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
        self._today = today
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

    def read(self, data: bytes, frame: Frame | None = None) -> KYCResult | RejectedImage:
        """Read one photo. ``frame`` is where the caller's guide frame was, as fractions of
        the photo; it is only used when the card's edges cannot be found."""
        started = time.perf_counter()
        outcome = self._read(data, frame)
        # Log the outcome and timing only: never image bytes or field values (personal data).
        logger.info(
            "kyc read outcome=%s ms=%.0f",
            outcome.reason if isinstance(outcome, RejectedImage) else "ok",
            (time.perf_counter() - started) * 1000,
        )
        return outcome

    def _read(self, data: bytes, frame: Frame | None) -> KYCResult | RejectedImage:
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
        from_frame = False
        if card is None:
            # No edges to find. Where the text is says where the card is, and does not
            # depend on the caller's frame being right; the frame is the last resort.
            card = self._card_from_its_text(image)
            from_frame = card is not None
        if card is None and frame is not None:
            card = self._document.find_card_in_frame(image, frame)
            from_frame = card is not None
        if card is None:
            return RejectedImage(
                reason="card_not_found",
                detail="Place the whole card inside the frame on a plain, darker background.",
            )

        # The gates were set on a card 1000 across; measured at that size whatever the
        # size it is read at.
        quality = self._quality.check(_fit_width(card, CARD_W))
        if quality.is_blurry:
            return RejectedImage(
                reason="blurry", detail="Hold the phone steady and retake the photo."
            )
        if quality.is_glared:
            return RejectedImage(reason="glare", detail="Move out of direct light to avoid glare.")
        if quality.is_dark:
            return RejectedImage(reason="dark", detail="The photo is too dark; add more light.")

        readable = improve_readability(card)
        fields, reasons, mrz = self._read_oriented(readable)

        if from_frame:
            # The crop may be loose or tilted, so a person should compare it with the photo.
            reasons = [*reasons, _FRAME_REASON]

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
            mrz=list(mrz.lines) if mrz is not None else None,
        )

    def read_passport(
        self, data: bytes, frame: Frame | None = None
    ) -> PassportResult | RejectedImage:
        """Read a passport's data page. Only the machine-readable zone is used: its two
        lines carry every field, each with a check digit.

        ``frame`` is where the caller's guide frame was. The page is not flattened (a
        passport lies in an open booklet with no clean edge to find), so it should be
        photographed level; the zone is then found by its content.
        """
        started = time.perf_counter()
        outcome = self._read_passport(data, frame)
        logger.info(
            "passport read outcome=%s ms=%.0f",
            outcome.reason if isinstance(outcome, RejectedImage) else "ok",
            (time.perf_counter() - started) * 1000,
        )
        return outcome

    def _read_passport(self, data: bytes, frame: Frame | None) -> PassportResult | RejectedImage:
        image = decode_image(data)
        if image is None:
            return RejectedImage(reason="undecodable", detail="Not a JPEG, PNG or WebP image.")
        page = frame_region(image, frame) if frame is not None else image
        if min(page.shape[:2]) < self._settings.min_short_side_px:
            min_side = self._settings.min_short_side_px
            return RejectedImage(
                reason="too_small",
                detail=f"The passport page must be at least {min_side}px on its short side.",
            )
        page = _fit_width(page, _PASSPORT_WIDTH)

        quality = self._quality.check(page)
        if quality.is_blurry:
            return RejectedImage(
                reason="blurry", detail="Hold the phone steady and retake the photo."
            )
        if quality.is_dark:
            return RejectedImage(reason="dark", detail="The photo is too dark; add more light.")

        readable = improve_readability(page)
        mrz = self._passport_mrz(readable)
        if mrz is None:
            return RejectedImage(
                reason="mrz_not_found",
                detail="Fit the whole photo page in the frame, with the two lines of <<< at "
                "the bottom, and hold the phone level.",
            )
        fields, reasons = _passport_fields(mrz)
        return PassportResult(
            fields=fields,
            quality=quality,
            ocr_engine=self._ocr.engine,
            model_revision=self._ocr.model_revision,
            needs_review=bool(reasons),
            review_reasons=reasons,
        )

    def _passport_mrz(self, readable: Image) -> PassportMrz | None:
        """The zone from the page as photographed, or from the page turned upside down."""
        for page in (readable, np.ascontiguousarray(np.rot90(readable, 2))):
            found = find_td3(self._ocr.read_lines(page))
            if found is None:
                found = find_td3(self._long_rows_in_pieces(page))
            if found is not None:
                return parse_td3(*found, today=self._today())
        return None

    def _long_rows_in_pieces(self, page: Image) -> list[OcrLine]:
        """The page's long rows, each read in several pieces and joined.

        A line recogniser built for card fields reads a short stretch of text; given all
        44 characters of a zone line at once it returns only the start. So each row that
        spans most of the page is cut at gaps between characters and read piece by piece.
        """
        width = page.shape[1]
        rows: list[OcrLine] = []
        for x, y, w, h in _rows_of(detect_text_lines(page)):
            if w < width * _LONG_ROW_FRACTION:
                continue
            pieces = max(2, round(w / (h * _PIECE_ASPECT)))
            cuts = _gaps_near_even_splits(page[y : y + h, x : x + w], pieces)
            text, confidence = "", 1.0
            for start, end in pairwise([0, *cuts, w]):
                # A margin of plain page on each side, as a whole line would have.
                piece = np.full((h + 2 * _PIECE_PAD, end - start + 2 * _PIECE_PAD), 255, np.uint8)
                piece[_PIECE_PAD : _PIECE_PAD + h, _PIECE_PAD : _PIECE_PAD + end - start] = page[
                    y : y + h, x + start : x + end
                ]
                read = self._ocr.read_lines(piece)
                text += "".join(line.text for line in sorted(read, key=lambda item: item.bbox[0]))
                confidence = min([confidence, *(line.confidence for line in read)])
            rows.append(OcrLine(text=text, bbox=(x, y, w, h), confidence=confidence))
        return rows

    def _card_from_its_text(self, image: Image) -> Image | None:
        """The part of the photo that holds the printed text, scaled to card width.
        Only for an OCR provider that can find text in a whole photo."""
        locate = getattr(self._ocr, "text_region", None)
        if locate is None:
            return None
        region = locate(image)
        if region is None:
            return None
        x0, y0, x1, y1 = region
        if x1 - x0 < self._settings.min_short_side_px // 2 or y1 - y0 < 2:
            return None
        # Kept as large as the photo allows, up to a point: small or ornate print (the Khmer
        # name) reads better with more pixels than the flattened card's 1000 across.
        width = min(max(x1 - x0, CARD_W), _READING_WIDTH)
        return _fit_width(np.ascontiguousarray(image[y0:y1, x0:x1]), width)

    def _read_oriented(
        self, readable: Image
    ) -> tuple[dict[FieldName, FieldResult], list[str], IdCardMrz | None]:
        """OCR + extract, resolving the 180° ambiguity with content.

        A rectangle is 180°-symmetric, so geometry can't tell upright from upside-down.
        We read the card, and if the result looks garbled (no machine-readable zone and too
        few valid fields), we read the 180°-flipped card and keep whichever orientation
        produced more valid fields.
        """
        fields, reasons, mrz = self._extract_and_validate(readable)
        if mrz is not None or self._valid_field_count(fields) >= _MIN_VALID_FIELDS_FOR_ORIENTATION:
            return fields, reasons, mrz

        flipped = np.ascontiguousarray(np.rot90(readable, 2))
        flipped_fields, flipped_reasons, flipped_mrz = self._extract_and_validate(flipped)
        if flipped_mrz is not None or self._valid_field_count(
            flipped_fields
        ) > self._valid_field_count(fields):
            return flipped_fields, flipped_reasons, flipped_mrz
        return fields, reasons, mrz

    def _extract_and_validate(
        self, readable: Image
    ) -> tuple[dict[FieldName, FieldResult], list[str], IdCardMrz | None]:
        lines = self._ocr.read_lines(readable)
        extracted = self._extractor.extract(lines)
        fields, reasons = self._validator.validate(extracted)
        found = find_td1(lines)
        if found is None:
            return fields, reasons, None
        mrz = parse_td1(*found, today=self._today())
        return (*_with_the_zone(fields, reasons, mrz, self._today()), mrz)

    @staticmethod
    def _valid_field_count(fields: dict[FieldName, FieldResult]) -> int:
        return sum(1 for f in fields.values() if f.valid)


def _default_extractor() -> FieldExtractor:
    from khmer_kyc_ocr.pipeline.extract import RuleBasedFieldExtractor

    return RuleBasedFieldExtractor()
