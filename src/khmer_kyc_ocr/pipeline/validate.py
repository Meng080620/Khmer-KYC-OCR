"""Validation: turn extracted candidates into validated field results.

Every parser returns ``valid=False`` rather than raising — an OCR miss is an expected
outcome that routes to human review. Review reasons name the field and the problem,
never the field value (values are personal data and must not reach logs/tickets).
"""

from collections.abc import Callable
from datetime import date

from khmer_kyc_ocr.pipeline import parse
from khmer_kyc_ocr.pipeline.extract import ExtractedField
from khmer_kyc_ocr.schemas import FieldName, FieldResult

PARSERS: dict[FieldName, Callable[[str], parse.Parsed]] = {
    "document_type": parse.parse_document_type,
    "id_number": parse.parse_id_number,
    "name_kh": parse.parse_name_khmer,
    "name_en": parse.parse_name_latin,
    "date_of_birth": parse.parse_date,
    "gender": parse.parse_sex,
    "nationality": parse.parse_nationality,
    "place_of_birth": parse.parse_free_text,
    "address": parse.parse_free_text,
    "issue_date": parse.parse_date,
    "expiry_date": parse.parse_expiry,
}


class Validator:
    def __init__(self, *, min_confidence: float, today: Callable[[], date] = date.today) -> None:
        self.min_confidence = min_confidence
        self.today = today

    def validate(
        self, extracted: dict[FieldName, ExtractedField]
    ) -> tuple[dict[FieldName, FieldResult], list[str]]:
        fields: dict[FieldName, FieldResult] = {}
        reasons: list[str] = []

        for name, candidate in extracted.items():
            parser = PARSERS[name]
            parsed = parser(candidate.value if candidate.value is not None else candidate.raw)
            confidence = min(max(candidate.confidence, 0.0), 1.0)
            fields[name] = FieldResult(
                value=parsed.value,
                raw=candidate.raw,
                confidence=confidence,
                valid=parsed.valid,
            )
            if not parsed.valid:
                reasons.append(f"{name}: failed validation")
            elif confidence < self.min_confidence:
                reasons.append(f"{name}: low confidence {confidence:.2f}")

        self._cross_field_checks(fields, reasons)
        return fields, reasons

    def _cross_field_checks(self, fields: dict[FieldName, FieldResult], reasons: list[str]) -> None:
        today = self.today()

        dob = fields.get("date_of_birth")
        if dob and dob.valid and dob.value is not None and date.fromisoformat(dob.value) > today:
            reasons.append("date_of_birth: in the future")

        expiry = fields.get("expiry_date")
        if (
            expiry
            and expiry.valid
            and expiry.value is not None
            and date.fromisoformat(expiry.value) < today
        ):
            reasons.append("expiry_date: card expired")

        issue = fields.get("issue_date")
        if (
            issue
            and expiry
            and issue.valid
            and expiry.valid
            and issue.value
            and expiry.value
            and date.fromisoformat(issue.value) > date.fromisoformat(expiry.value)
        ):
            reasons.append("issue_date: after expiry_date")
