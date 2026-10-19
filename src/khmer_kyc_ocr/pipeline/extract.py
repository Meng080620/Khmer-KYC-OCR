"""Field extraction: map OCR lines to structured KYC fields, layout-agnostic.

The rule-based extractor does not assume a fixed template. It works over the recognised
text lines (with their positions) using two strategies:

1. **Label match** — a line contains a printed label in Khmer or English; the value is the
   text after the label (or the following line when the value is on its own line).
2. **Format match** — fields with an unambiguous shape (a 9-digit ID number, dates) are
   found by their format, independent of where they sit on the card.

Only fields we actually observe are returned; the extractor never invents values.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Protocol

from khmer_kyc_ocr.pipeline.ocr import OcrLine
from khmer_kyc_ocr.pipeline.parse import khmer_to_arabic_digits
from khmer_kyc_ocr.schemas import FieldName

_LABELS: dict[FieldName, list[str]] = {
    "name_kh": ["គោត្តនាម និងនាម", "គោត្តនាមនិងនាម", "នាមខ្លួន"],
    "name_en": ["ឈ្មោះជាឡាតាំង", "latin name", "name in latin", "surname"],
    "gender": ["ភេទ", "sex", "gender"],
    "nationality": ["សញ្ជាតិ", "nationality"],
    "place_of_birth": ["ទីកន្លែងកំណើត", "place of birth"],
    "address": ["អាសយដ្ឋាន", "ទីលំនៅ", "address"],
}

_DOCUMENT_TYPE_LABELS = [
    "អត្តសញ្ញាណប័ណ្ណសញ្ជាតិខ្មែរ",
    "អត្តសញ្ញាណប័ណ្ណខ្មែរ",
    "អត្តសញ្ញាណប័ណ្ណ",
    "national identity card",
    "identity card",
]

_BIRTH_LABELS = ("ថ្ងៃខែឆ្នាំកំណើត", "date of birth")
_VALIDITY_LABELS = ("សុពលភាព", "validity", "valid", "expiry", "expires")

_DATE_RE = re.compile(r"\d{1,2}\s*[./\-]\s*\d{1,2}\s*[./\-]\s*\d{4}")
_ID_RE = re.compile(r"^\d{9}$")
_DIGIT_LOOKALIKES = str.maketrans(
    {"O": "0", "o": "0", "D": "0", "I": "1", "l": "1", "|": "1", "S": "5", "B": "8"}
)


@dataclass(frozen=True, slots=True)
class ExtractedField:
    """A raw, pre-validation field candidate."""

    raw: str
    value: str | None
    confidence: float


class FieldExtractor(Protocol):
    def extract(self, lines: list[OcrLine]) -> dict[FieldName, ExtractedField]: ...


# A label read with a character or two wrong is still that label. Short labels are only
# matched exactly: three characters are too few to tell a near miss from another word.
_FUZZY_MIN_LABEL = 6
_FUZZY_RATIO = 0.72


def _label_span(text: str, label: str) -> tuple[int, int] | None:
    """Where ``label`` is in ``text``, allowing a recogniser's small misreadings."""
    lower, wanted = text.lower(), label.lower()
    exact = lower.find(wanted)
    if exact >= 0:
        return exact, exact + len(wanted)
    if len(wanted) < _FUZZY_MIN_LABEL:
        return None
    best: tuple[float, int, int] | None = None
    for size in range(len(wanted) - 2, len(wanted) + 3):
        for start in range(max(0, len(lower) - size) + 1):
            ratio = SequenceMatcher(None, lower[start : start + size], wanted).ratio()
            if ratio >= _FUZZY_RATIO and (best is None or ratio > best[0]):
                best = (ratio, start, start + size)
    return None if best is None else (best[1], best[2])


def _has_label(text: str, labels: tuple[str, ...] | list[str]) -> bool:
    return any(_label_span(text, label) is not None for label in labels)


def _strip_label(text: str, label: str) -> str:
    span = _label_span(text, label)
    if span is None:
        return text
    rest = text[span[1] :]
    # A label matched loosely can stop a character or two short of its colon, and a
    # recogniser sometimes reads the colon itself as a stray character.
    colon = re.match(r"^.{0,3}?[:៖]", rest)
    if colon is not None:
        rest = rest[colon.end() :]
    rest = rest.lstrip(" :៖").strip()
    stray = re.match(r"^\S\s+(?=\S)", rest)
    if stray is not None and text[span[0] : span[1]].lower() != label.lower():
        rest = rest[stray.end() :]
    return rest


def _clean_digits(text: str) -> str:
    return khmer_to_arabic_digits(text).translate(_DIGIT_LOOKALIKES)


def _normalise_id(text: str) -> str | None:
    cleaned = _clean_digits(text)
    runs: list[str] = re.findall(r"\d+", cleaned)
    for run in runs:
        if len(run) == 9:
            return run
    digits = re.sub(r"\D", "", cleaned)
    return digits if _ID_RE.fullmatch(digits) else None


class RuleBasedFieldExtractor:
    def extract(self, lines: list[OcrLine]) -> dict[FieldName, ExtractedField]:
        ordered = sorted(lines, key=lambda ln: (ln.bbox[1], ln.bbox[0]))
        found: dict[FieldName, ExtractedField] = {}

        self._extract_id_number(ordered, found)
        self._extract_dates(ordered, found)
        self._extract_document_type(ordered, found)
        for field, labels in _LABELS.items():
            if field not in found:
                self._extract_by_label(field, labels, ordered, found)
        return found

    def _extract_id_number(
        self, lines: list[OcrLine], found: dict[FieldName, ExtractedField]
    ) -> None:
        for line in lines:
            value = _normalise_id(line.text)
            if value is not None:
                found["id_number"] = ExtractedField(line.text, value, line.confidence)
                return

    def _extract_dates(self, lines: list[OcrLine], found: dict[FieldName, ExtractedField]) -> None:
        for line in lines:
            if _has_label(line.text, _VALIDITY_LABELS) and not _has_label(line.text, _BIRTH_LABELS):
                dates = _DATE_RE.findall(khmer_to_arabic_digits(line.text))
                # The line reads "from <date> until <date>". With only one of the two read
                # there is no telling which it is, and calling it both would make a valid
                # card look expired on the day it was issued.
                if len(dates) >= 2:
                    found["issue_date"] = ExtractedField(line.text, dates[0], line.confidence)
                    found["expiry_date"] = ExtractedField(line.text, dates[-1], line.confidence)
                    break
        for line in lines:
            if _has_label(line.text, _BIRTH_LABELS):
                dates = _DATE_RE.findall(khmer_to_arabic_digits(line.text))
                if dates:
                    found["date_of_birth"] = ExtractedField(line.text, dates[0], line.confidence)
                    return

    def _extract_document_type(
        self, lines: list[OcrLine], found: dict[FieldName, ExtractedField]
    ) -> None:
        for line in lines:
            lower = line.text.lower()
            has_title = any(lbl in lower for lbl in _DOCUMENT_TYPE_LABELS)
            if has_title and not re.search(r"\d", line.text):
                found["document_type"] = ExtractedField(line.text, line.text, line.confidence)
                return

    def _extract_by_label(
        self,
        field: FieldName,
        labels: list[str],
        lines: list[OcrLine],
        found: dict[FieldName, ExtractedField],
    ) -> None:
        for line in lines:
            for label in labels:
                if _label_span(line.text, label) is None:
                    continue
                value = _strip_label(line.text, label)
                if not value:
                    value = self._next_line_value(line, lines)
                elif field == "address":
                    value = f"{value} {self._continuation(line, lines)}".strip()
                if value:
                    found[field] = ExtractedField(value, value, line.confidence)
                    return

    def _continuation(self, line: OcrLine, lines: list[OcrLine]) -> str:
        """The address runs on to a second line: the line right below, if it is plain text
        with no label of its own."""
        y, h = line.bbox[1], line.bbox[3]
        below = [ln for ln in lines if y + h // 2 < ln.bbox[1] < y + h * 2]
        if not below:
            return ""
        text = min(below, key=lambda ln: ln.bbox[1]).text.strip()
        labelled = ":" in text or "៖" in text or "<" in text or re.search(r"\d", text)
        return "" if labelled else text

    def _next_line_value(self, line: OcrLine, lines: list[OcrLine]) -> str:
        y, h = line.bbox[1], line.bbox[3]
        candidates = [ln for ln in lines if ln.bbox[1] > y + h // 2]
        if not candidates:
            return ""
        return min(candidates, key=lambda ln: ln.bbox[1]).text.strip()


class AIExtractionProvider:
    """Structured extraction backed by a language/vision model instead of rules.

    This is an extension point for a future LLM-based extractor. It is deliberately not
    wired to any model by default: sending identity text to a third-party model is a
    privacy decision. A concrete implementation should accept ``lines`` plus an optional
    document image, and return the same :class:`ExtractedField` mapping — after passing
    every field through the :mod:`~khmer_kyc_ocr.pipeline.validate` validators, because a
    model can hallucinate plausible-looking values.
    """

    engine = "ai-extraction"

    def extract(self, lines: list[OcrLine]) -> dict[FieldName, ExtractedField]:
        raise NotImplementedError(
            "AIExtractionProvider requires an explicit model endpoint and is disabled by "
            "default (identity data must not be sent to a third-party model without review)."
        )
