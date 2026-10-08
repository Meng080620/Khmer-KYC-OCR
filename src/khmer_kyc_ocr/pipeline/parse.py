"""Turn raw recognizer strings into validated KYC field values.

Every parser returns a `Parsed` with `valid=False` instead of raising: an OCR miss is an
expected outcome that routes to human review, not an exception.
"""

import re
from dataclasses import dataclass
from datetime import date

_KHMER_DIGITS = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")
# Letters the recognizer commonly emits for digits. Only applied to numeric-only fields.
_DIGIT_LOOKALIKES = str.maketrans(
    {"O": "0", "o": "0", "D": "0", "I": "1", "l": "1", "|": "1", "S": "5", "B": "8"}
)

# Cambodian national ID numbers are 9 digits on the cards we have seen.
ID_NUMBER_RE = re.compile(r"^\d{9}$")
_KHMER_BLOCK = "ក-៿᧠-᧿"
_KHMER_NAME_RE = re.compile(rf"^[{_KHMER_BLOCK}]+( [{_KHMER_BLOCK}]+)*$")
_LATIN_NAME_RE = re.compile(r"^[A-Z]+([ '\-][A-Z]+)*$")
_DATE_RE = re.compile(r"(\d{1,2})\s*[./\-]\s*(\d{1,2})\s*[./\-]\s*(\d{4})")
_MIN_YEAR, _MAX_YEAR = 1900, 2100

_SEX = {"ប្រុស": "M", "ស្រី": "F", "M": "M", "F": "F"}


@dataclass(frozen=True, slots=True)
class Parsed:
    value: str | None
    valid: bool


_INVALID = Parsed(value=None, valid=False)


def khmer_to_arabic_digits(text: str) -> str:
    return text.translate(_KHMER_DIGITS)


def _strip_label(text: str) -> str:
    """Crops often catch the printed label ("ភេទ:", "ID:"); keep what follows the last colon."""
    return text.rsplit(":", 1)[-1].strip() if ":" in text else text.strip()


def _collapse_spaces(text: str) -> str:
    # Khmer text often carries zero-width spaces (U+200B) as word-break hints.
    return " ".join(text.replace("\u200b", "").split())


def parse_id_number(raw: str) -> Parsed:
    text = khmer_to_arabic_digits(_strip_label(raw)).translate(_DIGIT_LOOKALIKES)
    digits = re.sub(r"\D", "", text)
    return Parsed(digits, True) if ID_NUMBER_RE.fullmatch(digits) else _INVALID


def _to_iso(day: int, month: int, year: int) -> Parsed:
    if not _MIN_YEAR <= year <= _MAX_YEAR:
        return _INVALID
    try:
        return Parsed(date(year, month, day).isoformat(), True)
    except ValueError:  # 31.02, month 13, ...
        return _INVALID


def parse_date(raw: str) -> Parsed:
    match = _DATE_RE.search(khmer_to_arabic_digits(raw))
    return _INVALID if match is None else _to_iso(*(int(g) for g in match.groups()))


def parse_expiry(raw: str) -> Parsed:
    """The validity line prints "from <date> until <date>"; expiry is the last date on it."""
    matches = _DATE_RE.findall(khmer_to_arabic_digits(raw))
    return _INVALID if not matches else _to_iso(*(int(g) for g in matches[-1]))


def parse_sex(raw: str) -> Parsed:
    value = _SEX.get(_strip_label(raw).upper())  # upper() is a no-op on Khmer
    return Parsed(value, True) if value else _INVALID


def parse_name_latin(raw: str) -> Parsed:
    name = _collapse_spaces(_strip_label(raw)).upper()
    return Parsed(name, True) if _LATIN_NAME_RE.fullmatch(name) else _INVALID


def parse_name_khmer(raw: str) -> Parsed:
    name = _collapse_spaces(_strip_label(raw))
    return Parsed(name, True) if _KHMER_NAME_RE.fullmatch(name) else _INVALID


def parse_issue(raw: str) -> Parsed:
    """The validity line prints "from <date> until <date>"; issue is the first date on it."""
    matches = _DATE_RE.findall(khmer_to_arabic_digits(raw))
    return _INVALID if not matches else _to_iso(*(int(g) for g in matches[0]))


# ---------------------------------------------------------------------------
# Free-text and enumerated fields (document type, nationality, address, ...)
# ---------------------------------------------------------------------------

_DOCUMENT_TYPES = {
    "national_id": (
        "អត្តសញ្ញាណប័ណ្ណ",
        "អត្តសញ្ញាណប័ណ្ណសញ្ជាតិខ្មែរ",
        "អត្តសញ្ញាណប័ណ្ណខ្មែរ",
        "national identity card",
        "national id",
        "identity card",
    ),
    "passport": ("លិខិតឆ្លងដែន", "passport"),
}
_NATIONALITIES = {
    "Khmer": ("ខ្មែរ", "khmer"),
    "Cambodian": ("cambodian", "cambodia"),
}
# A free-text value must contain letters in some script and stay under a sane length.
_TEXT_RE = re.compile(r"[\u1780-\u17ffA-Za-z]", re.UNICODE)
_MAX_TEXT_LEN = 120


def parse_document_type(raw: str) -> Parsed:
    # The document type is a title/header line, not a "label: value" line, so do NOT
    # strip after a colon here (that would discard the value).
    text = _collapse_spaces(raw).lower()
    for value, labels in _DOCUMENT_TYPES.items():
        if any(label in text for label in labels):
            return Parsed(value, True)
    return _INVALID


def parse_nationality(raw: str) -> Parsed:
    text = _collapse_spaces(_strip_label(raw)).lower()
    for value, labels in _NATIONALITIES.items():
        if any(label in text for label in labels):
            return Parsed(value, True)
    return _INVALID


def parse_free_text(raw: str) -> Parsed:
    """Address / place of birth: keep any line that has letters, cleaned of the label."""
    text = _collapse_spaces(_strip_label(raw))
    if text and _TEXT_RE.search(text) and len(text) <= _MAX_TEXT_LEN:
        return Parsed(text, True)
    return _INVALID
