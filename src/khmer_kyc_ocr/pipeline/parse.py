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
# TODO(spike): confirm against the real sample set before production; a new card series may differ.
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
    # After look-alike mapping, anything non-digit is noise (separators, card-edge artifacts).
    # The strict length check below still rejects reads that lost or gained a digit.
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
