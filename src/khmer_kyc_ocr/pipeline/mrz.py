"""Read a machine-readable zone (ICAO 9303): a passport's two lines of 44 characters
(TD3) or an ID card's three lines of 30 (TD1).

Most of what a reading needs is in those lines, each number and date protected by a check
digit, so a value is only reported as valid when its check digit agrees. Recognisers
confuse look-alike characters (O/0, I/1, S/5...) in this typeface; where a position can
only hold a digit or only a letter the confusion is undone, and the check digit then says
whether that was right.
"""

import re
from dataclasses import dataclass
from datetime import date
from itertools import pairwise

from khmer_kyc_ocr.pipeline.ocr import OcrLine

TD3_LENGTH = 44
# Recognisers return a few characters short or long; the two lines are found by shape and
# then cut or padded to length.
_MIN_LENGTH, _MAX_LENGTH = 36, 50

_WEIGHTS = (7, 3, 1)
_TO_DIGIT = str.maketrans("ODQIL|ZSBG", "0001112586")
_TO_LETTER = str.maketrans("012568", "OIZSGB")
_FILLER_LOOKALIKES = str.maketrans(
    {"«": "<", "‹": "<", "(": "<", "[": "<", "{": "<", "≤": "<", ">": "<", "»": "<", "›": "<"}
)
# A recogniser that knows Khmer sometimes gives a zone digit as the Khmer numeral.
_KHMER_DIGITS = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")


def check_digit(text: str) -> int:
    """ICAO 9303 check digit: digits as themselves, A-Z as 10-35, '<' as 0; weights 7,3,1."""
    total = 0
    for i, char in enumerate(text):
        if char.isdigit():
            value = int(char)
        elif "A" <= char <= "Z":
            value = ord(char) - 55
        else:
            value = 0
        total += value * _WEIGHTS[i % 3]
    return total % 10


@dataclass(frozen=True, slots=True)
class MrzField:
    """One value from the zone. ``valid`` is None where nothing in the zone can confirm it."""

    value: str | None
    raw: str
    valid: bool | None


@dataclass(frozen=True, slots=True)
class PassportMrz:
    document_number: MrzField
    surname: MrzField
    given_names: MrzField
    nationality: MrzField
    date_of_birth: MrzField
    gender: MrzField
    expiry_date: MrzField
    confidence: float
    # False when the check digit over the whole second line disagrees.
    composite_valid: bool


def _normalise(text: str) -> str:
    return re.sub(r"\s+", "", text.upper().translate(_FILLER_LOOKALIKES).translate(_KHMER_DIGITS))


def _looks_like_mrz(text: str) -> bool:
    if not _MIN_LENGTH <= len(text) <= _MAX_LENGTH:
        return False
    allowed = sum(1 for c in text if c.isalnum() or c == "<")
    return allowed / len(text) > 0.9 and ("<" in text or sum(c.isdigit() for c in text) >= 10)


def _fit(text: str) -> str:
    return text[:TD3_LENGTH].ljust(TD3_LENGTH, "<")


def _rows(lines: list[OcrLine]) -> list[tuple[str, float]]:
    """Join the pieces of each printed row: line detection splits a row at wide gaps."""
    rows: list[list[OcrLine]] = []
    for line in sorted(lines, key=lambda item: (item.bbox[1], item.bbox[0])):
        _, y, _, h = line.bbox
        if rows:
            _, prev_y, _, prev_h = rows[-1][0].bbox
            if abs((y + h / 2) - (prev_y + prev_h / 2)) < max(h, prev_h) / 2:
                rows[-1].append(line)
                continue
        rows.append([line])
    joined: list[tuple[str, float]] = []
    for row in rows:
        row.sort(key=lambda item: item.bbox[0])
        text = _normalise("".join(item.text for item in row))
        joined.append((text, min(item.confidence for item in row)))
    return joined


def find_td3(lines: list[OcrLine]) -> tuple[str, str, float] | None:
    """The two zone lines, top then bottom, with the lower of their confidences."""
    rows = _rows(lines)
    for (upper, upper_conf), (lower, lower_conf) in pairwise(rows):
        if not (_looks_like_mrz(upper) and _looks_like_mrz(lower)):
            continue
        # The top line starts with the document type, "P" for a passport.
        if upper[0] not in "PF" and not upper.startswith("<"):
            continue
        return _fit(upper), _fit(lower), min(upper_conf, lower_conf)
    return None


def _date(text: str, *, born: bool, today: date) -> str | None:
    """YYMMDD to YYYY-MM-DD. A birth date is never in the future; an expiry date is taken
    as this century."""
    if not text.isdigit():
        return None
    year, month, day = int(text[:2]), int(text[2:4]), int(text[4:6])
    century = 2000
    if born and 2000 + year > today.year:
        century = 1900
    try:
        return date(century + year, month, day).isoformat()
    except ValueError:
        return None


def _name(text: str) -> str | None:
    words = [word for word in text.translate(_TO_LETTER).split("<") if word]
    return " ".join(words) if words and all(word.isalpha() for word in words) else None


def parse_td3(upper: str, lower: str, confidence: float, *, today: date) -> PassportMrz:
    names = upper[5:]
    surname_raw, _, given_raw = names.partition("<<")
    surname, given = _name(surname_raw), _name(given_raw)

    number_raw = lower[0:9]
    number = number_raw.rstrip("<")
    number_check = lower[9].translate(_TO_DIGIT)
    nationality_raw = lower[10:13]
    nationality = nationality_raw.translate(_TO_LETTER).rstrip("<")
    birth_raw = lower[13:19].translate(_TO_DIGIT)
    birth_check = lower[19].translate(_TO_DIGIT)
    gender_raw = lower[20]
    expiry_raw = lower[21:27].translate(_TO_DIGIT)
    expiry_check = lower[27].translate(_TO_DIGIT)
    personal = lower[28:42]
    personal_check = lower[42].translate(_TO_DIGIT)
    composite_check = lower[43].translate(_TO_DIGIT)

    def agrees(text: str, digit: str) -> bool:
        return digit.isdigit() and check_digit(text) == int(digit)

    birth = _date(birth_raw, born=True, today=today)
    expiry = _date(expiry_raw, born=False, today=today)
    composite = (
        number_raw
        + number_check
        + birth_raw
        + birth_check
        + expiry_raw
        + expiry_check
        + personal
        + personal_check
    )
    return PassportMrz(
        document_number=MrzField(
            number or None, number_raw, bool(number) and agrees(number_raw, number_check)
        ),
        # Names carry no check digit of their own.
        surname=MrzField(surname, surname_raw, None if surname else False),
        given_names=MrzField(given, given_raw, None if given else False),
        nationality=MrzField(
            nationality if len(nationality) == 3 and nationality.isalpha() else None,
            nationality_raw,
            None if len(nationality) == 3 and nationality.isalpha() else False,
        ),
        date_of_birth=MrzField(
            birth, lower[13:19], birth is not None and agrees(birth_raw, birth_check)
        ),
        gender=MrzField(
            gender_raw if gender_raw in "MF" else None,
            gender_raw,
            None if gender_raw in "MF<" else False,
        ),
        expiry_date=MrzField(
            expiry, lower[21:27], expiry is not None and agrees(expiry_raw, expiry_check)
        ),
        confidence=confidence,
        composite_valid=agrees(composite, composite_check),
    )


TD1_LENGTH = 30
# An ID card's lines are shorter than a passport's, and a recogniser miscounts the run of
# '<' that pads them, so their length says little: they are found by what they start with.
_TD1_MIN_LENGTH = 18


@dataclass(frozen=True, slots=True)
class IdCardMrz:
    lines: tuple[str, str, str]
    document_number: MrzField
    surname: MrzField
    given_names: MrzField
    nationality: MrzField
    date_of_birth: MrzField
    gender: MrzField
    expiry_date: MrzField
    confidence: float
    composite_valid: bool


def _td1_row(text: str) -> bool:
    if len(text) < _TD1_MIN_LENGTH:
        return False
    allowed = sum(1 for c in text if c.isalnum() or c == "<")
    return allowed / len(text) > 0.9 and "<" in text


def _pad(text: str) -> str:
    return text[:TD1_LENGTH].ljust(TD1_LENGTH, "<")


# A digit, or a letter a recogniser gives in its place.
_D = "[0-9ODQILZSBG]"
# Document type and issuing state, the nine-character number, its check digit.
_TD1_FIRST = re.compile(r"([IAC1][A-Z0-9<])([A-Z]{3})([A-Z0-9<]{9})([A-Z0-9])")
# Birth date and check digit, sex, expiry date and check digit, nationality.
_TD1_SECOND = re.compile(rf"({_D}{{6}})({_D})([MF<])({_D}{{6}})({_D})([A-Z]{{3}})")
# Surname, two fillers, given names.
_TD1_THIRD = re.compile(r"([A-Z]+(?:<[A-Z]+)*)<<([A-Z]+(?:<[A-Z]+)*)")


def find_td1(lines: list[OcrLine]) -> tuple[str, str, str, float] | None:
    """The three zone lines of an ID card, top to bottom, with the lowest confidence.

    Found by what each line holds, not by where the rows fall: a recogniser miscounts the
    padding, adds a stray character at an end, and a line detector sometimes hands back
    two of the lines as one. Each line is rebuilt to its 30 characters from the parts
    that matter; the padding in between carries nothing.
    """
    rows = [(text, confidence) for text, confidence in _rows(lines) if _td1_row(text)]
    if not rows:
        return None
    zone = "|".join(text for text, _ in rows)
    confidence = min(confidence for _, confidence in rows)

    first = _TD1_FIRST.search(zone)
    if first is None:
        return None
    second = _TD1_SECOND.search(zone, first.end())
    if second is None:
        return None
    third = _TD1_THIRD.search(zone, second.end())
    if third is None:
        return None

    # The overall check digit is the last character of the second line: whatever digit
    # ends the stretch between the nationality and the next line.
    rest = zone[second.end() : third.start()].split("|")[0].rstrip("<")
    overall = rest[-1:].translate(_TO_DIGIT)
    line_two = second.group(0) + "<" * 11 + (overall if overall.isdigit() else "<")
    return (
        _pad(first.group(0)),
        line_two,
        _pad(f"{third.group(1)}<<{third.group(2)}"),
        confidence,
    )


def parse_td1(first: str, second: str, third: str, confidence: float, *, today: date) -> IdCardMrz:
    def agrees(text: str, digit: str) -> bool:
        return digit.isdigit() and check_digit(text) == int(digit)

    number_raw = first[5:14]
    number_check = first[14].translate(_TO_DIGIT)
    # A Khmer ID number is all digits, so a letter in it is usually a misread digit; other
    # countries' numbers have letters. Whichever reading its check digit agrees with wins.
    as_read, as_digits = number_raw, number_raw.translate(_TO_DIGIT)
    number_ok = agrees(as_read, number_check)
    number = as_read
    if not number_ok and agrees(as_digits, number_check):
        number, number_ok = as_digits, True
    number = number.rstrip("<")
    number_ok = number_ok and bool(number)

    birth_raw = second[0:6].translate(_TO_DIGIT)
    birth_check = second[6].translate(_TO_DIGIT)
    gender_raw = second[7]
    expiry_raw = second[8:14].translate(_TO_DIGIT)
    expiry_check = second[14].translate(_TO_DIGIT)
    nationality_raw = second[15:18]
    nationality = nationality_raw.translate(_TO_LETTER).rstrip("<")
    birth = _date(birth_raw, born=True, today=today)
    expiry = _date(expiry_raw, born=False, today=today)

    surname_raw, _, given_raw = third.partition("<<")
    surname, given = _name(surname_raw), _name(given_raw)

    composite = (
        number.ljust(9, "<")
        + number_check
        + first[15:30]
        + birth_raw
        + birth_check
        + expiry_raw
        + expiry_check
        + second[18:29]
    )
    return IdCardMrz(
        lines=(first[:5] + number.ljust(9, "<") + first[14:], second, third),
        document_number=MrzField(number or None, number_raw, number_ok),
        surname=MrzField(surname, surname_raw, None if surname else False),
        given_names=MrzField(given, given_raw, None if given else False),
        nationality=MrzField(
            nationality if len(nationality) == 3 and nationality.isalpha() else None,
            nationality_raw,
            None if len(nationality) == 3 and nationality.isalpha() else False,
        ),
        date_of_birth=MrzField(
            birth, second[0:6], birth is not None and agrees(birth_raw, birth_check)
        ),
        gender=MrzField(
            gender_raw if gender_raw in "MF" else None,
            gender_raw,
            None if gender_raw in "MF<" else False,
        ),
        expiry_date=MrzField(
            expiry, second[8:14], expiry is not None and agrees(expiry_raw, expiry_check)
        ),
        confidence=confidence,
        composite_valid=agrees(composite, second[29].translate(_TO_DIGIT)),
    )
