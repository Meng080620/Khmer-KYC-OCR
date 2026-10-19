"""The passport zone: ICAO 9303's own specimen, plus the mistakes a recogniser makes."""

from datetime import date

from khmer_kyc_ocr.pipeline.mrz import check_digit, find_td1, find_td3, parse_td1, parse_td3
from khmer_kyc_ocr.pipeline.ocr import OcrLine

UPPER = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
LOWER = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
TODAY = date(2026, 10, 8)


def line(text: str, y: int, x: int = 40, confidence: float = 0.9) -> OcrLine:
    return OcrLine(text=text, bbox=(x, y, 1300, 40), confidence=confidence)


def test_check_digit_matches_the_specimen() -> None:
    assert check_digit("L898902C3") == 6
    assert check_digit("740812") == 2
    assert check_digit("120415") == 9
    assert check_digit("ZE184226B<<<<<") == 1


def test_reads_every_field_of_the_specimen() -> None:
    mrz = parse_td3(UPPER, LOWER, 0.9, today=TODAY)
    assert (mrz.document_number.value, mrz.document_number.valid) == ("L898902C3", True)
    assert (mrz.surname.value, mrz.given_names.value) == ("ERIKSSON", "ANNA MARIA")
    assert mrz.nationality.value == "UTO"
    assert (mrz.date_of_birth.value, mrz.date_of_birth.valid) == ("1974-08-12", True)
    assert mrz.gender.value == "F"
    assert (mrz.expiry_date.value, mrz.expiry_date.valid) == ("2012-04-15", True)
    assert mrz.composite_valid


def test_a_birth_year_is_never_in_the_future() -> None:
    lower = "L898902C36UTO3008122F3204159ZE184226B<<<<<10"
    mrz = parse_td3(UPPER, lower, 0.9, today=TODAY)
    assert mrz.date_of_birth.value == "1930-08-12"
    assert mrz.expiry_date.value == "2032-04-15"


def test_lookalike_characters_are_undone_where_only_a_digit_fits() -> None:
    # 7408122 read as 74O8I22, 1204159 as I2O4I59.
    lower = "L898902C36UTO74O8I22F I2O4I59ZE184226B<<<<<10".replace(" ", "")
    mrz = parse_td3(UPPER, lower, 0.9, today=TODAY)
    assert (mrz.date_of_birth.value, mrz.date_of_birth.valid) == ("1974-08-12", True)
    assert (mrz.expiry_date.value, mrz.expiry_date.valid) == ("2012-04-15", True)


def test_a_misread_digit_fails_its_check_digit() -> None:
    lower = "L898902C36UTO7408132F1204159ZE184226B<<<<<10"
    mrz = parse_td3(UPPER, lower, 0.9, today=TODAY)
    assert mrz.date_of_birth.valid is False
    assert mrz.document_number.valid is True
    assert not mrz.composite_valid


def test_finds_the_zone_among_the_other_lines_and_joins_split_rows() -> None:
    lines = [
        line("PASSPORT", 100),
        line("ERIKSSON", 200),
        line("P<UTOERIKSSON<<ANNA<MARIA", 700),
        line("<<<<<<<<<<<<<<<<<<<", 702, x=900),  # the same printed row, detected in two pieces
        line("L898902C36UTO7408122F1204159ZE184226B<<<<<10", 760, confidence=0.7),
    ]
    found = find_td3(lines)
    assert found == (UPPER, LOWER, 0.7)


def test_spaces_and_guillemets_from_the_recogniser_are_tolerated() -> None:
    lines = [
        line("P<UTOERIKSSON«ANNA<MARIA<<<<<<<<<<<<<<<<<<", 700),
        line("L898902C36 UTO7408122F1204159 ZE184226B<<<<<10", 760),
    ]
    found = find_td3(lines)
    assert found is not None
    assert found[1] == LOWER


def test_no_zone_on_a_page_without_one() -> None:
    assert find_td3([line("KINGDOM OF CAMBODIA", 100), line("PASSPORT", 200)]) is None


# The ICAO TD1 specimen (an ID card): three lines of 30.
TD1 = (
    "I<UTOD231458907<<<<<<<<<<<<<<<",
    "7408122F1204159UTO<<<<<<<<<<<6",
    "ERIKSSON<<ANNA<MARIA<<<<<<<<<<",
)


def test_reads_every_field_of_the_id_card_specimen() -> None:
    mrz = parse_td1(*TD1, 0.9, today=TODAY)
    assert (mrz.document_number.value, mrz.document_number.valid) == ("D23145890", True)
    assert (mrz.date_of_birth.value, mrz.date_of_birth.valid) == ("1974-08-12", True)
    assert mrz.gender.value == "F"
    assert (mrz.expiry_date.value, mrz.expiry_date.valid) == ("2012-04-15", True)
    assert mrz.nationality.value == "UTO"
    assert (mrz.surname.value, mrz.given_names.value) == ("ERIKSSON", "ANNA MARIA")
    assert mrz.composite_valid
    assert mrz.lines == TD1


def test_finds_an_id_card_zone_whose_padding_was_miscounted() -> None:
    # What a recogniser gives: six '<' too many on each line, the last check digit after them.
    lines = [
        line("KINGDOM OF UTOPIA", 100),
        line("I<UTOD231458907" + "<" * 21, 400),
        line("7408122F1204159UTO" + "<" * 17 + "6", 450),
        line("ERIKSSON<<ANNA<MARIA" + "<" * 16, 500),
    ]
    found = find_td1(lines)
    assert found is not None
    assert found[:3] == TD1


def test_an_id_card_misread_fails_its_check_digit() -> None:
    right = parse_td1("IDKHM1234567897<<<<<<<<<<<<<<<", TD1[1], TD1[2], 0.9, today=TODAY)
    assert (right.document_number.value, right.document_number.valid) == ("123456789", True)
    wrong = parse_td1("IDKHM1234567807<<<<<<<<<<<<<<<", TD1[1], TD1[2], 0.9, today=TODAY)
    assert wrong.document_number.valid is False
    assert wrong.date_of_birth.valid is True
    assert not wrong.composite_valid


def test_a_digit_read_as_its_lookalike_letter_is_recovered_by_the_check_digit() -> None:
    # 123456789 read as I23456789, and 0 as O.
    mrz = parse_td1("IDKHMI234567897<<<<<<<<<<<<<<<", TD1[1], TD1[2], 0.9, today=TODAY)
    assert (mrz.document_number.value, mrz.document_number.valid) == ("123456789", True)
    assert mrz.lines[0] == "IDKHM1234567897<<<<<<<<<<<<<<<"


def test_no_id_card_zone_in_a_passport_or_plain_text() -> None:
    assert find_td1([line(UPPER, 700), line(LOWER, 760)]) is None
    assert find_td1([line("KINGDOM OF CAMBODIA", 100), line("NAME", 200), line("X", 300)]) is None


def test_finds_the_id_card_zone_through_the_mistakes_a_real_reading_makes() -> None:
    # Seen on a real card: a stray '<' before the second line, its last check digit given
    # as a Khmer numeral, '>' for '<', and the padding miscounted throughout.
    lines = [
        line("I<UTOD231458907" + "<" * 21, 400),
        line("<7408122F1204159UTO" + "<" * 15 + "៦", 450),
        line("ERIKSSON<<ANNA<MARIA" + "<" * 14 + ">", 500),
    ]
    found = find_td1(lines)
    assert found is not None
    assert found[:3] == TD1


def test_finds_the_id_card_zone_when_two_of_its_lines_come_back_as_one() -> None:
    lines = [
        line("I<UTOD231458907" + "<" * 15 + "7408122F1204159UTO" + "<" * 11 + "6", 400),
        line("ERIKSSON<<ANNA<MARIA" + "<" * 10, 500),
    ]
    found = find_td1(lines)
    assert found is not None
    assert found[:3] == TD1
