"""The rule-based field extractor is the layout-agnostic heart of the pipeline."""

from khmer_kyc_ocr.pipeline.extract import RuleBasedFieldExtractor
from khmer_kyc_ocr.pipeline.ocr import OcrLine


def _line(text: str, y: int = 10, confidence: float = 0.97) -> OcrLine:
    return OcrLine(text=text, bbox=(10, y, 400, 24), confidence=confidence)


def _extract(lines: list[OcrLine]) -> dict[str, str | None]:
    found = RuleBasedFieldExtractor().extract(lines)
    return {name: f.value for name, f in found.items()}


def test_extracts_every_field_from_a_full_card() -> None:
    lines = [
        _line("ព្រះរាជាណាចក្រកម្ពុជា", y=0),
        _line("អត្តសញ្ញាណប័ណ្ណ", y=30),
        _line("លេខអត្តសញ្ញាណប័ណ្ណ: 010845612", y=60),
        _line("គោត្តនាម និងនាម: សុខ ដារា", y=90),
        _line("ឈ្មោះជាឡាតាំង: SOK DARA", y=120),
        _line("ថ្ងៃខែឆ្នាំកំណើត: 01.02.1990", y=150),
        _line("ភេទ: ប្រុស", y=180),
        _line("សញ្ជាតិ: ខ្មែរ", y=210),
        _line("ទីកន្លែងកំណើត: ភ្នំពេញ", y=240),
        _line("អាសយដ្ឋាន: ភ្នំពេញ កម្ពុជា", y=270),
        _line("សុពលភាព: 05.03.2020 ដល់ 04.03.2030", y=300),
    ]
    assert _extract(lines) == {
        "document_type": "អត្តសញ្ញាណប័ណ្ណ",
        "id_number": "010845612",
        "name_kh": "សុខ ដារា",
        "name_en": "SOK DARA",
        "date_of_birth": "01.02.1990",
        "gender": "ប្រុស",
        "nationality": "ខ្មែរ",
        "place_of_birth": "ភ្នំពេញ",
        "address": "ភ្នំពេញ កម្ពុជា",
        "issue_date": "05.03.2020",
        "expiry_date": "04.03.2030",
    }


def test_id_number_found_by_format_without_a_label() -> None:
    lines = [_line("some header"), _line("010845612")]
    assert _extract(lines)["id_number"] == "010845612"


def test_undetected_fields_are_absent() -> None:
    lines = [_line("010845612"), _line("ឈ្មោះជាឡាតាំង: SOK DARA")]
    result = _extract(lines)
    assert "id_number" in result and "name_en" in result
    assert "date_of_birth" not in result
    assert "address" not in result


def test_value_on_next_line_is_captured() -> None:
    lines = [_line("ភេទ:", y=0), _line("ស្រី", y=30)]
    assert _extract(lines)["gender"] == "ស្រី"


def test_khmer_numerals_are_normalised_in_the_id() -> None:
    lines = [_line("លេខ: ០១០៨៤៥៦១២")]
    assert _extract(lines)["id_number"] == "010845612"


def test_validity_line_yields_issue_and_expiry() -> None:
    lines = [_line("សុពលភាព: 05.03.2020 ដល់ថ្ងៃ 04.03.2030")]
    result = _extract(lines)
    assert result["issue_date"] == "05.03.2020"
    assert result["expiry_date"] == "04.03.2030"


def test_layout_agnostic_does_not_care_about_order() -> None:
    lines = [
        _line("អាសយដ្ឋាន: ភ្នំពេញ", y=500),
        _line("ថ្ងៃខែឆ្នាំកំណើត: 01.02.1990", y=400),
        _line("010845612", y=10),
        _line("ឈ្មោះជាឡាតាំង: SOK DARA", y=300),
    ]
    result = _extract(lines)
    assert result["id_number"] == "010845612"
    assert result["date_of_birth"] == "01.02.1990"
    assert result["name_en"] == "SOK DARA"
    assert result["address"] == "ភ្នំពេញ"


def _piece(text: str, x: int, y: int, w: int, confidence: float = 0.97) -> OcrLine:
    return OcrLine(text=text, bbox=(x, y, w, 42), confidence=confidence)


def test_label_and_value_read_as_two_pieces_of_one_line_stay_together() -> None:
    # The projection line finder cuts a printed line at a wide space between words, as it
    # does in DejaVu Sans: the label's value is beside it, not on the line below.
    lines = [
        _piece("Surname:", 58, 272, 198),
        _piece("SOK DARA", 264, 272, 215),
        _piece("Date of Birth:", 59, 381, 272),
        _piece("01.02.1990", 339, 381, 232, confidence=0.43),
        _piece("Validity:", 56, 490, 170),
        _piece("05.03.2020 - 04.03.2030", 233, 490, 498),
    ]
    found = RuleBasedFieldExtractor().extract(lines)
    assert found["name_en"].value == "SOK DARA"
    assert found["date_of_birth"].value == "01.02.1990"
    assert found["date_of_birth"].confidence == 0.43  # as sure as its least sure piece
    assert found["issue_date"].value == "05.03.2020"
    assert found["expiry_date"].value == "04.03.2030"


def test_text_far_along_the_same_row_is_not_joined_on() -> None:
    lines = [_piece("ភេទ: ប្រុស", 10, 100, 160), _piece("010845612", 700, 100, 230)]
    result = _extract(lines)
    assert result["gender"] == "ប្រុស"
    assert result["id_number"] == "010845612"
