"""Field parsing is where OCR noise becomes a KYC decision, so it carries the edge cases."""

import pytest

from khmer_kyc_ocr.pipeline.parse import (
    khmer_to_arabic_digits,
    parse_date,
    parse_expiry,
    parse_id_number,
    parse_name_khmer,
    parse_name_latin,
    parse_sex,
)


class TestKhmerDigits:
    def test_converts_all_ten_digits(self) -> None:
        assert khmer_to_arabic_digits("០១២៣៤៥៦៧៨៩") == "0123456789"

    def test_leaves_other_characters_untouched(self) -> None:
        assert khmer_to_arabic_digits("ថ្ងៃ ០១.០២") == "ថ្ងៃ 01.02"


class TestIdNumber:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("010845612", "010845612"),
            ("០១០៨៤៥៦១២", "010845612"),  # Khmer numerals
            ("0108 456 12", "010845612"),  # recognizer split into groups
            ("O1O845612", "010845612"),  # letter O read for zero
            ("01084561l", "010845611"),  # lowercase L read for one
            ("អត្តសញ្ញាណប័ណ្ណ: 010845612", "010845612"),  # label bled into ROI
            ("010845612   \u201c \u201c", "010845612"),  # card-edge noise read as quotes
        ],
    )
    def test_normalises_common_ocr_noise(self, raw: str, expected: str) -> None:
        result = parse_id_number(raw)
        assert result.value == expected
        assert result.valid

    @pytest.mark.parametrize("raw", ["", "01084561", "0108456123", "ABCDEFGHI"])
    def test_rejects_wrong_length_or_non_numeric(self, raw: str) -> None:
        assert not parse_id_number(raw).valid


class TestDate:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("01.02.1990", "1990-02-01"),
            ("០១.០២.១៩៩០", "1990-02-01"),
            ("01/02/1990", "1990-02-01"),
            ("01-02-1990", "1990-02-01"),
            ("ថ្ងៃខែឆ្នាំកំណើត: 01.02.1990", "1990-02-01"),
        ],
    )
    def test_parses_day_month_year_to_iso(self, raw: str, expected: str) -> None:
        result = parse_date(raw)
        assert result.value == expected
        assert result.valid

    @pytest.mark.parametrize("raw", ["31.02.1990", "01.13.1990", "01.02.1850", "garbage", ""])
    def test_rejects_impossible_dates(self, raw: str) -> None:
        assert not parse_date(raw).valid


class TestExpiry:
    def test_takes_the_last_date_on_the_validity_line(self) -> None:
        result = parse_expiry("សុពលភាព: ០៥.០៣.២០២០ ដល់ថ្ងៃ ០៤.០៣.២០៣០")
        assert result.value == "2030-03-04"
        assert result.valid

    def test_no_date_is_invalid(self) -> None:
        assert not parse_expiry("សុពលភាព:").valid


class TestSex:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("ប្រុស", "M"), ("ស្រី", "F"), ("ភេទ: ស្រី", "F"), ("M", "M"), ("f", "F")],
    )
    def test_maps_khmer_and_latin(self, raw: str, expected: str) -> None:
        result = parse_sex(raw)
        assert result.value == expected
        assert result.valid

    def test_unknown_is_invalid(self) -> None:
        assert not parse_sex("x").valid


class TestNames:
    def test_latin_name_is_uppercased_and_cleaned(self) -> None:
        result = parse_name_latin("  sok   dara ")
        assert result.value == "SOK DARA"
        assert result.valid

    def test_latin_name_with_digits_is_invalid(self) -> None:
        assert not parse_name_latin("S0K DARA").valid

    def test_khmer_name_strips_label_and_keeps_script(self) -> None:
        result = parse_name_khmer("គោត្តនាម និងនាម: សុខ  ដារា")
        assert result.value == "សុខ ដារា"
        assert result.valid

    def test_khmer_name_rejects_latin_only(self) -> None:
        assert not parse_name_khmer("SOK DARA").valid
