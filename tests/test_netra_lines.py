"""Joining the detector's boxes into printed lines. Pure geometry: no model is loaded."""

import numpy as np

from khmer_kyc_ocr.pipeline.netra import NetraOCRProvider, _join_pieces_of_a_line


def test_a_label_and_its_value_become_one_line() -> None:
    label, value = (240, 119, 426, 162), (402, 117, 943, 162)
    assert _join_pieces_of_a_line([value, label]) == [(240, 117, 943, 162)]


def test_a_value_that_arrives_in_pieces_still_reaches_back_to_its_label() -> None:
    # The piece nearest the label is the last to join the value: one pass would miss it.
    boxes = [(240, 119, 426, 162), (820, 117, 943, 162), (600, 117, 800, 162), (402, 117, 580, 162)]
    assert _join_pieces_of_a_line(boxes) == [(240, 117, 943, 162)]


def test_lines_above_and_below_each_other_stay_apart() -> None:
    upper, lower = (240, 100, 600, 140), (240, 150, 600, 190)
    assert _join_pieces_of_a_line([lower, upper]) == [upper, lower]


def test_two_columns_far_apart_on_one_row_stay_apart() -> None:
    left, right = (50, 100, 200, 140), (700, 100, 900, 140)
    assert _join_pieces_of_a_line([left, right]) == [left, right]


class _Detected:
    label = "text"

    def __init__(self, bbox: tuple[int, int, int, int]) -> None:
        self.bbox = bbox


class _Detector:
    def __init__(self, boxes: list[tuple[int, int, int, int]]) -> None:
        self.boxes = boxes

    def detect(self, _: object) -> list[_Detected]:
        return [_Detected(box) for box in self.boxes]


def _provider(boxes: list[tuple[int, int, int, int]]) -> NetraOCRProvider:
    import threading

    provider = object.__new__(NetraOCRProvider)  # no models: only the geometry is under test
    provider._detector = _Detector(boxes)
    provider._lock = threading.Lock()
    return provider


def test_the_text_region_keeps_the_lowest_lines_of_a_card_that_fills_the_photo() -> None:
    # Ten lines down a photo, the last three (the machine-readable zone) far below the middle.
    lines = [(300, 60 + i * 110, 2200, 110 + i * 110) for i in range(7)]
    zone = [(200, 900 + i * 130, 2300, 960 + i * 130) for i in range(3)]
    photo = np.zeros((1381, 2455, 3), dtype=np.uint8)
    region = _provider([*lines, *zone]).text_region(photo)
    assert region is not None
    assert region[3] >= 1220  # the bottom line is inside
    assert region[1] <= 60


def test_the_text_region_ignores_a_line_standing_alone_far_from_the_rest() -> None:
    card = [(900, 1500 + i * 60, 1700, 1540 + i * 60) for i in range(6)]
    stray = (100, 100, 300, 140)
    photo = np.zeros((4000, 3000, 3), dtype=np.uint8)
    region = _provider([*card, stray]).text_region(photo)
    assert region is not None
    assert region[1] > 1000


def test_too_few_lines_is_not_a_card() -> None:
    photo = np.zeros((1000, 1000, 3), dtype=np.uint8)
    assert _provider([(10, 10, 200, 40), (10, 60, 200, 90)]).text_region(photo) is None
