import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from khmer_kyc_ocr.pipeline.warp import CARD_H, CARD_W, Frame, find_card, find_card_in_frame
from tests.synthetic import make_card, photograph


def _error(a: NDArray[np.uint8], b: NDArray[np.uint8]) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


@pytest.mark.parametrize("angle", [-20.0, 0.0, 8.0, 25.0, 90.0, 180.0, 265.0])
def test_recovers_landscape_card_from_rotated_photo(angle: float) -> None:
    card = make_card(seed=1)
    found = find_card(photograph(card, angle_deg=angle))

    assert found is not None
    assert found.shape == (CARD_H, CARD_W, 3)
    # A rectangle is 180°-symmetric, so geometry alone can only promise upright-or-flipped;
    # the service resolves which one using the ID-number field.
    # Mean absolute pixel error; a misalignment of even ~10px blows past the threshold.
    assert min(_error(found, card), _error(found, np.rot90(card, 2))) < 12


def test_returns_none_when_no_card_present() -> None:
    blank = np.full((1200, 1600, 3), 30, dtype=np.uint8)
    assert find_card(blank) is None


def test_returns_none_when_card_is_a_tiny_part_of_the_photo() -> None:
    assert find_card(photograph(make_card(), scale=0.2)) is None


def _on_light_table(card: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], Frame]:
    """A light card lying straight on a table of the same shade: nothing marks its edge.
    Returns the photo and where a guide frame around the card would be."""
    photo = np.full((2000, 1500, 3), 235, dtype=np.uint8)
    width = 1100
    height = round(width * CARD_H / CARD_W)
    x, y = 200, 650
    photo[y : y + height, x : x + width] = cv2.resize(card, (width, height))
    return photo, (x / 1500, y / 2000, width / 1500, height / 2000)


def test_light_card_on_light_table_is_not_found_by_its_edges() -> None:
    photo, _ = _on_light_table(make_card(seed=2))
    assert find_card(photo) is None


def test_guide_frame_gives_the_card_when_its_edges_cannot_be_found() -> None:
    card = make_card(seed=2)
    photo, frame = _on_light_table(card)

    found = find_card_in_frame(photo, frame)

    assert found is not None
    assert found.shape == (CARD_H, CARD_W, 3)
    assert _error(found, card) < 12


def test_guide_frame_still_straightens_a_card_whose_edges_show_inside_it() -> None:
    card = make_card(seed=3)
    photo = photograph(card, angle_deg=6.0)  # dark table: edges are there to find
    # A frame drawn loosely around the tilted card.
    frame = (0.2, 0.2, 0.6, 0.6)

    found = find_card_in_frame(photo, frame)

    assert found is not None
    assert min(_error(found, card), _error(found, np.rot90(card, 2))) < 12


def test_guide_frame_is_kept_inside_the_photo() -> None:
    photo, _ = _on_light_table(make_card())
    found = find_card_in_frame(photo, (0.0, 0.0, 1.0, 1.0))
    assert found is not None and found.shape == (CARD_H, CARD_W, 3)
