import numpy as np
import pytest
from numpy.typing import NDArray

from khmer_kyc_ocr.pipeline.warp import CARD_H, CARD_W, find_card
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
