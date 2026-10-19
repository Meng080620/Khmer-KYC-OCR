"""Locate the ID card in a photo and flatten it to a fixed-size, landscape image."""

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]

# ISO/IEC 7810 ID-1 (85.60 x 53.98 mm), the size of Cambodian ID cards, scaled to 1000px wide.
CARD_W, CARD_H = 1000, 631
# A card smaller than this share of the frame is too low-resolution to read reliably;
# the client should ask for a closer photo instead of us guessing.
MIN_AREA_FRACTION = 0.15

# Where the caller says the card is: (x, y, width, height) as fractions of the photo,
# from the guide frame the person lined the card up in.
Frame = tuple[float, float, float, float]
# How far outside the guide frame to look for the card's edges: people overshoot a little.
_FRAME_MARGIN = 0.12
# A quad found inside the frame only counts as the card if it is card-shaped and fills
# most of the frame. Otherwise it is the portrait box or a block of text, not the card.
_CARD_RATIO = CARD_W / CARD_H
_RATIO_TOLERANCE = 0.25
_MIN_FRAME_FILL = 0.5


def _order_corners(pts: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return corners clockwise from top-left, with the long edge first so output is landscape."""
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]  # clockwise in image coords (y points down)
    ordered = np.roll(ordered, -int(np.argmin(ordered.sum(axis=1))), axis=0)
    top = np.linalg.norm(ordered[1] - ordered[0])
    side = np.linalg.norm(ordered[2] - ordered[1])
    if side > top:  # card is portrait in the photo: rotate the corner order by one
        ordered = np.roll(ordered, -1, axis=0)
    return ordered.astype(np.float32)


def _largest_quad(image: Image) -> NDArray[np.float32] | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    # Close small gaps in the card border (glare, fingers) so it forms one contour.
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = MIN_AREA_FRACTION * image.shape[0] * image.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < min_area:
            return None
        hull = cv2.convexHull(contour)
        approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        if len(approx) == 4:
            quad: NDArray[np.float32] = approx.reshape(4, 2).astype(np.float32)
            return quad
    return None


def _flatten(image: Image, quad: NDArray[np.float32]) -> Image:
    target = np.array(
        [[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(_order_corners(quad), target)
    card = cv2.warpPerspective(image, matrix, (CARD_W, CARD_H), flags=cv2.INTER_CUBIC)
    return np.asarray(card, dtype=np.uint8)  # no copy: already uint8, this only narrows the type


def _frame_box(image: Image, frame: Frame, margin: float) -> tuple[int, int, int, int] | None:
    """The frame in pixels, grown by ``margin`` and kept inside the photo. None if degenerate."""
    height, width = image.shape[:2]
    x, y, w, h = frame
    left, right = (x - w * margin) * width, (x + w * (1 + margin)) * width
    top, bottom = (y - h * margin) * height, (y + h * (1 + margin)) * height
    x0, x1 = max(0, round(left)), min(width, round(right))
    y0, y1 = max(0, round(top)), min(height, round(bottom))
    return None if x1 - x0 < 2 or y1 - y0 < 2 else (x0, y0, x1, y1)


def frame_region(image: Image, frame: Frame) -> Image:
    """The framed part of the photo, with the usual margin around it. The whole photo if
    the frame is degenerate."""
    box = _frame_box(image, frame, _FRAME_MARGIN)
    if box is None:
        return image
    x0, y0, x1, y1 = box
    return np.ascontiguousarray(image[y0:y1, x0:x1])


def _looks_like_the_framed_card(quad: NDArray[np.float32], frame_area: float) -> bool:
    (_, (side_a, side_b), _) = cv2.minAreaRect(quad)
    short, long = sorted((float(side_a), float(side_b)))
    if short <= 0:
        return False
    card_shaped = abs(long / short - _CARD_RATIO) <= _CARD_RATIO * _RATIO_TOLERANCE
    return card_shaped and float(cv2.contourArea(quad)) >= _MIN_FRAME_FILL * frame_area


def find_card_in_frame(image: Image, frame: Frame) -> Image | None:
    """The card from where the caller's guide frame says it is, for when its edges can't be
    found in the whole photo (a light card on a light table has no edge to find).

    First look for the edges again in just the framed part, where the card is the main
    thing in view; that still corrects tilt and perspective. Failing that, take the frame
    itself as the card: the person lined it up there, and reading a slightly loose crop
    beats refusing the photo.
    """
    exact = _frame_box(image, frame, 0.0)
    loose = _frame_box(image, frame, _FRAME_MARGIN)
    if exact is None or loose is None:
        return None
    ex0, ey0, ex1, ey1 = exact
    lx0, ly0, lx1, ly1 = loose

    region = image[ly0:ly1, lx0:lx1]
    quad = _largest_quad(region)
    if quad is not None and _looks_like_the_framed_card(quad, (ex1 - ex0) * (ey1 - ey0)):
        return _flatten(np.ascontiguousarray(region), quad)

    corners = np.array([[ex0, ey0], [ex1, ey0], [ex1, ey1], [ex0, ey1]], dtype=np.float32)
    return _flatten(image, corners)


def find_card(image: Image) -> Image | None:
    """Flattened CARD_W x CARD_H card, upright or rotated 180°, or None if no card is found.

    A rectangle is 180°-symmetric, so geometry cannot tell upright from upside-down;
    the caller resolves that with content (see service.py).
    """
    quad = _largest_quad(image)
    return None if quad is None else _flatten(image, quad)
