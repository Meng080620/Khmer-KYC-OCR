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


def find_card(image: Image) -> Image | None:
    """Flattened CARD_W x CARD_H card, upright or rotated 180°, or None if no card is found.

    A rectangle is 180°-symmetric, so geometry cannot tell upright from upside-down;
    the caller resolves that with content (see service.py).
    """
    quad = _largest_quad(image)
    if quad is None:
        return None
    target = np.array(
        [[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(_order_corners(quad), target)
    card = cv2.warpPerspective(image, matrix, (CARD_W, CARD_H), flags=cv2.INTER_CUBIC)
    return np.asarray(card, dtype=np.uint8)  # no copy: already uint8, this only narrows the type
