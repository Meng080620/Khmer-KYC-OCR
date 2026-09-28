"""Synthetic card images, so no real ID ever needs to live in the repo."""

import cv2
import numpy as np
from numpy.typing import NDArray

from khmer_kyc_ocr.pipeline.warp import CARD_H, CARD_W

Image = NDArray[np.uint8]


def make_card(seed: int = 0) -> Image:
    """A flat card: light background with dark blocks, so alignment errors are measurable."""
    rng = np.random.default_rng(seed)
    card = np.full((CARD_H, CARD_W, 3), 235, dtype=np.uint8)
    for _ in range(25):
        x, y = int(rng.integers(20, CARD_W - 120)), int(rng.integers(20, CARD_H - 60))
        w, h = int(rng.integers(40, 110)), int(rng.integers(15, 45))
        card[y : y + h, x : x + w] = rng.integers(0, 90)
    return card


def photograph(card: Image, *, angle_deg: float = 12.0, scale: float = 1.0) -> Image:
    """Place the card on a dark table with rotation and perspective, like a phone photo."""
    canvas_h, canvas_w = 1400, 1800
    src = np.array([[0, 0], [CARD_W, 0], [CARD_W, CARD_H], [0, CARD_H]], dtype=np.float32)
    center = np.array([canvas_w / 2, canvas_h / 2], dtype=np.float32)
    theta = np.deg2rad(angle_deg)
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32
    )
    half = np.array([CARD_W / 2, CARD_H / 2], dtype=np.float32)
    corners = ((src - half) * scale) @ rot.T + center
    corners[1] += (25, -15)  # mild keystone so it is a true perspective, not just a rotation
    matrix = cv2.getPerspectiveTransform(src, corners.astype(np.float32))
    table = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)
    warped = cv2.warpPerspective(
        card, matrix, (canvas_w, canvas_h), dst=table, borderMode=cv2.BORDER_TRANSPARENT
    )
    return np.asarray(warped, dtype=np.uint8)
