import sys
from pathlib import Path

import cv2
import numpy as np

from deck_keeper.services import detection, plane, titles

sys.path.insert(0, str(Path(__file__).parent))
from cascade_photo import card  # noqa: E402


def _table(tilt: bool):
    rng = np.random.default_rng(4)
    img = np.empty((1400, 1800, 3), np.uint8)
    img[:] = (120, 70, 40)
    for x, y in ((150, 150), (550, 150), (950, 150), (150, 700), (550, 700), (950, 700)):
        img[y:y + 349, x:x + 250] = card(rng)
    if tilt:
        H, W = img.shape[:2]
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]),
                                        np.float32([[220, 90], [W - 60, 0], [W, H], [0, H - 120]]))
        img = cv2.warpPerspective(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    return img


def _square_error(img):
    quads = [np.float32(f.corners) for f in detection.find_cards(img)]
    return float(np.median([plane._corner_error(q) for q in quads])), len(quads)


def test_a_tilted_photo_is_straightened():
    img = _table(tilt=True)
    before, n = _square_error(img)
    view = plane.estimate([f.corners for f in detection.find_cards(img)], (img.shape[1], img.shape[0]))
    assert view is not None and before > 3
    after, m = _square_error(view.warp(img))
    assert after < 1.5 and m >= n


def test_a_straight_photo_is_left_alone():
    img = _table(tilt=False)
    assert plane.estimate([f.corners for f in detection.find_cards(img)], (img.shape[1], img.shape[0])) is None


def test_the_name_box_sits_on_the_title_bar():
    img = _table(tilt=False)
    f = detection.find_cards(img)[0]
    quad, _ = titles.for_card(img, f.corners)
    ys = [p[1] for p in quad]
    x, y, w, h = f.bbox
    # The synthetic card's title bar runs 20-44 px down a 349 px card.
    assert y <= min(ys) <= y + 0.09 * h and y + 0.06 * h <= max(ys) <= y + 0.15 * h
    strip = titles.cut(img, quad)
    assert strip.shape[0] == 72 and strip.shape[1] > 400
