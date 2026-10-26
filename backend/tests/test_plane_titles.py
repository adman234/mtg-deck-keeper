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


def test_a_strip_title_box_is_tight_and_level_on_a_turned_card():
    """A card in a column lying a little turned, and the strip placed a little to one side: the
    name box still sits exactly on the title bar, along it."""
    rng = np.random.default_rng(5)
    img = np.full((900, 900, 3), (120, 70, 40), np.uint8)
    c = card(rng, 250, 349)
    x0, y0 = 320, 250
    img[y0:y0 + 349, x0:x0 + 250] = c
    M = cv2.getRotationMatrix2D((x0 + 125, y0 + 175), -4, 1.0)  # turned 4 degrees
    img = cv2.warpAffine(img, M, (900, 900), borderMode=cv2.BORDER_REPLICATE)
    rot = lambda pts: (np.hstack([np.float32(pts), np.ones((len(pts), 1), np.float32)]) @ M.T)
    bar = rot([[x0 + 20, y0 + 20], [x0 + 229, y0 + 20], [x0 + 229, y0 + 44], [x0 + 20, y0 + 44]])
    # The strip as the column walk might give it: unturned, 30 px right, down to where the next card lies.
    strip = [[x0 + 30, y0], [x0 + 280, y0], [x0 + 280, y0 + 60], [x0 + 30, y0 + 60]]
    quad, _ = titles.in_strip(img, strip)
    assert np.abs(np.float32(quad) - bar).max() < 6, (quad, bar.round())
