import cv2
import numpy as np

from deck_keeper.services.detection import OUT_H, OUT_W, find_cards


def _table_with_cards(cards):
    img = np.full((1200, 1600, 3), (40, 90, 50), np.uint8)
    img = cv2.add(img, np.random.default_rng(0).integers(0, 25, img.shape, dtype=np.uint8))
    for cx, cy, angle in cards:
        for size, color in (((252, 352), (20, 20, 20)), ((226, 326), (180, 200, 220)), ((190, 140), (90, 60, 140))):
            box = cv2.boxPoints(((cx, cy), size, angle)).astype(np.int32)
            cv2.fillPoly(img, [box], color)
    return img


def test_finds_separate_cards_including_rotated_ones():
    img = _table_with_cards([(250, 300, 5), (650, 320, -8), (1100, 300, 90), (400, 850, 0), (900, 850, 15)])
    found = find_cards(img)
    assert len(found) == 5
    assert all(f.image.shape == (OUT_H, OUT_W, 3) for f in found)


def test_empty_table_finds_nothing():
    assert find_cards(_table_with_cards([])) == []
