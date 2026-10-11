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


def test_lone_art_box_is_dropped():
    """A card whose black border blends into a dark mat is found only by its art box; that crop
    has no title line, so it is dropped when the photo's other cards are full size."""
    img = _table_with_cards([(250, 300, 0), (650, 300, 0), (1050, 300, 0)])
    # The fourth card: border same colour as the mat, so only the art frame stands out.
    cv2.fillPoly(img, [cv2.boxPoints(((650, 850), (252, 352), 0)).astype(np.int32)], (40, 90, 50))
    cv2.fillPoly(img, [cv2.boxPoints(((650, 800), (180, 132), 0)).astype(np.int32)], (200, 210, 230))
    found = find_cards(img)
    assert len(found) == 3
    assert all(f.bbox[1] < 600 for f in found)
