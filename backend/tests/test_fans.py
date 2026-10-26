import cv2
import numpy as np

from deck_keeper.services import detection, fans, piles, vision


def _fanned_pile(n: int, step: int = 42, seed: int = 1) -> tuple[np.ndarray, list[int]]:
    """A photo of n sleeved copies fanned to the left on a wooden table. Returns the image and
    the pile's box."""
    rng = np.random.default_rng(seed)
    img = np.empty((900, 1200, 3), np.uint8)
    img[:] = (40, 110, 190)  # orange-brown wood
    img[:] = np.clip(img.astype(int) + rng.integers(-12, 12, img.shape), 0, 255).astype(np.uint8)
    w, h = 300, 418
    right, top = 900, 220
    for k in reversed(range(n)):  # back card first
        x = right - w - k * step
        cv2.rectangle(img, (x, top), (x + w, top + h), (70, 70, 70), -1)  # sleeve
        cv2.rectangle(img, (x, top), (x + w, top + h), (25, 25, 25), 4)  # sleeve edge
        cv2.rectangle(img, (x + 9, top + 9), (x + w - 9, top + h - 9), (20, 20, 20), -1)  # card border
        cv2.rectangle(img, (x + 16, top + 16), (x + w - 16, top + h - 16), (60, 140, 60), -1)  # green frame
        cv2.rectangle(img, (x + 26, top + 60), (x + w - 26, top + 250), (90, 120, 150), -1)  # art
        cv2.rectangle(img, (x + 26, top + 270), (x + w - 26, top + h - 40), (200, 225, 235), -1)  # text box
    left = right - w - (n - 1) * step
    return img, [left, top, right - left, h]


def _count(img, box):
    found = [f for f in detection.find_cards(img) if piles._x_overlap(box, f.bbox) > 0.5]
    assert found
    front = max(found, key=lambda f: f.bbox[0])  # fanned left: the front card is on the right
    return fans.count_pile(img, box, front.corners)


def test_counts_the_slivers_of_a_fanned_pile():
    for n in (2, 3, 4, 5):
        img, box = _fanned_pile(n)
        result = _count(img, box)
        assert (result.count, result.direction) == (n, "left"), n


def test_counts_a_pile_photographed_at_an_angle():
    img, box = _fanned_pile(4)
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, np.float32([[120, 60], [w - 40, 0], [w, h], [0, h - 30]]))
    tilted = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    corners = cv2.perspectiveTransform(np.float32([[[box[0], box[1]]], [[box[0] + box[2], box[1] + box[3]]]]), M)
    (x1, y1), (x2, y2) = corners.reshape(2, 2).min(0), corners.reshape(2, 2).max(0)
    assert _count(tilted, [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]).count == 4


def test_a_pile_fanned_to_the_right_is_counted_too():
    img, box = _fanned_pile(3)
    mirrored = cv2.flip(img, 1)
    box = [img.shape[1] - box[0] - box[2], box[1], box[2], box[3]]
    found = [f for f in detection.find_cards(mirrored) if piles._x_overlap(box, f.bbox) > 0.5]
    result = fans.count_pile(mirrored, box, min(found, key=lambda f: f.bbox[0]).corners)
    assert (result.count, result.direction) == (3, "right")


def test_the_table_beyond_the_pile_is_not_counted():
    """The gap of table, and the next pile beyond it, aren't more slivers of this one."""
    img, box = _fanned_pile(3)
    x = box[0] - 70  # another pile's front card edge, a gap of wood away
    cv2.rectangle(img, (x - 300, box[1]), (x, box[1] + box[3]), (70, 70, 70), -1)
    cv2.rectangle(img, (x - 300, box[1]), (x, box[1] + box[3]), (25, 25, 25), 4)
    assert _count(img, box).count == 3


def test_sliver_count_wins_over_the_models():
    fan = fans.FanCount(4, "left", [0.8, 0.8, 0.8])
    assert piles.combine_counts(5, 6, fan) == (4, False)
    assert piles.combine_counts(5, 1, fans.FanCount(1, "left")) == (5, True)  # saw no slivers
    assert piles.combine_counts(5, None, fan) == (4, False)


def test_scan_counts_piles_from_their_slivers(tmp_path):
    """A pile the model calls 5 is counted 4 from its slivers once its front card is found."""
    img, box = _fanned_pile(4)
    h, w = img.shape[:2]
    card = vision.ReadCard(name="Nettle Sentinel", count=5, confidence=1.0,
                           extra={"box": [box[0] * 1000 // w, box[1] * 1000 // h,
                                          (box[0] + box[2]) * 1000 // w, (box[1] + box[3]) * 1000 // h]})
    piles.place_boxes([card], (w, h), (w, h))
    found = detection.find_cards(img)
    piles.snap_boxes([card], [f.bbox for f in found], [f.corners for f in found])
    piles.count_fans(img, [card])
    card.extra["zoom"] = vision.ZoomRead("Nettle Sentinel", 6, 0.9, "{}")
    from deck_keeper.services.matching import CardIndex
    d = piles.decide(CardIndex([(1, "Nettle Sentinel")]), card, [card])
    assert (d.count, d.count_differs) == (4, False)
