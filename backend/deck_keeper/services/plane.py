"""Straighten a photo of cards on a table into a top-down view.

A card is a 63 x 88 mm rectangle, so the four corners of any card OpenCV outlines give the
perspective of the whole table. The transform is taken from the card that also turns the other
outlined cards into upright rectangles of one size (a badly outlined card can't win), and the
whole photo is warped with it: then every card is an upright rectangle of the same size, which
makes the model's boxes, the close-ups and the title bars line up. Photos taken nearly straight
down are left as they are.
"""

from dataclasses import dataclass

import cv2
import numpy as np

CARD_RATIO = 63 / 88
TILT = 3.0  # degrees: a photo whose cards' corners are this close to square isn't warped
MAX_GROW = 1.6  # the straightened photo has at most this many times the pixels


@dataclass
class Plane:
    M: np.ndarray  # photo -> straightened
    size: tuple[int, int]  # straightened image (w, h)
    card: tuple[float, float]  # a card's size in the straightened image (w, h)

    def to_straight(self, pts) -> np.ndarray:
        return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), self.M).reshape(-1, 2)

    def to_photo(self, pts) -> np.ndarray:
        return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), np.linalg.inv(self.M)).reshape(-1, 2)

    def warp(self, img: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(img, self.M, self.size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _to_rectangle(corners) -> np.ndarray:
    """Homography that makes this card an upright card-shaped rectangle (keeping its size)."""
    src = np.float32(corners)
    w = (np.linalg.norm(src[1] - src[0]) + np.linalg.norm(src[2] - src[3])) / 2
    h = (np.linalg.norm(src[3] - src[0]) + np.linalg.norm(src[2] - src[1])) / 2
    if w > h:  # lying sideways: keep it sideways, just square it up
        cw, ch = w, w * CARD_RATIO
    else:
        cw, ch = h * CARD_RATIO, h
    dst = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    return cv2.getPerspectiveTransform(src, dst)


def _corner_error(quad: np.ndarray) -> float:
    """Degrees the corners are off square: perspective (a card itself is always square)."""
    err = 0.0
    for i in range(4):
        a, b, c = quad[i - 1], quad[i], quad[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
        err = max(err, abs(np.degrees(np.arccos(np.clip(cos, -1, 1))) - 90))
    return err


def _rotation(quad: np.ndarray) -> float:
    """Degrees the card's top edge is off level (or upright, for a card lying sideways)."""
    top = quad[1] - quad[0]
    a = float(np.degrees(np.arctan2(top[1], top[0]))) % 90
    return min(a, 90 - a)


def _shape_error(quad: np.ndarray) -> float:
    w = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    h = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    return abs(min(w, h) / max(w, h, 1e-9) - CARD_RATIO)


def estimate(found: list[list[list[float]]], photo: tuple[int, int]) -> Plane | None:
    """The table's top-down view from the outlined cards' corners, or None when the photo is
    already straight enough (or there's nothing to go on)."""
    quads = [np.float32(c) for c in found]
    # Cards lie at slightly different angles anyway; only perspective, or the whole photo being
    # turned, is worth a warp.
    if not quads or (np.median([_corner_error(q) for q in quads]) < TILT
                     and np.median([_rotation(q) for q in quads]) < TILT * 1.5):
        return None
    best, best_err = None, None
    for q in quads:
        H = _to_rectangle(q)
        mapped = [cv2.perspectiveTransform(o.reshape(-1, 1, 2), H).reshape(4, 2) for o in quads]
        long = [max(np.linalg.norm(m[1] - m[0]), np.linalg.norm(m[3] - m[0])) for m in mapped]
        size_spread = float(np.std(long) / (np.mean(long) + 1e-9))
        err = float(np.median([_corner_error(m) for m in mapped])) + float(np.median([_rotation(m) for m in mapped])) \
            + 100 * float(np.median([_shape_error(m) for m in mapped])) + 50 * size_spread
        if best_err is None or err < best_err:
            best, best_err = H, err
    W, Hh = photo
    corners = cv2.perspectiveTransform(np.float32([[0, 0], [W, 0], [W, Hh], [0, Hh]]).reshape(-1, 1, 2), best).reshape(4, 2)
    lo, hi = corners.min(0), corners.max(0)
    # Keep about the photo's own resolution: scale so the cards are as big as they were.
    mapped = [cv2.perspectiveTransform(q.reshape(-1, 1, 2), best).reshape(4, 2) for q in quads]
    card_now = float(np.median([np.linalg.norm(m[3] - m[0]) for m in mapped]))
    card_before = float(np.median([np.linalg.norm(q[3] - q[0]) for q in quads]))
    s = card_before / max(card_now, 1e-9)
    area = (hi - lo).prod() * s * s
    if area > MAX_GROW * W * Hh:
        s *= float(np.sqrt(MAX_GROW * W * Hh / area))
    M = np.array([[s, 0, -lo[0] * s], [0, s, -lo[1] * s], [0, 0, 1]]) @ best
    size = (int((hi - lo)[0] * s), int((hi - lo)[1] * s))
    cw = float(np.median([min(np.linalg.norm(m[1] - m[0]), np.linalg.norm(m[3] - m[0])) for m in mapped])) * s
    return Plane(M, size, (cw, cw / CARD_RATIO))
