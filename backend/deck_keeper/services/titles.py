"""Find the title bar of a card: the box with the card's name across its top.

Every identified card should come with a tight box around its name, in the photo and in the
crop the model reads. A title bar is the same shape on every card: about as wide as the card and
5-7% of its height, between two long straight lines (the box's top and bottom borders), near the
top of the card. Given the card's size in the (straightened) photo, bars are found as pairs of
long horizontal lines spaced like that.
"""

from dataclasses import dataclass

import cv2
import numpy as np

BAR_H = (0.03, 0.085)  # bar height range, in card heights
BAR_W = (0.6, 1.1)  # bar width range, in card widths
OVERLAP = 0.75  # top and bottom border lines overlap this much
# Where the title bar sits on a card, as fractions of its size, when its borders can't be seen.
DEFAULT = (0.035, 0.105, 0.05, 0.95)  # top, bottom, left, right


@dataclass
class Bar:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    def quad(self) -> list[list[float]]:
        return [[self.x1, self.y1], [self.x2, self.y1], [self.x2, self.y2], [self.x1, self.y2]]


def _lines(gray: np.ndarray, card_w: float) -> list[tuple[int, int, int]]:
    """Long horizontal lines: (y, x1, x2)."""
    g = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    mask = (gy > max(25.0, float(np.percentile(gy, 92)))).astype(np.uint8)
    k = max(5, int(card_w * 0.35))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    mask = cv2.dilate(mask, np.ones((1, 9), np.uint8))  # bridge small gaps (mana symbols, text)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if BAR_W[0] * card_w <= w <= BAR_W[1] * card_w * 1.1 and h <= max(6, card_w * 0.03):
            out.append((int(y + h // 2), int(x), int(x + w)))
    return sorted(out)


def find(gray: np.ndarray, card_w: float, card_h: float) -> list[Bar]:
    """Bar-shaped boxes in an image of upright cards of about this size, top first."""
    lines = _lines(gray, card_w)
    bars = []
    for i, (ya, a1, a2) in enumerate(lines):
        for yb, b1, b2 in lines[i + 1:]:
            dy = yb - ya
            if dy < BAR_H[0] * card_h:
                continue
            if dy > BAR_H[1] * card_h:
                break
            overlap = min(a2, b2) - max(a1, b1)
            shorter = min(a2 - a1, b2 - b1)
            if overlap < OVERLAP * shorter or shorter < BAR_W[0] * card_w:
                continue
            score = overlap / max(a2 - a1, b2 - b1)
            bars.append(Bar(max(a1, b1), ya, min(a2, b2), yb, round(float(score), 2)))
    # Overlapping candidates (a line shared by two pairs): keep the best of each cluster.
    bars.sort(key=lambda b: -b.score)
    kept: list[Bar] = []
    for b in bars:
        if not any(min(b.y2, k.y2) - max(b.y1, k.y1) > 0 for k in kept):
            kept.append(b)
    return sorted(kept, key=lambda b: b.y1)


def in_card(card: np.ndarray) -> Bar:
    """The title bar of one upright card image (a card cut out and straightened): a bar-shaped
    box where a title bar sits, or else the usual place."""
    h, w = card.shape[:2]
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    top = gray[: int(h * 0.2)]
    for b in find(top, w * 0.98, h * 0.98):
        if 0.025 * h <= b.y1 <= 0.12 * h and 0.045 * h <= b.y2 - b.y1 <= 0.085 * h:
            return b
    t, b, l, r = DEFAULT
    return Bar(int(l * w), int(t * h), int(r * w), int(b * h), 0.0)


def _upright(corners) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Transform that cuts a card out upright (title at the top), and the cut-out's size."""
    src = np.float32(corners)
    w = (np.linalg.norm(src[1] - src[0]) + np.linalg.norm(src[2] - src[3])) / 2
    h = (np.linalg.norm(src[3] - src[0]) + np.linalg.norm(src[2] - src[1])) / 2
    if w > h:  # lying sideways: turn it so its long side is upright
        src = np.roll(src, -1, axis=0)
        w, h = h, w
    W, H = max(8, int(w)), max(8, int(h))
    M = cv2.getPerspectiveTransform(src, np.float32([[0, 0], [W, 0], [W, H], [0, H]]))
    return M, src, W, H


def for_card(img: np.ndarray, corners) -> tuple[list[list[float]], float]:
    """The title bar of a card outlined at `corners` in img, as a quad in img, and how sure
    (0 when it's just where a title bar usually is)."""
    M, _, W, H = _upright(corners)
    card = cv2.warpPerspective(img, M, (W, H))
    bar = in_card(card)
    quad = cv2.perspectiveTransform(np.float32(bar.quad()).reshape(-1, 1, 2), np.linalg.inv(M)).reshape(4, 2)
    return quad.round(1).tolist(), bar.score


def band_of_box(box: list[int], card_h: float | None) -> list[list[float]]:
    """Where the title should be in an upright box the model drew around a card (approximate)."""
    x, y, w, h = box
    th = (card_h or h) * (DEFAULT[1] - DEFAULT[0])
    top = y + (card_h or h) * DEFAULT[0]
    return [[x + w * 0.04, top], [x + w * 0.96, top], [x + w * 0.96, top + th], [x + w * 0.04, top + th]]


def cut(img: np.ndarray, quad, height: int = 72) -> np.ndarray:
    """The title bar straightened and scaled to a fixed height (what the model reads, and what
    the review screen shows next to the name)."""
    src = np.float32(quad)
    w = np.linalg.norm(src[1] - src[0])
    h = max(np.linalg.norm(src[3] - src[0]), 1.0)
    out_w = max(height, int(height * w / h))
    dst = np.float32([[0, 0], [out_w, 0], [out_w, height], [0, height]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (out_w, height), flags=cv2.INTER_CUBIC)
