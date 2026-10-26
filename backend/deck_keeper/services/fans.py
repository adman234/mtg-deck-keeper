"""Count the cards in a fanned pile from the edges they show (OpenCV, no model calls).

Copies of a card are usually fanned so each card behind the front one shows a sliver at one
side. Every sliver starts with the same edge (sleeve edge, card border, frame) as the front
card, so the front card's edge is used as a template and its repeats are counted outwards.
The table is first straightened using a card OpenCV found (a card is a 63 x 88 rectangle, so
its four corners give the perspective of the whole table), which makes the edges upright.

Only cards that show an edge can be counted: a card hidden exactly behind another is missed,
so the result is a minimum.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

CARD_W = 300  # straightened card size in pixels
CARD_H = round(CARD_W * 88 / 63)
EDGE_W = 0.06  # template: this much of a card's width from its edge inwards
BAND = (0.1, 0.9)  # rows of the card used (clear of the stepped tops and bottoms)
MIN_STEP, MAX_STEP = 0.05, 0.45  # gap between neighbouring slivers, in card widths
STEP_RATIO = 0.6
MATCH = 0.4  # template match needed to count an edge
SLIVER_ALIKE = 0.3  # and a sliver must look this much like the one before it
SLIVER_COLOUR = 15.0  # and differ this little in colour (Lab a/b)


@dataclass
class FanCount:
    count: int
    direction: str  # left | right, relative to the front card standing upright
    edges: list[float] = field(default_factory=list)  # match scores of the counted edges


def _homography(corners: list[list[float]], at: tuple[float, float]) -> np.ndarray:
    """Map the table so this card becomes an upright CARD_W x CARD_H rectangle at `at`."""
    src = np.array(corners, dtype="float32")
    w = np.linalg.norm(src[1] - src[0]) + np.linalg.norm(src[2] - src[3])
    h = np.linalg.norm(src[3] - src[0]) + np.linalg.norm(src[2] - src[1])
    if w > h:  # card lies sideways: rotate so it stands upright
        src = np.roll(src, -1, axis=0)
    x, y = at
    dst = np.array([[x, y], [x + CARD_W, y], [x + CARD_W, y + CARD_H], [x, y + CARD_H]], dtype="float32")
    return cv2.getPerspectiveTransform(src, dst)


def _straighten(img: np.ndarray, pile: list[int], corners: list[list[float]]) -> tuple[np.ndarray, list[float], list[float]]:
    """Straightened view of the pile's area, where the reference card ends up in it, and the
    pile box's left and right ends in it."""
    H = _homography(corners, (0, 0))
    x, y, w, h = pile
    box = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype="float32")
    ref = np.array([[[0, 0]], [[CARD_W, CARD_H]]], dtype="float32")
    pts = np.concatenate([cv2.perspectiveTransform(box, H), ref])
    lo = pts.min(axis=(0, 1)) - CARD_W * 0.15
    hi = pts.max(axis=(0, 1)) + CARD_W * 0.15
    size = hi - lo
    if size.max() > CARD_W * 8:
        raise ValueError("pile too far from the reference card")
    shift = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]], [0, 0, 1]])
    view = cv2.warpPerspective(img, shift @ H, (int(size[0]), int(size[1])), flags=cv2.INTER_LINEAR)
    ends = pts[:4, 0, 0] - lo[0]
    return view, [-lo[0], -lo[1]], [float(ends.min()), float(ends.max())]


def _long_edges(gray: np.ndarray) -> np.ndarray:
    """Per column, the share of rows with a strong left-right brightness change: card and sleeve
    edges run the full height, text and art don't."""
    g = cv2.GaussianBlur(gray, (3, 3), 0).astype(float)
    gx = cv2.dilate(np.abs(cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)), np.ones((1, 3)))
    return (gx > max(40.0, float(np.percentile(gx, 90)))).mean(axis=0)


def _chain(scores: np.ndarray, start: int, stop: int, alike) -> list[tuple[int, float]]:
    """Walk left from `start` along template matches spaced like slivers, no further than `stop`.
    A step must be at least STEP_RATIO of the previous one: a sliver's sleeve edge and card edge
    sit close together and both look like an edge. alike(new, previous, width) checks the new
    sliver looks like the previous one, so a gap of table and the next pile's edge don't count."""
    out = []
    e, step = start, None
    while True:
        near = MIN_STEP * CARD_W if step is None else max(MIN_STEP * CARD_W, STEP_RATIO * step)
        lo, hi = max(stop, int(e - MAX_STEP * CARD_W)), int(e - near)
        if hi <= lo:
            break
        window = scores[lo:hi + 1]
        best = window.max()
        if best < MATCH:
            break
        # The nearest good match: a farther one would skip a sliver.
        peaks = [i for i in range(len(window)) if window[i] >= max(MATCH, best * 0.8)
                 and (i == 0 or window[i] >= window[i - 1]) and (i == len(window) - 1 or window[i] >= window[i + 1])]
        new = lo + peaks[-1]
        width = (e - new) if step is None else min(e - new, step)
        if not alike(new, e, width):
            break
        step, e = e - new, new
        out.append((e, float(scores[e])))
    return out


def _count_left(view: np.ndarray, front_x: float, top: float, pile_left: float) -> list[tuple[int, float]] | None:
    """Edges of the cards behind the front one, walking left from the front card's left edge.
    None when the view doesn't reach far enough to measure."""
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    y1, y2 = int(top + BAND[0] * CARD_H), int(top + BAND[1] * CARD_H)
    if y1 < 0 or y2 > gray.shape[0]:
        return None
    # OpenCV often outlines the card inside the sleeve: start from the sleeve's outer edge, the
    # strongest full-height edge just outside the outline, since every sliver shows that much.
    lo, hi = int(front_x - 0.09 * CARD_W), int(front_x + 0.01 * CARD_W)
    if lo < 0:
        return None
    edges = _long_edges(gray[y1:y2, lo:hi])
    x = lo + int(np.argmax(edges)) if edges.max() > 0.3 else int(round(front_x))
    tw = max(6, int(EDGE_W * CARD_W))
    if x + tw > gray.shape[1]:
        return None
    tmpl = gray[y1:y2, x:x + tw]
    dy = int(0.03 * CARD_H)
    region = gray[max(0, y1 - dy):min(gray.shape[0], y2 + dy), :x + tw]
    r = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED).max(axis=0)
    lab = cv2.cvtColor(view, cv2.COLOR_BGR2LAB).astype(float)

    def alike(new: int, prev: int, width: int) -> bool:
        a = gray[y1:y2, prev:prev + width]
        b = gray[max(0, y1 - dy):min(gray.shape[0], y2 + dy), new:new + width]
        if a.size == 0 or b.shape[0] < a.shape[0] or b.shape[1] < a.shape[1]:
            return False
        if float(cv2.matchTemplate(b, a, cv2.TM_CCOEFF_NORMED).max()) < SLIVER_ALIKE:
            return False
        # Same sleeves: the colour hardly changes from one sliver to the next (the table does).
        ca = np.median(lab[y1:y2, prev:prev + width].reshape(-1, 3), axis=0)
        cb = np.median(lab[y1:y2, new:new + width].reshape(-1, 3), axis=0)
        return float(np.linalg.norm(ca[1:] - cb[1:])) <= SLIVER_COLOUR

    return _chain(r, x, max(0, int(pile_left - 0.12 * CARD_W)), alike)


def count_pile(img: np.ndarray, pile: list[int], corners: list[list[float]]) -> FanCount | None:
    """Count the cards in the pile boxed by `pile` ([x, y, w, h]) whose front card OpenCV
    outlined at `corners`. Tries fans to the left and to the right (up and down fans show as
    left and right once a sideways front card is turned upright). None when it can't measure."""
    try:
        view, (ox, oy), (left, right) = _straighten(img, pile, corners)
    except (ValueError, cv2.error):
        return None
    width = view.shape[1]
    results = []
    for direction, v, fx, end in (("left", view, ox, left),
                                  ("right", cv2.flip(view, 1), width - ox - CARD_W, width - right)):
        edges = _count_left(v, fx, oy, end)
        if edges is not None:
            results.append((direction, edges))
    if not results:
        return None
    direction, edges = max(results, key=lambda r: len(r[1]))
    return FanCount(1 + len(edges), direction, [round(s, 2) for _, s in edges])
