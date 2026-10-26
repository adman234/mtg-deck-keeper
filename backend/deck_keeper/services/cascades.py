"""Find the cards in cascading columns: each card laid over the one behind it so only the title
bars of the cards behind show (the usual way to photograph a 100-card deck).

A small model can't place 100 title bars in one photo, but OpenCV can find them: the card at the
front of a column is fully visible, and its four corners give the table's perspective. Straightened
that way, every card behind it shows the same top edge (sleeve, border, title bar frame) a little
further up, so the front card's top edge is used as a template and followed up the column, one
card per step, allowing for the column leaning a little to one side. Each title strip between two
edges is then cut out of the full-resolution photo, straightened and enlarged, for the model to
read one title at a time.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from .fans import _homography, _long_edges

CARD_W = 300  # straightened card size used to find the edges
CARD_H = round(CARD_W * 88 / 63)
EDGE_H = 0.065  # template: this much of a card's height from its top edge down (sleeve top to title bar)
BAND = (0.12, 0.88)  # columns of the card compared for colour (clear of the corners)
SIDE = 0.03  # the edge template reaches this far past the card's sides
TOP_SEARCH = 0.08  # the front card's top may be this far above its outline (sleeve, or a box inside the card)
MIN_STEP, MAX_STEP = 0.05, 0.3  # gap between neighbouring title bars, in card heights
FIRST_STEP = 0.07  # a cascade shows at least a whole title bar; lines inside one are closer
STEP_RATIO = 0.6
DRIFT = 0.12  # the column may lean this much of a card's width per card
MATCH = 0.45
STRIP_ALIKE = 0.25
STRIP_COLOUR = 18.0
MIN_STRIPS = 2  # fewer title bars than this above a card isn't a column
MAX_STRIPS = 40
READ_W = 900  # title strips are cut out this wide for the model
TITLE = (0.025, 0.11)  # the title bar's top and bottom below a card's top edge, in card heights


@dataclass
class Strip:
    corners: list[list[float]]  # in the photo: top-left, top-right, bottom-right, bottom-left
    bbox: list[int]  # x, y, w, h in the photo
    score: float
    title: list[list[float]] | None = None  # the title bar within the strip, in the photo


@dataclass
class Column:
    front: int  # index of the fully visible card it ends with
    strips: list[Strip] = field(default_factory=list)  # nearest the front card first


def _view(img: np.ndarray, corners: list[list[float]]) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """The table straightened around a card, reaching up far enough for a long column."""
    H = _homography(corners, (0, 0))
    up = CARD_H * MAX_STRIPS * 0.12
    shift = np.array([[1, 0, CARD_W], [0, 1, up], [0, 0, 1]], dtype=float)
    M = shift @ H
    view = cv2.warpPerspective(img, M, (CARD_W * 3, int(up + CARD_H * 1.1)), flags=cv2.INTER_LINEAR)
    return view, M, (float(CARD_W), up)


def _top_candidates(gray: np.ndarray, x: int, y: int) -> list[int]:
    """Lines that could be the front card's top edge. OpenCV may have outlined the card, its
    sleeve, or a rectangle inside it (the frame), so every strong full-width edge from a little
    above the outline to a little inside it is a candidate."""
    lo, hi = max(0, int(y - TOP_SEARCH * CARD_H)), int(y + 0.1 * CARD_H)
    band = gray[lo:hi, int(x + BAND[0] * CARD_W):int(x + BAND[1] * CARD_W)]
    if band.size == 0:
        return [y]
    strength = np.convolve(_long_edges(band.T), np.ones(3) / 3, "same")
    peaks = [lo + i for i in range(1, len(strength) - 1) if strength[i] >= 0.3
             and strength[i] >= strength[i - 1] and strength[i] >= strength[i + 1]]
    return peaks or [y]


def walk(view: np.ndarray, x: float, y: float) -> tuple[int, list[tuple[int, int, float]]]:
    """The front card's top edge and the top edges of the cards behind it, walking up from the
    card outlined at (x, y) in the straightened view: (top, [(top, left, score)], nearest first).
    Of the walks from each candidate top edge, the one whose edges match best (ties: the longer)."""
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(view, cv2.COLOR_BGR2LAB).astype(float)
    best: tuple[int, list[tuple[int, int, float]]] = (int(round(y)), [])
    for top in _top_candidates(gray, int(round(x)), int(round(y))):
        edges = _walk_from(gray, lab, int(round(x)), top)
        if len(edges) >= MIN_STRIPS and (not best[1] or _quality(edges) > _quality(best[1])):
            best = (top, edges)
    return best


def _quality(edges: list[tuple[int, int, float]]) -> tuple[float, int]:
    return round(float(np.mean([s for _, _, s in edges])), 2), len(edges)


def _walk_from(gray: np.ndarray, lab: np.ndarray, x0: int, top: int) -> list[tuple[int, int, float]]:
    b0, b1 = int(BAND[0] * CARD_W), int(BAND[1] * CARD_W)
    th = max(6, int(EDGE_H * CARD_H))
    # The whole width, corners included: a top edge alone is a line that matches anywhere along it.
    m = int(SIDE * CARD_W)
    t0, t1 = -m, CARD_W + m
    tmpl = gray[top:top + th, x0 + t0:x0 + t1]
    if x0 + t0 < 0 or tmpl.shape[0] < th or tmpl.shape[1] < t1 - t0:
        return []
    drift = int(DRIFT * CARD_W)
    out: list[tuple[int, int, float]] = []
    e, left, step = top, x0, None

    def strip(t: int, b: int, lx: int) -> np.ndarray:
        return lab[t:b, lx + b0:lx + b1]

    while len(out) < MAX_STRIPS:
        near = FIRST_STEP * CARD_H if step is None else max(MIN_STEP * CARD_H, STEP_RATIO * step)
        lo, hi = int(e - MAX_STEP * CARD_H), int(e - near)
        if lo < 0 or hi <= lo:
            break
        a, b = max(0, left + t0 - drift), min(gray.shape[1], left + t1 + drift)
        region = gray[lo:hi + th, a:b]
        if region.shape[0] < th or region.shape[1] < tmpl.shape[1]:
            break
        r = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)  # rows: top edge lo..hi, cols: shift
        rows = r.max(axis=1)
        best = float(rows.max())
        if best < MATCH:
            break
        good = max(MATCH, best * 0.8)
        peaks = [i for i in range(len(rows)) if rows[i] >= good
                 and (i == 0 or rows[i] >= rows[i - 1]) and (i == len(rows) - 1 or rows[i] >= rows[i + 1])]
        if step is None:
            peaks.sort(reverse=True)  # nearest first
        else:
            peaks.sort(key=lambda i: abs((e - (lo + i)) - step))
        chosen = None
        for i in peaks:
            new = lo + i
            nl = a + int(np.argmax(r[i])) - t0
            # The strip just found must look like the one before it (same sleeves and frames);
            # the table or a different column won't.
            prev = strip(e, e + (e - new), left) if step is None else strip(e, e + step, left)
            cur = strip(new, e, nl)
            if prev.size == 0 or cur.size == 0:
                continue
            if np.linalg.norm(np.median(prev.reshape(-1, 3), 0)[1:] - np.median(cur.reshape(-1, 3), 0)[1:]) > STRIP_COLOUR:
                continue
            chosen = (new, nl, float(rows[i]))
            break
        if chosen is None:
            break
        new, nl, score = chosen
        step, e, left = e - new, new, nl
        out.append((e, left, score))
    return out


def find(img: np.ndarray, found: list[list[list[float]]]) -> list[Column]:
    """Columns ending in each card OpenCV found whole (given by its corners)."""
    columns = []
    for i, corners in enumerate(found):
        try:
            view, M, (x, y) = _view(img, corners)
        except cv2.error:
            continue
        below, edges = walk(view, x, y)
        if len(edges) < MIN_STRIPS:
            continue
        inv = np.linalg.inv(M)
        col = Column(i)
        for top, left, score in edges:
            quad = np.float32([[[left, top]], [[left + CARD_W, top]], [[left + CARD_W, below]], [[left, below]]])
            pts = cv2.perspectiveTransform(quad, inv).reshape(4, 2)
            x1, y1 = pts.min(0)
            x2, y2 = pts.max(0)
            # The title bar: where it sits on a card, as far down as the strip shows.
            t, b = top + TITLE[0] * CARD_H, min(below, top + TITLE[1] * CARD_H)
            l, r = left + 0.04 * CARD_W, left + 0.96 * CARD_W
            title = cv2.perspectiveTransform(np.float32([[[l, t]], [[r, t]], [[r, b]], [[l, b]]]), inv).reshape(4, 2)
            col.strips.append(Strip(pts.tolist(), [int(x1), int(y1), int(x2 - x1), int(y2 - y1)], round(score, 2),
                                    title.round(1).tolist()))
            below = top
        columns.append(col)
    return columns


def cut(img: np.ndarray, strip: Strip) -> np.ndarray:
    """The title strip straightened and enlarged for reading."""
    src = np.float32(strip.corners)
    w = np.linalg.norm(src[1] - src[0])
    h = np.linalg.norm(src[3] - src[0])
    out_h = max(24, int(READ_W * h / max(w, 1)))
    dst = np.float32([[0, 0], [READ_W, 0], [READ_W, out_h], [0, out_h]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (READ_W, out_h), flags=cv2.INTER_CUBIC)
