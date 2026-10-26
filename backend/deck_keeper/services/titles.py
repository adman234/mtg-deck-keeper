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


def for_outlined(img: np.ndarray, corners) -> tuple[list[list[float]], float]:
    """The title bar of a card outlined whole: looked for like a strip's (levelled, tight), in the
    card's top part, or else as for_card does."""
    _, src, _, _ = _upright(corners)
    tl, tr, br, bl = src
    found = in_strip(img, [tl, tr, tr + (br - tr) * TOP_PART, tl + (bl - tl) * TOP_PART])
    return found if found else for_card(img, corners)


def band_of_box(box: list[int], card_h: float | None) -> list[list[float]]:
    """Where the title should be in an upright box the model drew around a card (approximate)."""
    x, y, w, h = box
    th = (card_h or h) * (DEFAULT[1] - DEFAULT[0])
    top = y + (card_h or h) * DEFAULT[0]
    return [[x + w * 0.04, top], [x + w * 0.96, top], [x + w * 0.96, top + th], [x + w * 0.04, top + th]]


def expand(quad, left: float = 0.05, right: float = 0.02, vertical: float = 0.3) -> list[list[float]]:
    """A title bar's quad grown by fractions of its width (each end) and height (top and bottom):
    room around the name for reading it."""
    tl, tr, br, bl = (np.float32(p) for p in quad)
    a, d = tr - tl, bl - tl
    return [(tl - a * left - d * vertical).tolist(), (tr + a * right - d * vertical).tolist(),
            (br + a * right + d * vertical).tolist(), (bl - a * left + d * vertical).tolist()]


def cut(img: np.ndarray, quad, height: int = 72) -> np.ndarray:
    """The title bar straightened and scaled to a fixed height (what the model reads, and what
    the review screen shows next to the name)."""
    src = np.float32(quad)
    w = np.linalg.norm(src[1] - src[0])
    h = max(np.linalg.norm(src[3] - src[0]), 1.0)
    out_w = max(height, int(height * w / h))
    dst = np.float32([[0, 0], [out_w, 0], [out_w, height], [0, height]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (out_w, height), flags=cv2.INTER_CUBIC)


STRIP_W = 600  # a strip is straightened with its card this many pixels wide to look for the bar
STRIP_SIDE = 0.45  # ...reaching this far (of a card width) past either side, since its place across is rough
MAX_TILT = 8.0  # degrees: a card behind may lie turned this much from the one in front
TOP_PART = 0.17  # of a whole card's height: where to look for its title bar
BAR_STRIP_H = (0.04, 0.08)  # a title bar's height, in card heights
BAR_STRIP_W = (0.75, 0.97)  # its width, in card widths


def _tilt(gray: np.ndarray, min_len: float) -> float:
    """The angle of the long straight edges running across an image (card tops, title bar
    borders), in degrees: what to turn it by to level them."""
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=int(min_len * 0.5), minLineLength=int(min_len),
                            maxLineGap=8)
    if lines is None:
        return 0.0
    angles, weights = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        a = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if abs(a) <= MAX_TILT:
            angles.append(a)
            weights.append(float(np.hypot(x2 - x1, y2 - y1)))
    if not angles:
        return 0.0
    order = np.argsort(angles)
    cum = np.cumsum(np.array(weights)[order])
    return float(np.array(angles)[order][np.searchsorted(cum, cum[-1] / 2)])  # length-weighted median


def _runs(on: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """Runs of True in a 1-D mask, bridging gaps up to `gap` long: [(start, end)]."""
    out: list[tuple[int, int]] = []
    for i in np.flatnonzero(on):
        if out and i - out[-1][1] <= gap:
            out[-1] = (out[-1][0], int(i))
        else:
            out.append((int(i), int(i)))
    return out


def _bar_in_strip(level: np.ndarray, card_h: float, want_y: float) -> Bar | None:
    """The title bar in a levelled strip (card STRIP_W pixels wide): two rows of strong
    horizontal edges, a bar's height apart, near where the bar should be; its ends are where the
    top border stops. When the card in front hides the bar's bottom border, the strip's bottom
    edge stands in for it."""
    H, W = level.shape
    g = cv2.GaussianBlur(level, (3, 3), 0).astype(np.float32)
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    mid = gy[:, int(W * 0.5 - STRIP_W * 0.3):int(W * 0.5 + STRIP_W * 0.3)]
    rows = np.convolve(np.median(mid, axis=1) + mid.mean(axis=1), np.ones(3) / 3, "same")
    if rows.max() <= 0:
        return None
    rows = rows / rows.max()
    lo_h, hi_h = int(BAR_STRIP_H[0] * card_h), int(BAR_STRIP_H[1] * card_h)
    best = None
    for y1 in range(1, min(H - 4, int(want_y + 0.06 * card_h))):
        if rows[y1] < 0.25 or rows[y1] < rows[y1 - 1] or rows[y1] < rows[y1 + 1]:
            continue
        for y2 in range(y1 + lo_h, min(H, y1 + hi_h + 1)):
            if not (y2 >= H - 2 or (rows[y2] >= 0.25 and rows[y2] >= rows[y2 - 1] and rows[y2] >= rows[min(H - 1, y2 + 1)])):
                continue
            hidden = y2 >= H - 2
            score = rows[y1] + (0.3 if hidden else rows[y2]) - abs(y1 - want_y) / card_h
            if best is None or score > best[0]:
                best = (score, y1, y2)
    if best is None:
        return None
    _, y1, y2 = best
    # Across: the bar's ends. Its borders run unbroken from one end to the other and stop there,
    # and each end is an upright edge the bar's full height.
    border = gy[max(0, y1 - 2):y1 + 3].max(axis=0)
    if y2 < H - 2:
        border = np.minimum(border, gy[max(0, y2 - 2):y2 + 3].max(axis=0)) * 0.5 + border * 0.5
    on = (border > 0.35 * max(1.0, float(np.percentile(border, 95)))).astype(np.float32)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))[y1 + 2:max(y1 + 3, y2 - 1)]
    ends = np.convolve(np.median(gx, axis=0), np.ones(3) / 3, "same")
    ends = ends / max(1.0, float(ends.max()))
    cum = np.concatenate([[0.0], np.cumsum(on)])
    edge = int(STRIP_W * 0.04)
    best_x = None
    for w in range(int(BAR_STRIP_W[0] * STRIP_W), min(W, int(BAR_STRIP_W[1] * STRIP_W)), 3):
        for x1 in range(edge, W - w - edge, 2):
            x2 = x1 + w
            inside = (cum[x2] - cum[x1]) / w
            outside = ((cum[x1] - cum[x1 - edge]) + (cum[x2 + edge] - cum[x2])) / (2 * edge)
            score = inside - 0.6 * outside + 0.4 * (ends[x1] + ends[x2]) + 0.15 * w / STRIP_W
            if best_x is None or score > best_x[0]:
                best_x = (score, x1, x2)
    if best_x is None or best_x[0] < 0.5:
        return None
    _, x1, x2 = best_x
    return Bar(x1, y1, x2, y2, round(float(best[0]), 2))


def in_strip(img: np.ndarray, corners) -> tuple[list[list[float]], float] | None:
    """The title bar of a card in a title strip of a cascade (strip corners in img: the card's
    top edge down to the next card's), as a quad in img and how sure; None when no bar is found.
    The strip is straightened, levelled by its long edges (a card behind may lie a little turned)
    and searched for the bar's two borders, so the box sits tightly and level on the name."""
    tl, tr, br, bl = (np.float32(p) for p in corners)
    across = (tr - tl) / max(float(np.linalg.norm(tr - tl)), 1.0)
    card_w = float(np.linalg.norm(tr - tl))
    scale = STRIP_W / max(card_w, 1.0)
    strip_h = float(np.linalg.norm(bl - tl))
    side = across * card_w * STRIP_SIDE
    up = (tl - bl) / max(strip_h, 1.0) * card_w * 0.03  # a little above the card's top edge too
    src = np.float32([tl - side + up, tr + side + up, br + side, bl - side])
    W = int(STRIP_W * (1 + 2 * STRIP_SIDE))
    H = max(16, int((strip_h + card_w * 0.03) * scale))
    M = cv2.getPerspectiveTransform(src, np.float32([[0, 0], [W, 0], [W, H], [0, H]]))
    patch = cv2.cvtColor(cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
    angle = _tilt(patch, STRIP_W * 0.3)
    R = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    level = cv2.warpAffine(patch, R, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    card_h = STRIP_W * 88 / 63
    want_y = card_w * 0.03 * scale + 0.035 * card_h
    bar = _bar_in_strip(level, card_h, want_y)
    if bar is None:
        return None
    # Back to the photo: undo the levelling, then the straightening.
    Ri = cv2.invertAffineTransform(R)
    pts = np.float32(bar.quad())
    pts = (np.hstack([pts, np.ones((4, 1), np.float32)]) @ Ri.T).astype(np.float32)
    quad = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), np.linalg.inv(M)).reshape(4, 2)
    return quad.round(1).tolist(), bar.score
