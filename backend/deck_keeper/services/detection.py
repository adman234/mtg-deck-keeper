"""OpenCV pre-pass: find card-shaped rectangles and cut each one out, flattened and upright.

Small vision models read one card far more reliably than a whole table of cards.
Fanned or overlapping cards won't be found; the scanner then relies on the
whole-photo pass instead.
"""

from dataclasses import dataclass

import cv2
import numpy as np

CARD_RATIO = 63 / 88  # width / height, about 0.716
RATIO_TOLERANCE = 0.12
OUT_W, OUT_H = 488, 680


@dataclass
class FoundCard:
    bbox: list[int]  # x, y, w, h in the source image
    image: np.ndarray  # upright BGR crop


def _order(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype("float32")
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")


def _iou(a: list[int], b: list[int]) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / float(a[2] * a[3] + b[2] * b[3] - inter or 1)


def _candidates(gray: np.ndarray) -> list[np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(blurred, 40, 140), np.ones((3, 3), np.uint8), iterations=2)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)
    out = []
    for mask in (edges, thresh, 255 - thresh):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        out.extend(contours)
    return out


def find_cards(img: np.ndarray, max_cards: int = 100) -> list[FoundCard]:
    h, w = img.shape[:2]
    area = h * w
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    quads: list[tuple[float, np.ndarray, list[int]]] = []
    for c in _candidates(gray):
        a = cv2.contourArea(c)
        if a < area * 0.004 or a > area * 0.95:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        rect = cv2.minAreaRect(approx)
        rw, rh = rect[1]
        if min(rw, rh) == 0:
            continue
        ratio = min(rw, rh) / max(rw, rh)
        if abs(ratio - CARD_RATIO) > RATIO_TOLERANCE:
            continue
        # Reject shapes that are far from rectangular (rect fill ratio).
        if a / (rw * rh) < 0.85:
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        quads.append((a, approx, [int(x), int(y), int(bw), int(bh)]))

    # Largest first; drop duplicates (same card found by several masks) and inner boxes (art frames).
    quads.sort(key=lambda q: -q[0])
    kept: list[tuple[float, np.ndarray, list[int]]] = []
    for q in quads:
        bx = q[2]
        if any(_iou(bx, k[2]) > 0.4 or _contains(k[2], bx) for k in kept):
            continue
        kept.append(q)
        if len(kept) >= max_cards:
            break

    # Reading order: top-to-bottom rows, then left-to-right.
    kept.sort(key=lambda q: (round(q[2][1] / max(1, q[2][3] * 0.5)), q[2][0]))
    found = []
    for _, approx, bbox in kept:
        found.append(FoundCard(bbox=bbox, image=_warp(img, approx)))
    return found


def _contains(outer: list[int], inner: list[int]) -> bool:
    return (inner[0] >= outer[0] - 4 and inner[1] >= outer[1] - 4
            and inner[0] + inner[2] <= outer[0] + outer[2] + 4
            and inner[1] + inner[3] <= outer[1] + outer[3] + 4)


def _warp(img: np.ndarray, approx: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = _order(approx)
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    src = np.array([tl, tr, br, bl], dtype="float32")
    if width > height:
        # Card lies sideways: rotate so the title ends up at the top. Which way is
        # ambiguous; the model copes with upside-down far better than sideways.
        src = np.array([bl, tl, tr, br], dtype="float32")
    dst = np.array([[0, 0], [OUT_W - 1, 0], [OUT_W - 1, OUT_H - 1], [0, OUT_H - 1]], dtype="float32")
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (OUT_W, OUT_H))
