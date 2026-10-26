"""Synthetic photos of cascading columns for tests: sleeved cards overlapped so only the title
bars of the cards behind show."""

import cv2
import numpy as np


def card(rng, w=250, h=349) -> np.ndarray:
    c = np.empty((h, w, 3), np.uint8)
    c[:] = (60, 120, 40)  # green sleeve
    cv2.rectangle(c, (0, 0), (w - 1, h - 1), (20, 50, 15), 3)  # sleeve edge
    cv2.rectangle(c, (7, 7), (w - 8, h - 8), (25, 25, 25), -1)  # card border
    frame = tuple(int(v) for v in rng.integers(90, 200, 3))
    cv2.rectangle(c, (14, 14), (w - 15, h - 15), frame, -1)  # frame colour
    cv2.rectangle(c, (20, 20), (w - 21, 44), (215, 225, 230), -1)  # title bar
    cv2.rectangle(c, (20, 20), (w - 21, 44), (40, 40, 40), 1)
    for k in range(int(rng.integers(5, 11))):  # title "letters"
        x = 26 + k * 13
        cv2.rectangle(c, (x, 27), (x + int(rng.integers(5, 10)), 38), (30, 30, 30), -1)
    art = rng.integers(30, 220, (int(h * 0.38), w - 44, 3)).astype(np.uint8)
    c[52:52 + art.shape[0], 22:22 + art.shape[1]] = cv2.GaussianBlur(art, (9, 9), 0)
    cv2.rectangle(c, (22, int(h * 0.62)), (w - 23, h - 30), (200, 215, 225), -1)  # text box
    return c


def odd_card(w=250, h=349) -> np.ndarray:
    """A card unlike the ones in the column: unsleeved, an old brown frame (a basic land laid
    down last, say)."""
    c = np.empty((h, w, 3), np.uint8)
    c[:] = (40, 70, 110)
    cv2.rectangle(c, (0, 0), (w - 1, h - 1), (230, 235, 235), 9)  # white border
    cv2.rectangle(c, (24, 30), (w - 25, h * 2 // 3), (150, 110, 120), -1)  # art
    return c


def column(n: int, step: int = 38, drift: int = -4, seed: int = 3, tilt: bool = False,
           front_shift: int = 0, odd_front: bool = False, beside: int = 0) -> tuple[np.ndarray, int]:
    """A photo with one column of n cards (the front one whole, front_shift to the side of where
    the column puts it; odd_front: unlike the others), and optionally a second column of `beside`
    cards to its left. Returns the image and n."""
    rng = np.random.default_rng(seed)
    img = np.empty((1500, 1200, 3), np.uint8)
    img[:] = (120, 70, 40)  # blue mat
    img[:] = np.clip(img.astype(int) + rng.integers(-10, 10, img.shape), 0, 255).astype(np.uint8)
    w, h = 250, 349
    for fx, count in ((480, n), (120, beside)):
        fy = 1500 - h - 120
        for k in reversed(range(count)):  # back card first
            x, y = fx + k * drift, fy - k * step
            if k == 0 and fx == 480:
                x += front_shift
            img[y:y + h, x:x + w] = odd_card(w, h) if k == 0 and fx == 480 and odd_front else card(rng, w, h)
    if tilt:
        H, W = img.shape[:2]
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]),
                                        np.float32([[90, 40], [W - 30, 0], [W, H], [0, H - 50]]))
        img = cv2.warpPerspective(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    return img, n
