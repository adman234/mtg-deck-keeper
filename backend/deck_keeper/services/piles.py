"""Locate each entry of the whole-photo read and re-read it from a close-up.

The whole-photo read sees every pile at a few hundred pixels, which is too small to count
stacked card edges or to read some titles. The model is asked for a box per entry; each box
is cut out of the full-resolution photo and read again on its own. The close-ups also serve
as review thumbnails and show where in the photo each line came from.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import matching, vision

log = logging.getLogger(__name__)

PAD_TOP = 0.15  # models often start the box a little below the title bar
PAD_TOP_RETRY = 0.6  # second try when the first close-up had no title in it
PAD_SIDE = 0.04  # keep the neighbouring pile out of the close-up
PAD_SIDE_PILE = 0.25  # but for a pile, reach far enough sideways to see the cards fanned behind
PAD_BOTTOM = 0.05
OVERLAP_GUARD = 0.3
MIN_SIDE = 0.02  # boxes thinner than this (fraction of the photo) are noise
DUPLICATE_IOU = 0.6
ZOOM_MAX_SIDE = 1024


def _iou(a: list[float], b: list[float]) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def place_boxes(cards: list[vision.ReadCard], sent: tuple[int, int], photo: tuple[int, int]) -> str | None:
    """Convert model boxes to [x, y, w, h] in photo pixels, stored as extra["bbox"].

    Boxes are requested on a 0-1000 scale, but some models (Qwen2.5-VL) answer in pixels of the
    image they were sent. If any coordinate in the reply is over 1000, the reply uses pixels.
    Returns the scale used, or None when the reply had no usable boxes.
    """
    boxes = [c.extra.get("box") for c in cards if c.extra.get("box")]
    if not boxes:
        return None
    pixels = any(v > 1000 for b in boxes for v in b)
    sx, sy = (photo[0] / sent[0], photo[1] / sent[1]) if pixels else (photo[0] / 1000, photo[1] / 1000)
    for c in cards:
        b = c.extra.get("box")
        if not b:
            continue
        x1, y1 = max(0.0, b[0] * sx), max(0.0, b[1] * sy)
        x2, y2 = min(float(photo[0]), b[2] * sx), min(float(photo[1]), b[3] * sy)
        if x2 - x1 >= photo[0] * MIN_SIDE and y2 - y1 >= photo[1] * MIN_SIDE:
            c.extra["bbox"] = [round(x1), round(y1), round(x2 - x1), round(y2 - y1)]
    return "pixels" if pixels else "0-1000"


def drop_duplicates(cards: list[vision.ReadCard]) -> tuple[list[vision.ReadCard], list[vision.ReadCard]]:
    """Two entries boxed on the same spot are the same pile read twice; keep the surer one."""
    kept: list[vision.ReadCard] = []
    dropped: list[vision.ReadCard] = []
    for c in sorted(cards, key=lambda c: -c.confidence):
        box = c.extra.get("bbox")
        if box and any(k.extra.get("bbox") and _iou(box, k.extra["bbox"]) > DUPLICATE_IOU for k in kept):
            dropped.append(c)
        else:
            kept.append(c)
    order = {id(c): i for i, c in enumerate(cards)}
    kept.sort(key=lambda c: order[id(c)])
    return kept, dropped


def crop_region(img_size: tuple[int, int], bbox: list[int], top: float = PAD_TOP, pile: bool = False) -> list[int]:
    """The area a close-up covers, [x, y, w, h]: extra room above (models often start the box
    below the title bar). Little at the sides for a single card (where the neighbouring pile is),
    more for a pile, because models often box only its front card and miss the cards fanned out
    behind it."""
    w, h = img_size
    x, y, bw, bh = bbox
    side = PAD_SIDE_PILE if pile else PAD_SIDE
    x1, x2 = int(max(0, x - bw * side)), int(min(w, x + bw * (1 + side)))
    y1, y2 = int(max(0, y - bh * top)), int(min(h, y + bh * (1 + PAD_BOTTOM)))
    return [x1, y1, x2 - x1, y2 - y1]


def _crop(img: np.ndarray, bbox: list[int], top: float = PAD_TOP, pile: bool = False) -> np.ndarray:
    x, y, w, h = crop_region((img.shape[1], img.shape[0]), bbox, top, pile)
    crop = img[y:y + h, x:x + w]
    scale = min(1.0, ZOOM_MAX_SIDE / max(crop.shape[:2]))
    if scale < 1:
        crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return crop


async def zoom_in(img: np.ndarray, cards: list[vision.ReadCard], crop_dir: Path, stem: str,
                  cfg: vision.VisionConfig, sem: asyncio.Semaphore) -> None:
    """Save a close-up of every boxed entry and re-read it (stored as extra["zoom"])."""
    crop_dir.mkdir(exist_ok=True)

    async def read(i: int, card: vision.ReadCard, top: float, suffix: str) -> vision.ZoomRead | None:
        pile = card.count >= 2
        ok, buf = cv2.imencode(".jpg", _crop(img, card.extra["bbox"], top, pile), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return None
        path = crop_dir / f"{stem}-pile-{i:03d}{suffix}.jpg"
        path.write_bytes(buf.tobytes())
        async with sem:
            result = await vision.read_pile(cfg, buf.tobytes())
        if result is not None and (result.name or "crop_path" not in card.extra):
            card.extra["crop_path"] = str(path)
            card.extra["zoom_region"] = crop_region((img.shape[1], img.shape[0]), card.extra["bbox"], top, pile)
        return result

    async def one(i: int, card: vision.ReadCard) -> None:
        result = await read(i, card, PAD_TOP, "")
        if result is not None and not result.name:
            # No title in view: the box most likely starts below it. Try once with more room above.
            retry = await read(i, card, PAD_TOP_RETRY, "-tall")
            if retry is not None:
                retry.retried = True
                result = retry if retry.name else result
        card.extra["zoom"] = result

    await asyncio.gather(*(one(i, c) for i, c in enumerate(cards) if c.extra.get("bbox")))


@dataclass
class Decision:
    name: str
    count: int
    used: str  # photo | zoom
    count_differs: bool
    unsure: bool = False  # the close-up couldn't confirm the name


def combine_counts(photo: int, zoom: int | None) -> tuple[int, bool]:
    """Pick the pile count from the whole-table read and the close-up. Returns (count, disagree).

    - Close-up says 1, whole table says more: the close-up box held only the front card and
      couldn't see the stack, so keep the whole-table count.
    - Otherwise when they disagree, take the lower: close-ups over-count sleeve edges and the
      edge of the next pile, and the whole-table read leans towards 5.
    Either way the line is flagged so the quantity gets a look.
    """
    if zoom is None or zoom == photo:
        return photo, False
    if zoom == 1 and photo >= 2:
        return photo, True
    return min(zoom, photo), True


def decide(index: matching.CardIndex, card: vision.ReadCard, others: list[vision.ReadCard]) -> Decision:
    """Pick between the whole-photo read and the close-up read of one entry."""
    zoom: vision.ZoomRead | None = card.extra.get("zoom")
    if zoom is None:
        return Decision(card.name, card.count, "photo", False)
    count, differs = combine_counts(card.count, zoom.count)
    if not zoom.name:
        # The close-up couldn't see a title, so the whole-table name is unconfirmed (and is
        # where made-up names come from).
        return Decision(card.name, count, "photo", differs, unsure=True)

    def best(name: str) -> matching.Match | None:
        hits = index.match(name, limit=1)
        return hits[0] if hits else None

    photo_hit, zoom_hit = best(card.name), best(zoom.name)
    photo_score = photo_hit.score if photo_hit else 0.0
    zoom_score = zoom_hit.score if zoom_hit else 0.0
    if zoom_hit and (not photo_hit or zoom_hit.card_id != photo_hit.card_id):
        # The close-up names a different card. If an overlapping entry is that card, this box was
        # probably drawn over the neighbouring pile, so keep the whole-photo read.
        box = card.extra.get("bbox")
        for o in others:
            if o is card or not box or not o.extra.get("bbox") or _iou(box, o.extra["bbox"]) < OVERLAP_GUARD:
                continue
            if any(h.card_id == zoom_hit.card_id for h in index.match(o.name, limit=1)):
                return Decision(card.name, count, "photo", True)
    if zoom_score >= photo_score - 5:
        return Decision(zoom.name, count, "zoom", differs)
    return Decision(card.name, count, "photo", differs)
