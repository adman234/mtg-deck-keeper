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

PAD = 0.12  # grow each box by this much per side: models draw tight or slightly off boxes
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


def _crop(img: np.ndarray, bbox: list[int]) -> np.ndarray:
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    px, py = bw * PAD, bh * PAD
    x1, y1 = int(max(0, x - px)), int(max(0, y - py))
    x2, y2 = int(min(w, x + bw + px)), int(min(h, y + bh + py))
    crop = img[y1:y2, x1:x2]
    scale = min(1.0, ZOOM_MAX_SIDE / max(crop.shape[:2]))
    if scale < 1:
        crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return crop


async def zoom_in(img: np.ndarray, cards: list[vision.ReadCard], crop_dir: Path, stem: str,
                  cfg: vision.VisionConfig, sem: asyncio.Semaphore) -> None:
    """Save a close-up of every boxed entry and re-read it (stored as extra["zoom"])."""
    crop_dir.mkdir(exist_ok=True)

    async def one(i: int, card: vision.ReadCard) -> None:
        crop = _crop(img, card.extra["bbox"])
        path = crop_dir / f"{stem}-pile-{i:03d}.jpg"
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return
        path.write_bytes(buf.tobytes())
        card.extra["crop_path"] = str(path)
        async with sem:
            card.extra["zoom"] = await vision.read_pile(cfg, buf.tobytes(), hint=card.name)

    await asyncio.gather(*(one(i, c) for i, c in enumerate(cards) if c.extra.get("bbox")))


@dataclass
class Decision:
    name: str
    count: int
    used: str  # photo | zoom
    count_differs: bool


def decide(index: matching.CardIndex, card: vision.ReadCard, others: list[vision.ReadCard]) -> Decision:
    """Pick between the whole-photo read and the close-up read of one entry."""
    zoom: vision.ZoomRead | None = card.extra.get("zoom")
    if zoom is None:
        return Decision(card.name, card.count, "photo", False)
    count = zoom.count or card.count
    differs = zoom.count is not None and zoom.count != card.count
    if not zoom.name:
        return Decision(card.name, count, "photo", differs)

    def best(name: str) -> matching.Match | None:
        hits = index.match(name, limit=1)
        return hits[0] if hits else None

    photo_hit, zoom_hit = best(card.name), best(zoom.name)
    photo_score = photo_hit.score if photo_hit else 0.0
    zoom_score = zoom_hit.score if zoom_hit else 0.0
    if zoom_hit and (not photo_hit or zoom_hit.card_id != photo_hit.card_id):
        # The close-up names a different card. If that card is another entry in this photo,
        # the box was probably drawn around the wrong pile, so trust the whole-photo read.
        other_ids = {h.card_id for o in others if o is not card for h in index.match(o.name, limit=1)}
        if zoom_hit.card_id in other_ids:
            return Decision(card.name, card.count, "photo", True)
    if zoom_score >= photo_score - 5:
        return Decision(zoom.name, count, "zoom", differs)
    return Decision(card.name, count, "photo", differs)
