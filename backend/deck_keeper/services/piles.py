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
PAD_TOP_SNAPPED = 0.05  # the box was snapped to a card OpenCV found, so it is where the card is
PAD_TOP_LOOSE = 0.45  # a single card's box can sit a third of a card too low (scan 12)
TRIM_BOTTOM_LOOSE = 0.3  # ...so leave out its bottom, where the next row's title would be
PAD_TOP_RETRY = 0.8  # second try when the first close-up had no title in it
PAD_SIDE = 0.04  # keep the neighbouring pile out of the close-up
PAD_SIDE_PILE = 0.25  # but for a pile, reach far enough sideways to see the cards fanned behind
PAD_BOTTOM = 0.05
OVERLAP_GUARD = 0.3
MIN_SIDE = 0.02  # boxes thinner than this (fraction of the photo) are noise
DUPLICATE_IOU = 0.6
ZOOM_MAX_SIDE = 1024
SNAP_IOU = 0.3  # a model box and a card OpenCV found overlapping this much are the same card
SNAP_COVER = 0.7  # or, for a pile, when the found (front) card lies this much inside the box
STRONG_MATCH = 80.0  # both reads name a real card this convincingly: ask the model to choose
NEIGHBOUR_MATCH = 75.0
PATTERN_FILL = 3  # one name given to this many single cards in a photo is the model repeating itself


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


def _cover(outer: list[float], inner: list[float]) -> float:
    """Fraction of inner's area that lies inside outer."""
    iw = max(0.0, min(outer[0] + outer[2], inner[0] + inner[2]) - max(outer[0], inner[0]))
    ih = max(0.0, min(outer[1] + outer[3], inner[1] + inner[3]) - max(outer[1], inner[1]))
    return iw * ih / (inner[2] * inner[3]) if inner[2] * inner[3] > 0 else 0.0


def _union(a: list[int], b: list[int]) -> list[int]:
    x1, y1 = min(a[0], b[0]), min(a[1], b[1])
    x2, y2 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return [x1, y1, x2 - x1, y2 - y1]


def _edges(b: list[int]) -> np.ndarray:
    return np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]], dtype=float)


def snap_boxes(cards: list[vision.ReadCard], found: list[list[int]]) -> dict:
    """Move the model's boxes onto the cards OpenCV found.

    The model's boxes are rough: in scan 12 whole rows sat a third of a card too low, so the
    close-ups missed the titles. A single card's box is replaced by the found card; a pile's box
    is widened to take in its found front card. Single cards with no found card are shifted by
    the median correction of the snapped single cards (those in the same row if any), since the
    model's error is much the same for neighbours. The model's own box is kept as
    extra["model_bbox"]. Returns a summary for the debug view.
    """
    boxed = [c for c in cards if c.extra.get("bbox")]
    pairs = []
    for i, c in enumerate(boxed):
        for j, f in enumerate(found):
            iou = _iou(c.extra["bbox"], f)
            if iou >= SNAP_IOU or (c.count >= 2 and _cover(c.extra["bbox"], f) >= SNAP_COVER):
                pairs.append((iou, i, j))
    used_cards, used_found = set(), set()
    deltas: list[tuple[float, np.ndarray]] = []  # (centre y, edge corrections) of snapped singles
    for _, i, j in sorted(pairs, reverse=True):
        if i in used_cards or j in used_found:
            continue
        used_cards.add(i)
        used_found.add(j)
        c, f = boxed[i], found[j]
        c.extra["model_bbox"] = c.extra["bbox"]
        c.extra["snapped"] = True
        if c.count >= 2:
            c.extra["bbox"] = _union(c.extra["bbox"], list(f))
        else:
            deltas.append((f[1] + f[3] / 2, _edges(f) - _edges(c.extra["model_bbox"])))
            c.extra["bbox"] = list(f)
    shifted = 0
    if len(deltas) >= 2:
        for i, c in enumerate(boxed):
            if i in used_cards or c.count >= 2:
                continue  # a pile's box covers cards fanned out behind, unlike any found card
            x, y, w, h = c.extra["bbox"]
            row = [d for cy, d in deltas if abs(cy - (y + h / 2)) < h / 2]
            d = np.median(row or [d for _, d in deltas], axis=0)
            if abs(d[1]) > h * 0.6 or abs(d[0]) > w * 0.6:
                continue  # too big to be the same kind of error; leave it alone
            e = _edges(c.extra["bbox"]) + d
            if e[2] - e[0] < w * 0.5 or e[3] - e[1] < h * 0.5:
                continue
            c.extra["model_bbox"] = c.extra["bbox"]
            c.extra["bbox"] = [round(e[0]), round(e[1]), round(e[2] - e[0]), round(e[3] - e[1])]
            c.extra["shifted"] = [round(v) for v in d]
            shifted += 1
    return {"found": [list(f) for f in found], "snapped": len(used_cards), "shifted": shifted,
            "missed": [list(found[j]) for j in range(len(found)) if j not in used_found]}


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


def crop_region(img_size: tuple[int, int], bbox: list[int], top: float = PAD_TOP, pile: bool = False,
                bottom: float = PAD_BOTTOM) -> list[int]:
    """The area a close-up covers, [x, y, w, h]: extra room above (models often start the box
    below the title bar). Little at the sides for a single card (where the neighbouring pile is),
    more for a pile, because models often box only its front card and miss the cards fanned out
    behind it. A negative bottom trims the box."""
    w, h = img_size
    x, y, bw, bh = bbox
    side = PAD_SIDE_PILE if pile else PAD_SIDE
    x1, x2 = int(max(0, x - bw * side)), int(min(w, x + bw * (1 + side)))
    y1, y2 = int(max(0, y - bh * top)), int(min(h, y + bh * (1 + bottom)))
    return [x1, y1, x2 - x1, y2 - y1]


def pads(card: vision.ReadCard, retry: bool = False) -> tuple[float, float]:
    """(top, bottom) room for a close-up. A single card whose box wasn't snapped to a found card
    may be boxed too low: reach well up for its title and leave out the bottom, where the title of
    the card below would be."""
    if card.count >= 2:
        return (PAD_TOP_RETRY if retry else PAD_TOP), PAD_BOTTOM
    if card.extra.get("snapped"):
        return (PAD_TOP_RETRY if retry else PAD_TOP_SNAPPED), PAD_BOTTOM
    return (PAD_TOP_RETRY if retry else PAD_TOP_LOOSE), -TRIM_BOTTOM_LOOSE


def _crop(img: np.ndarray, bbox: list[int], top: float = PAD_TOP, pile: bool = False,
          bottom: float = PAD_BOTTOM) -> np.ndarray:
    x, y, w, h = crop_region((img.shape[1], img.shape[0]), bbox, top, pile, bottom)
    crop = img[y:y + h, x:x + w]
    scale = min(1.0, ZOOM_MAX_SIDE / max(crop.shape[:2]))
    if scale < 1:
        crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return crop


async def zoom_in(img: np.ndarray, cards: list[vision.ReadCard], crop_dir: Path, stem: str,
                  cfg: vision.VisionConfig, sem: asyncio.Semaphore) -> None:
    """Save a close-up of every boxed entry and re-read it (stored as extra["zoom"])."""
    crop_dir.mkdir(exist_ok=True)

    async def read(i: int, card: vision.ReadCard, retry: bool, suffix: str) -> vision.ZoomRead | None:
        pile = card.count >= 2
        top, bottom = pads(card, retry)
        ok, buf = cv2.imencode(".jpg", _crop(img, card.extra["bbox"], top, pile, bottom), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return None
        path = crop_dir / f"{stem}-pile-{i:03d}{suffix}.jpg"
        path.write_bytes(buf.tobytes())
        async with sem:
            result = await vision.read_pile(cfg, buf.tobytes())
        if result is not None and (result.name or "crop_path" not in card.extra):
            card.extra["crop_path"] = str(path)
            card.extra["zoom_region"] = crop_region((img.shape[1], img.shape[0]), card.extra["bbox"], top, pile, bottom)
        return result

    async def one(i: int, card: vision.ReadCard) -> None:
        result = await read(i, card, False, "")
        if result is not None and not result.name:
            # No title in view: the box most likely starts below it. Try once with more room above.
            retry = await read(i, card, True, "-tall")
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


def _best(index: matching.CardIndex, name: str) -> matching.Match | None:
    hits = index.match(name, limit=1) if name else []
    return hits[0] if hits else None


def _neighbours(card: vision.ReadCard, others: list[vision.ReadCard]) -> list[vision.ReadCard]:
    """Entries boxed right next to this one (half a box away or less, in any direction)."""
    box = card.extra.get("bbox")
    if not box:
        return []
    x, y, w, h = box
    near = [x - w / 2, y - h / 2, w * 2, h * 2]
    return [o for o in others if o is not card and o.extra.get("bbox") and _cover(near, o.extra["bbox"]) > 0]


def _pattern_fill(index: matching.CardIndex, card: vision.ReadCard, others: list[vision.ReadCard],
                  hit: matching.Match | None) -> bool:
    """The whole-table read gave this name to several single cards: often it read one and
    repeated it along the row (scan 12: four "Misty Rainforest" for a Forest and three Bayous)."""
    if card.count >= 2 or hit is None:
        return False
    same = [o for o in others if o.count == 1 and (o.name.casefold() == card.name.casefold()
                                                   or getattr(_best(index, o.name), "card_id", None) == hit.card_id)]
    return len(same) >= PATTERN_FILL


def _judge(index: matching.CardIndex, card: vision.ReadCard, others: list[vision.ReadCard]) -> Decision | list[str]:
    """Pick between the whole-photo read and the close-up read of one entry, or return the two
    card names to ask the model to choose between."""
    zoom: vision.ZoomRead | None = card.extra.get("zoom")
    if zoom is None:
        return Decision(card.name, card.count, "photo", False)
    count, differs = combine_counts(card.count, zoom.count)
    if not zoom.name:
        # The close-up couldn't see a title, so the whole-table name is unconfirmed (and is
        # where made-up names come from).
        return Decision(card.name, count, "photo", differs, unsure=True)

    photo_hit, zoom_hit = _best(index, card.name), _best(index, zoom.name)
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
                return Decision(card.name, count, "photo", differs, unsure=True)
        if _pattern_fill(index, card, others, photo_hit):
            if zoom_score >= STRONG_MATCH:
                return Decision(zoom.name, count, "zoom", differs)
            return Decision(card.name, count, "photo", differs, unsure=True)
        if photo_score >= NEIGHBOUR_MATCH:
            # The close-up read the title of the card next to it (its box sat over the gap).
            for o in _neighbours(card, others):
                if getattr(_best(index, o.name), "card_id", None) == zoom_hit.card_id:
                    return Decision(card.name, count, "photo", differs, unsure=True)
        if photo_hit and photo_score >= STRONG_MATCH and zoom_score >= STRONG_MATCH:
            picked = card.extra.get("tiebreak")
            if picked is None:
                return [photo_hit.name, zoom_hit.name]
            if picked.get("name") == photo_hit.name:
                return Decision(photo_hit.name, count, "photo", differs)
            if picked.get("name") == zoom_hit.name:
                return Decision(zoom_hit.name, count, "zoom", differs)
            # The model couldn't choose: fall through to the usual rule, flagged.
            if zoom_score >= photo_score - 5:
                return Decision(zoom.name, count, "zoom", differs, unsure=True)
            return Decision(card.name, count, "photo", differs, unsure=True)
    if zoom_score >= photo_score - 5:
        return Decision(zoom.name, count, "zoom", differs)
    return Decision(card.name, count, "photo", differs)


def decide(index: matching.CardIndex, card: vision.ReadCard, others: list[vision.ReadCard]) -> Decision:
    """Pick between the whole-photo read and the close-up read of one entry."""
    result = _judge(index, card, others)
    if isinstance(result, Decision):
        return result
    # Two real cards and nobody asked the model to choose (or it couldn't be reached): the
    # close-up usually sees the title better, but say so.
    zoom = card.extra["zoom"]
    count, differs = combine_counts(card.count, zoom.count)
    return Decision(zoom.name, count, "zoom", differs, unsure=True)


async def settle(img: np.ndarray, cards: list[vision.ReadCard], index: matching.CardIndex,
                 cfg: vision.VisionConfig, sem: asyncio.Semaphore) -> None:
    """Where the whole-table read and the close-up name two different real cards, show the model
    a taller close-up and ask which of the two it is (stored as extra["tiebreak"])."""
    async def one(card: vision.ReadCard, options: list[str]) -> None:
        top, bottom = pads(card, retry=True)
        ok, buf = cv2.imencode(".jpg", _crop(img, card.extra["bbox"], top, card.count >= 2, bottom),
                               [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return
        async with sem:
            picked = await vision.choose_card(cfg, buf.tobytes(), options)
        card.extra["tiebreak"] = {"options": options, "name": picked[0] if picked else None,
                                  "confidence": picked[1] if picked else None}

    asks = []
    for card in cards:
        if card.extra.get("bbox") and isinstance(options := _judge(index, card, cards), list):
            asks.append(one(card, options))
    await asyncio.gather(*asks)
