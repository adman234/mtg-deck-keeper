"""Locate each entry of the whole-photo read and re-read it from a close-up.

The whole-photo read sees every pile at a few hundred pixels, which is too small to count
stacked card edges or to read some titles. The model is asked for a box per entry; each box
is cut out of the full-resolution photo and read again on its own. The close-ups also serve
as review thumbnails and show where in the photo each line came from.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import fans, matching, progress, vision

log = logging.getLogger(__name__)

PAD_TOP = 0.15  # models often start the box a little below the title bar
PAD_TOP_SNAPPED = 0.05  # the box was snapped to a card OpenCV found, so it is where the card is
PAD_TOP_LOOSE = 0.45  # a single card's box can sit a third of a card too low (scan 12)
TRIM_BOTTOM_LOOSE = 0.3  # ...so leave out its bottom, where the next row's title would be
PAD_TOP_RETRY = 0.8  # second try when the first close-up had no title in it
PAD_TOP_CROWDED = (0.1, 0.3)  # first and second try where cards overlap in rows (cascades, scan 17):
# reaching up as far as above reads the title of the card behind instead
PAD_SIDE = 0.04  # keep the neighbouring pile out of the close-up
PAD_SIDE_PILE = 0.25  # but for a pile, reach far enough sideways to see the cards fanned behind
PAD_BOTTOM = 0.05
OVERLAP_GUARD = 0.3
MIN_SIDE = 0.02  # boxes thinner than this (fraction of the photo) are noise
DUPLICATE_IOU = 0.6
ZOOM_MAX_SIDE = 1024
SIZE_SHORT = 0.85  # a pile's box shorter or narrower than this much of a card is grown to one
PILE_GROW = 1.1  # (a little more than a card tall: the cards behind stick out)
SIZE_OFF = 0.3  # a single card's box this far from a card's size is resized
RUNAWAY = 6  # a run of one name this long at the end of the list may be the model stuck
RUNAWAY_OVERLAP = 0.2
SNAP_IOU = 0.3  # a model box and a card OpenCV found overlapping this much are the same card
SNAP_COVER = 0.7  # or, for a pile, when the found (front) card lies this much inside the box
ROW_X_OVERLAP = 0.5  # in the same row, a box and a found card overlapping this much across are one card
STRONG_MATCH = 80.0  # both reads name a real card this convincingly: ask the model to choose
NEIGHBOUR_MATCH = 75.0
PATTERN_FILL = 3  # one name given to this many single cards in a photo is the model repeating itself
REPEATED_TRUST = 85.0  # a close-up must match this well to replace a repeated whole-table name


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


def _rows(boxes: list[list[int]]) -> list[list[int]]:
    """Group boxes into rows, top to bottom (lists of indices). A box joins the row above when its
    centre is within half a typical box height of that row's centre."""
    if not boxes:
        return []
    half = float(np.median([b[3] for b in boxes])) / 2
    rows: list[list[int]] = []
    centres: list[list[float]] = []
    for i in sorted(range(len(boxes)), key=lambda i: boxes[i][1] + boxes[i][3] / 2):
        cy = boxes[i][1] + boxes[i][3] / 2
        if rows and abs(cy - float(np.mean(centres[-1]))) < half:
            rows[-1].append(i)
            centres[-1].append(cy)
        else:
            rows.append([i])
            centres.append([cy])
    return rows


def _x_overlap(a: list[int], b: list[int]) -> float:
    inter = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    return inter / max(1, min(a[2], b[2]))


def _pair_by_rows(model: list[list[int]], found: list[list[int]],
                  model_rows: list[list[int]]) -> list[tuple[int, int]] | None:
    """Pair the n-th row of model boxes with the n-th row of found cards, then within a row by
    horizontal overlap. The model places cards across a row well but can be far off down the
    photo (scan 13: its three rows were squashed into the middle half), so overlap alone pairs
    boxes with the row above. Returns None when the row counts differ."""
    found_rows = _rows(found)
    if not model_rows or len(model_rows) != len(found_rows):
        return None
    pairs = []
    for mr, fr in zip(model_rows, found_rows):
        used_m, used_f = set(), set()
        for ov, i, j in sorted(((_x_overlap(model[i], found[j]), i, j) for i in mr for j in fr), reverse=True):
            if ov < ROW_X_OVERLAP or i in used_m or j in used_f:
                continue
            used_m.add(i)
            used_f.add(j)
            pairs.append((i, j))
    return pairs


def _pair_by_overlap(cards: list[vision.ReadCard], found: list[list[int]]) -> list[tuple[int, int]]:
    scored = []
    for i, c in enumerate(cards):
        for j, f in enumerate(found):
            iou = _iou(c.extra["bbox"], f)
            if iou >= SNAP_IOU or (c.count >= 2 and _cover(c.extra["bbox"], f) >= SNAP_COVER):
                scored.append((iou, i, j))
    pairs, used_m, used_f = [], set(), set()
    for _, i, j in sorted(scored, reverse=True):
        if i not in used_m and j not in used_f:
            used_m.add(i)
            used_f.add(j)
            pairs.append((i, j))
    return pairs


def snap_boxes(cards: list[vision.ReadCard], found: list[list[int]],
               corners: list[list[list[float]]] | None = None, shift: bool = True) -> dict:
    """Move the model's boxes onto the cards OpenCV found.

    The model's boxes are rough: in scans 12 and 13 whole rows sat a third of a card or more too
    low, so the close-ups missed the titles. A single card's box is replaced by its found card; a
    pile's box is widened to take in its found front card. Single cards with no found card are
    moved by the median correction of the snapped single cards in their row (the model's error
    is much the same along a row), or failing that by an estimate from the other rows. The
    model's own box is kept as extra["model_bbox"]. Returns a summary for the debug view.
    """
    boxed = [c for c in cards if c.extra.get("bbox")]
    model = [c.extra["bbox"] for c in boxed]
    model_rows = _rows(model)
    row_of = {i: r for r, row in enumerate(model_rows) for i in row}
    pairs = _pair_by_rows(model, found, model_rows) if found else []
    matched_by = "rows"
    if pairs is None:
        pairs, matched_by = _pair_by_overlap(boxed, found), "overlap"

    by_row: dict[int, list[np.ndarray]] = {}
    for i, j in pairs:
        c, f = boxed[i], list(found[j])
        c.extra["model_bbox"] = c.extra["bbox"]
        c.extra["snapped"] = True
        if corners:
            c.extra["corners"] = corners[j]
        if c.count >= 2:
            c.extra["bbox"] = _union(c.extra["bbox"], f)
            if corners:
                c.extra["front_corners"] = corners[j]
        else:
            by_row.setdefault(row_of[i], []).append(_edges(f) - _edges(c.extra["model_bbox"]))
            c.extra["bbox"] = f
    row_delta = {r: np.median(ds, axis=0) for r, ds in by_row.items()}
    centre = {r: float(np.mean([model[i][1] + model[i][3] / 2 for i in row])) for r, row in enumerate(model_rows)}

    shifted = 0
    snapped = {i for i, _ in pairs}
    for i, c in enumerate(boxed if shift else []):
        if i in snapped or c.count >= 2:
            continue  # a pile's box covers cards fanned out behind, unlike any found card
        r = row_of[i]
        if r in row_delta:
            d, source = row_delta[r], "row"
        elif len(row_delta) >= 2:
            # No found card in this row: carry the other rows' corrections across, in proportion.
            ys = [centre[k] for k in row_delta]
            d = np.array([np.polyval(np.polyfit(ys, [row_delta[k][n] for k in row_delta], 1), centre[r])
                          for n in range(4)])
            source = "estimate"
        else:
            continue
        x, y, w, h = c.extra["bbox"]
        e = _edges(c.extra["bbox"]) + d
        if e[2] - e[0] < w * 0.5 or e[3] - e[1] < h * 0.5 or np.abs(d).max() > 2 * max(w, h):
            continue  # not the same kind of error; leave it alone
        c.extra["model_bbox"] = c.extra["bbox"]
        c.extra["bbox"] = [round(e[0]), round(e[1]), round(e[2] - e[0]), round(e[3] - e[1])]
        c.extra["shifted"] = [round(v) for v in d]
        c.extra["shift_source"] = source
        shifted += 1
    used = {j for _, j in pairs}
    return {"found": [list(f) for f in found], "matched_by": matched_by, "snapped": len(pairs), "shifted": shifted,
            "missed": [list(found[j]) for j in range(len(found)) if j not in used],
            "missed_index": [j for j in range(len(found)) if j not in used]}


def fit_sizes(cards: list[vision.ReadCard], found: list[list[int]], photo: tuple[int, int]) -> int:
    """Give boxes OpenCV couldn't line up a card's size. Cards in one photo are about the same
    size, but the model often boxes only part of one: the fetchland piles in the lands photo
    were boxed to their top half. A pile's box grows down (its titles are at the top) and
    sideways to at least a card; a single card's box much too big or small is resized around its
    centre. Returns how many boxes changed (they keep the model's box as extra["model_bbox"])."""
    if len(found) < 2:
        return 0
    long = float(np.median([max(f[2], f[3]) for f in found]))
    short = float(np.median([min(f[2], f[3]) for f in found]))
    portrait = sum(f[3] >= f[2] for f in found) * 2 >= len(found)
    cw, ch = (short, long) if portrait else (long, short)
    W, H = photo
    changed = 0
    for c in cards:
        box = c.extra.get("bbox")
        if not box or c.extra.get("snapped"):
            continue
        x, y, w, h = box
        if c.count >= 2:
            if h < SIZE_SHORT * ch:
                h = ch * PILE_GROW
            if w < SIZE_SHORT * cw:
                x, w = x - (cw - w) / 2, cw
        elif abs(w - cw) > SIZE_OFF * cw or abs(h - ch) > SIZE_OFF * ch:
            x, y, w, h = x + w / 2 - cw / 2, y + h / 2 - ch / 2, cw, ch
        else:
            continue
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(W), x + w), min(float(H), y + h)
        new = [round(x1), round(y1), round(x2 - x1), round(y2 - y1)]
        if new != box:
            c.extra.setdefault("model_bbox", box)
            c.extra["bbox"] = new
            c.extra["resized"] = True
            changed += 1
    return changed


def _marching(boxes: list[list[int]]) -> bool:
    """Boxes stepping diagonally by a steady amount, as a stuck model's do. Real copies laid out
    together form a row or a column."""
    if len(boxes) < 4:
        return False
    steps = np.array([[b[0] - a[0], b[1] - a[1]] for a, b in zip(boxes, boxes[1:])], dtype=float)
    w, h = float(np.median([b[2] for b in boxes])), float(np.median([b[3] for b in boxes]))
    diagonal = (np.abs(steps[:, 0]) > 0.1 * w) & (np.abs(steps[:, 1]) > 0.1 * h)
    return bool(diagonal.mean() >= 0.75)


def drop_runaway(cards: list[vision.ReadCard]) -> dict | None:
    """Small models sometimes get stuck repeating the last name, with boxes sliding off the
    photo (a 100-card photo ended in ~70 "Coral Reef"). In a long run of one name at the end of
    the list, drop the entries with no box, those squeezed against the photo's edge, and those
    overlapping one already kept."""
    run = 0
    last = matching.normalize(cards[-1].name) if cards else ""
    for c in reversed(cards):
        if matching.normalize(c.name) != last:
            break
        run += 1
    if run < RUNAWAY:
        return None
    tail = cards[-run:]
    boxed = [c.extra["bbox"] for c in tail if c.extra.get("bbox")]
    if boxed:
        mw, mh = float(np.median([b[2] for b in boxed])), float(np.median([b[3] for b in boxed]))
    keep: list[vision.ReadCard] = []
    for c in tail:
        b = c.extra.get("bbox")
        if not b or b[2] < 0.6 * mw or b[3] < 0.6 * mh:
            continue
        # Separate copies of one card don't overlap; a stuck model's boxes step across the photo.
        if any(_iou(b, k.extra["bbox"]) > RUNAWAY_OVERLAP for k in keep):
            continue
        keep.append(c)
    if _marching([k.extra["bbox"] for k in keep]):
        keep = keep[:1]
    dropped = run - len(keep)
    if not dropped:
        return None
    cards[len(cards) - run:] = keep
    return {"name": tail[0].name, "dropped": dropped}


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
    """(top, bottom) room for a close-up. A single card whose box wasn't lined up with a found card
    (directly or by one in its row) may be boxed too low: reach well up for its title and leave out the bottom, where the title of
    the card below would be."""
    if card.count >= 2:
        return (PAD_TOP_RETRY if retry else PAD_TOP), PAD_BOTTOM
    if card.extra.get("crowded"):
        return PAD_TOP_CROWDED[retry], PAD_BOTTOM
    if card.extra.get("snapped"):
        return (PAD_TOP_RETRY if retry else PAD_TOP_SNAPPED), PAD_BOTTOM
    if card.extra.get("shift_source") == "row":  # moved in line with a found card beside it
        return (PAD_TOP_RETRY if retry else PAD_TOP), PAD_BOTTOM
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
        progress.tick()
        if result is not None and not result.name:
            # No title in view: the box most likely starts below it. Try once with more room above.
            progress.add(1)
            retry = await read(i, card, True, "-tall")
            progress.tick()
            if retry is not None:
                retry.retried = True
                result = retry if retry.name else result
        card.extra["zoom"] = result

    boxed = [(i, c) for i, c in enumerate(cards) if c.extra.get("bbox")]
    progress.stage("Close-ups of each card and pile", len(boxed))
    await asyncio.gather(*(one(i, c) for i, c in boxed))


@dataclass
class Decision:
    name: str
    count: int
    used: str  # photo | zoom
    count_differs: bool
    unsure: bool = False  # the close-up couldn't confirm the name
    repeated: bool = False  # the whole-table name was repeated along a row, so it's no fallback
    suggest: list[str] = field(default_factory=list)  # cards to offer first if left to the user


def count_fans(img: np.ndarray, cards: list[vision.ReadCard]) -> None:
    """Count the cards in each pile whose front card OpenCV found, from the slivers fanned out
    behind it (stored as extra["fan"])."""
    for c in cards:
        if c.extra.get("front_corners") and c.extra.get("bbox"):
            c.extra["fan"] = fans.count_pile(img, c.extra["bbox"], c.extra["front_corners"])


def combine_counts(photo: int, zoom: int | None, fan: fans.FanCount | None = None) -> tuple[int, bool]:
    """Pick the pile count from the whole-table read and the close-up. Returns (count, disagree).

    - Slivers counted by OpenCV win: the models can't count stacked card edges reliably (in the
      elves photo they said 5 or 6 for piles of 4). It can only count cards that show an edge,
      so a card hidden exactly behind another is missed; when it counts 2 or more fewer than
      both models, the line is flagged. A count of 1 means it saw no slivers
      (a squared-up pile, or the fan went unseen), so the models' counts are used instead.

    - Close-up says 1, whole table says more: the close-up box held only the front card and
      couldn't see the stack, so keep the whole-table count.
    - Otherwise when they disagree, take the lower: close-ups over-count sleeve edges and the
      edge of the next pile, and the whole-table read leans towards 5.
    Either way the line is flagged so the quantity gets a look.
    """
    if fan is not None and fan.count >= 2:
        # Far fewer than both models say: worth a look (a missed sliver, or a hidden card).
        return fan.count, zoom is not None and fan.count < min(photo, zoom) - 1
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


def _above(card: vision.ReadCard, others: list[vision.ReadCard]) -> list[vision.ReadCard]:
    """Entries boxed just above this one and overlapping it sideways (the card behind, in a row
    of overlapping cards)."""
    box = card.extra.get("bbox")
    if not box:
        return []
    x, y, w, h = box
    out = []
    for o in others:
        ob = o.extra.get("bbox")
        if o is card or not ob:
            continue
        overlap = min(x + w, ob[0] + ob[2]) - max(x, ob[0])
        if overlap > 0.5 * min(w, ob[2]) and y - 1.5 * h <= ob[1] < y:
            out.append(o)
    return out


def _pattern_group(index: matching.CardIndex, card: vision.ReadCard, others: list[vision.ReadCard],
                   hit: matching.Match | None) -> list[vision.ReadCard]:
    """The single cards the whole-table read gave this same name, when there are several: often
    it read one and repeated it along the row (scan 12: four "Misty Rainforest" for a Forest and
    three Bayous; scan 13: four "Meticulous Artisan"). Empty when the name isn't repeated."""
    if card.count >= 2 or hit is None:
        return []
    same = [o for o in others if o.count == 1 and (o.name.casefold() == card.name.casefold()
                                                   or getattr(_best(index, o.name), "card_id", None) == hit.card_id)]
    return same if len(same) >= PATTERN_FILL else []


@dataclass
class Ask:
    """Ask the model which of these cards the close-up shows; until then, go with fallback."""
    options: list[str]
    fallback: Decision


def _ask_or_pick(card: vision.ReadCard, options: list[str], fallback: Decision,
                 photo_hit: matching.Match) -> Decision | Ask:
    picked = card.extra.get("tiebreak")
    if picked is None:
        return Ask(options, fallback)
    name = picked.get("name")
    if name not in options:
        # The model couldn't choose ("none"), so the close-up most likely isn't of this card at
        # all: keep the whole-table name, flagged.
        return fallback
    return Decision(name, fallback.count, "photo" if name == photo_hit.name else "zoom", fallback.count_differs)


def _judge(index: matching.CardIndex, card: vision.ReadCard, others: list[vision.ReadCard]) -> Decision | Ask:
    """Pick between the whole-photo read and the close-up read of one entry, or say which card
    names to ask the model to choose between."""
    zoom: vision.ZoomRead | None = card.extra.get("zoom")
    fan = card.extra.get("fan")
    if zoom is None:
        return Decision(card.name, combine_counts(card.count, None, fan)[0], "photo", False)
    count, differs = combine_counts(card.count, zoom.count, fan)
    if not zoom.name:
        # The close-up couldn't see a title, so the whole-table name is unconfirmed (and is
        # where made-up names come from).
        return Decision(card.name, count, "photo", differs, unsure=True)

    photo_hit, zoom_hit = _best(index, card.name), _best(index, zoom.name)
    photo_score = photo_hit.score if photo_hit else 0.0
    zoom_score = zoom_hit.score if zoom_hit else 0.0
    keep_photo = Decision(photo_hit.name if photo_hit and photo_score >= STRONG_MATCH else card.name,
                          count, "photo", differs, unsure=True)
    if not (zoom_hit and photo_hit and zoom_hit.card_id == photo_hit.card_id):
        # The close-up names a different card. If an overlapping entry is that card, this box was
        # probably drawn over the neighbouring pile, so keep the whole-photo read.
        box = card.extra.get("bbox")
        for o in others:
            if not zoom_hit or o is card or not box or not o.extra.get("bbox") or _iou(box, o.extra["bbox"]) < OVERLAP_GUARD:
                continue
            if any(h.card_id == zoom_hit.card_id for h in index.match(o.name, limit=1)):
                return keep_photo
        group = _pattern_group(index, card, others, photo_hit)
        if group and zoom_hit and any(getattr(_best(index, o.name), "card_id", None) == zoom_hit.card_id
                                      for o in _above(card, others)):
            # The close-up reached up to the title of the card behind (scan 17: a Snow-Covered
            # Plains read as the Darksteel Plate above it, giving two Plates). A repeated name is
            # no reason to believe it over the whole-photo read.
            return keep_photo
        if group:
            if zoom_score >= REPEATED_TRUST:
                return Decision(zoom.name, count, "zoom", differs, repeated=True)
            # The repeated name is likely wrong and the close-up is no help either: leave it to
            # the user, offering the cards the row's other close-ups read (often the right one).
            suggest = []
            for o in group:
                hit = _best(index, getattr(o.extra.get("zoom"), "name", ""))
                if o is not card and hit and hit.score >= REPEATED_TRUST and hit.name not in suggest:
                    suggest.append(hit.name)
            return Decision(zoom.name, count, "zoom", differs, unsure=True, repeated=True, suggest=suggest)
        if zoom_hit and photo_score >= NEIGHBOUR_MATCH:
            # The close-up read the title of the card next to it (its box sat over the gap).
            for o in _neighbours(card, others):
                if getattr(_best(index, o.name), "card_id", None) == zoom_hit.card_id:
                    return keep_photo
        if photo_hit and photo_score >= STRONG_MATCH and zoom_score >= STRONG_MATCH:
            return _ask_or_pick(card, [photo_hit.name, zoom_hit.name], keep_photo, photo_hit)
    if zoom_score >= photo_score - 5:
        return Decision(zoom.name, count, "zoom", differs)
    return Decision(card.name, count, "photo", differs)


def decide(index: matching.CardIndex, card: vision.ReadCard, others: list[vision.ReadCard]) -> Decision:
    """Pick between the whole-photo read and the close-up read of one entry."""
    result = _judge(index, card, others)
    return result.fallback if isinstance(result, Ask) else result


async def settle(img: np.ndarray, cards: list[vision.ReadCard], index: matching.CardIndex,
                 cfg: vision.VisionConfig, sem: asyncio.Semaphore) -> None:
    """Where the whole-table read and the close-up name different real cards, or a repeated
    whole-table name has an unreadable close-up, show the model a taller close-up and ask which
    card it is (stored as extra["tiebreak"])."""
    async def one(card: vision.ReadCard, options: list[str]) -> None:
        top, bottom = pads(card, retry=True)
        ok, buf = cv2.imencode(".jpg", _crop(img, card.extra["bbox"], top, card.count >= 2, bottom),
                               [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return
        async with sem:
            picked = await vision.choose_card(cfg, buf.tobytes(), options)
        progress.tick()
        card.extra["tiebreak"] = {"options": options, "name": picked[0] if picked else None,
                                  "confidence": picked[1] if picked else None}

    asks = []
    for card in cards:
        if card.extra.get("bbox") and isinstance(ask := _judge(index, card, cards), Ask):
            asks.append(one(card, ask.options))
    if asks:
        progress.stage("Settling disagreements between reads", len(asks))
    await asyncio.gather(*asks)
