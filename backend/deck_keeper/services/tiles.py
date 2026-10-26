"""Read a photo of many cards in overlapping tiles.

A whole photo of a 100-card deck is sent to the model at about 1600 px, which leaves each title
bar some 15 px tall: too small for a small model, which then lists made-up names or gets stuck
repeating one. Read in tiles from the full-resolution photo, each title gets several times the
pixels and each reply lists far fewer cards. Cards seen in two tiles (where they overlap) are
kept once: the copy seen furthest from a tile's inner edge, where it was seen whole.
"""

import asyncio
import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

from . import piles, progress, vision

log = logging.getLogger(__name__)

TILE_ABOVE = 30  # a whole-photo read listing this many entries is redone in tiles
CARDS_PER_TILE = 18
OVERLAP = 0.18  # of a tile's size, so a card on a seam is whole in one of them
MIN_SIDE = 1024  # tiles are scaled up to at least this (the model resizes small images anyway)
SAME_CARD_IOU = 0.3
SAME_NAME_IOU = 0.15
SEAM = 0.02  # an entry boxed this close to a tile's inner edge (of the tile's size) was cut off by it
SMALL_CARDS = 9  # cards narrower than 1/9 of the photo: tile even if the read looked fine
CASCADE_TILES = 15  # this many title bars found in cascading columns: skip the whole-photo read
TILE_CARDS = (3.2, 2.4)  # a tile is about this many card widths and heights
MAX_TILES = 30


@dataclass
class Tile:
    x: int
    y: int
    w: int
    h: int


def reason(entries: int, cut_off: bool, runaway: dict | None, card_width: float | None, photo_w: int) -> str | None:
    """Why a photo should be read in tiles, or None."""
    if runaway:
        return f"the model got stuck repeating {runaway['name']!r}"
    if cut_off:
        return "the model's list was cut off"
    if entries >= TILE_ABOVE:
        return f"{entries} cards listed"
    if card_width and card_width * SMALL_CARDS < photo_w:
        return "the cards are small in the photo"
    return None


def plan(w: int, h: int, cards: int | None, card: tuple[float | None, float | None] = (None, None)) -> list[Tile]:
    """A grid of overlapping tiles, at least 2 x 2, shaped like the photo. Sized from the cards
    when their size is known (each tile about TILE_CARDS card widths by heights, so a tile of
    cascading columns holds a few columns of a dozen titles), else from how many were listed."""
    cw, ch = card
    if cw and ch:
        cols = max(2, math.ceil(w / (TILE_CARDS[0] * cw)))
        rows = max(2, math.ceil(h / (TILE_CARDS[1] * ch)))
        while cols * rows > MAX_TILES and max(cols, rows) > 2:
            if cols >= rows:
                cols -= 1
            else:
                rows -= 1
    else:
        n = max(4, math.ceil((cards or 0) / CARDS_PER_TILE))
        cols = max(2, round(math.sqrt(n * w / h)))
        rows = max(2, math.ceil(n / cols))
    tw, th = w / (cols - (cols - 1) * OVERLAP), h / (rows - (rows - 1) * OVERLAP)
    out = []
    for r in range(rows):
        for c in range(cols):
            x, y = round(c * tw * (1 - OVERLAP)), round(r * th * (1 - OVERLAP))
            out.append(Tile(x, y, min(round(tw), w - x), min(round(th), h - y)))
    return out


def _edge_margin(box: list[int], tile: Tile, photo: tuple[int, int]) -> float:
    """How far inside the tile a box sits, counting only edges shared with another tile."""
    x, y, w, h = box
    gaps = []
    if tile.x > 0:
        gaps.append(x - tile.x)
    if tile.y > 0:
        gaps.append(y - tile.y)
    if tile.x + tile.w < photo[0]:
        gaps.append(tile.x + tile.w - (x + w))
    if tile.y + tile.h < photo[1]:
        gaps.append(tile.y + tile.h - (y + h))
    return min(gaps) if gaps else float("inf")


def _contains(t: Tile, box: list[int]) -> bool:
    x, y, w, h = box
    return t.x <= x and t.y <= y and x + w <= t.x + t.w and y + h <= t.y + t.h


def merge(found: list[tuple[vision.ReadCard, Tile]], photo: tuple[int, int]) -> list[vision.ReadCard]:
    """Keep each card once: of entries boxed on the same spot by different tiles, the one seen
    furthest from a seam. Entries without a box can't be told apart and are dropped, and so are
    entries cut off by a seam that another tile sees whole: the tile's edge cuts the title too,
    and the model reads what's left of it (scan 18: "Genesis" for Genesis Chamber, "Spring" for
    Springjack Pasture), often another real card."""
    tiles = {id(t): t for _, t in found}.values()
    boxed = [(c, t, _edge_margin(c.extra["bbox"], t, photo)) for c, t in found if c.extra.get("bbox")]
    boxed = [(c, t, m) for c, t, m in boxed
             if m > SEAM * min(t.w, t.h) or not any(o is not t and _contains(o, c.extra["bbox"]) for o in tiles)]
    ranked = sorted(boxed, key=lambda e: -e[2])
    kept: list[tuple[vision.ReadCard, Tile]] = []
    for c, t, _ in ranked:
        same = False
        for k, kt in kept:
            if kt is t:
                continue  # one tile's own entries were already de-duplicated
            iou = piles._iou(c.extra["bbox"], k.extra["bbox"])
            if iou > SAME_CARD_IOU or (iou > SAME_NAME_IOU and c.name.casefold() == k.name.casefold()):
                same = True
                break
        if not same:
            kept.append((c, t))
    # Back in reading order: top to bottom, left to right.
    kept.sort(key=lambda e: (e[0].extra["bbox"][1] + e[0].extra["bbox"][3] / 2, e[0].extra["bbox"][0]))
    return [c for c, _ in kept]


async def read(img: np.ndarray, cfg: vision.VisionConfig, sem: asyncio.Semaphore, max_side: int,
               cards: int | None, card: tuple[float | None, float | None] = (None, None)
               ) -> tuple[list[vision.ReadCard], list[dict]]:
    """Read the photo tile by tile. Returns the merged entries (boxes in photo pixels, as
    extra["bbox"]) and a debug record per tile."""
    H, W = img.shape[:2]
    tiles = plan(W, H, cards, card)
    progress.stage(f"Reading the photo again in {len(tiles)} tiles", len(tiles))

    async def one(t: Tile) -> tuple[Tile, list[vision.ReadCard], str | None, str | None]:
        crop = img[t.y:t.y + t.h, t.x:t.x + t.w]
        long = max(t.w, t.h)
        scale = min(max_side, max(MIN_SIDE, long)) / long
        if scale != 1.0:
            crop = cv2.resize(crop, (round(t.w * scale), round(t.h * scale)),
                              interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        try:
            async with sem:
                entries, reply = await vision.read_cards_raw(cfg, buf.tobytes(), False, prompt=vision.TILE_PROMPT)
        except vision.VisionError as e:
            progress.tick()
            return t, [], None, str(e)
        progress.tick()
        piles.place_boxes(entries, (crop.shape[1], crop.shape[0]), (t.w, t.h))
        piles.drop_runaway(entries)
        for e in entries:
            if e.extra.get("bbox"):
                x, y, w, h = e.extra["bbox"]
                e.extra["bbox"] = [x + t.x, y + t.y, w, h]
        entries, _ = piles.drop_duplicates(entries)
        return t, entries, reply, None

    results = await asyncio.gather(*(one(t) for t in tiles))
    found = [(c, t) for t, entries, _, _ in results for c in entries]
    debug = [{"region": [t.x, t.y, t.w, t.h], "cards": len(entries), "reply": reply, "error": err}
             for t, entries, reply, err in results]
    return merge(found, (W, H)), debug
