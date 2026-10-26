"""Background worker: turns uploaded photos into reviewable scan lines."""

import asyncio
import hashlib
import io
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import httpx
import numpy as np
from PIL import Image, ImageOps
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import app_settings, build_id
from ..db import session_scope
from ..models import Card, Detection, ScanImage, ScanJob, ScanLine
from . import cascades, decklists, detection, evidence, matching, piles, progress, resolver, scryfall, tiles, vision

log = logging.getLogger(__name__)

BASIC_LANDS = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes",
               "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp",
               "Snow-Covered Mountain", "Snow-Covered Forest"}
STORED_MAX_SIDE = 4096  # full phone resolution: title strips and tiles need every pixel

_wakeup = asyncio.Event()


def notify() -> None:
    _wakeup.set()


# --- upload handling -------------------------------------------------------

def store_upload(job_dir: Path, index: int, filename: str, data: bytes) -> dict:
    """Validate an uploaded photo, fix EXIF rotation, shrink it and save as JPEG."""
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:
        raise ValueError(f"{filename} is not an image we can read") from e
    img.thumbnail((STORED_MAX_SIDE, STORED_MAX_SIDE))
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / f"photo-{index:03d}.jpg"
    img.save(path, "JPEG", quality=90)
    return {"filename": filename[:255], "path": str(path), "width": img.width, "height": img.height,
            "sha256": hashlib.sha256(data).hexdigest()}


def _jpeg_for_model(img: np.ndarray, max_side: int) -> bytes:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("Could not encode image")
    return buf.tobytes()


# --- processing ------------------------------------------------------------

@dataclass
class _Read:
    source: str
    crop_path: str | None
    bbox: list[int] | None
    card: vision.ReadCard


@dataclass
class _ImageResult:
    crops: int
    reads: list[_Read]
    photo_cards: int | None  # None when the whole-photo read failed
    photo_error: str | None
    debug: dict | None = None


async def _process_image(img_row: ScanImage, prepass: bool, cfg: vision.VisionConfig,
                         sem: asyncio.Semaphore, max_side: int, layout: str = "piles",
                         zoom: bool = True, index: matching.CardIndex | None = None) -> _ImageResult:
    img = await asyncio.to_thread(cv2.imread, img_row.path)
    if img is None:
        raise vision.VisionError(f"Could not open {img_row.filename}")

    async def ask(jpeg: bytes, single: bool) -> list[vision.ReadCard]:
        async with sem:
            return await vision.read_cards(cfg, jpeg, single, layout)

    async def ask_photo(jpeg: bytes) -> tuple[list[vision.ReadCard], str]:
        async with sem:
            return await vision.read_cards_raw(cfg, jpeg, False, layout)

    # 1. Whole photo first. When it gives usable boxes, the close-ups re-read each card and pile.
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    sent = (int(w * scale), int(h * scale))
    reads: list[_Read] = []
    photo_cards, photo_error, debug = None, None, None
    cards: list[vision.ReadCard] = []
    crop_dir, stem = Path(img_row.path).parent / "crops", Path(img_row.path).stem
    progress.stage("Finding card outlines")
    outlines = await asyncio.to_thread(detection.find_cards, img)  # cards OpenCV sees whole (no model)
    progress.stage("Reading the whole photo")
    try:
        cards, reply = await ask_photo(_jpeg_for_model(img, max_side))
    except vision.VisionError as e:
        photo_error = str(e)
        debug = {"error": str(e)}
        log.warning("Whole-photo read of %s failed: %s", img_row.filename, e)
    else:
        box_scale = piles.place_boxes(cards, sent, (w, h))
        listed = len(cards)
        runaway = piles.drop_runaway(cards)
        debug = {"reply": reply, "sent_size": list(sent), "box_scale": box_scale, "runaway": runaway}
        # 2. Many small cards: read the full-resolution photo again in overlapping tiles.
        card_w = float(np.median([min(f.bbox[2:]) for f in outlines])) if outlines else None
        why = tiles.reason(listed, vision.cut_off(reply), runaway, card_w, w)
        if zoom and why:
            tiled, tile_debug = await tiles.read(img, cfg, sem, max_side, listed)
            debug["tiled"], debug["tiles"] = why, tile_debug
            if tiled:
                cards, box_scale = tiled, box_scale or "tiles"
        cards, duplicates = piles.drop_duplicates(cards)
        debug["duplicates"] = [{"name": d.name, "count": d.count, "box": d.extra.get("bbox")} for d in duplicates]

    # 3. Cascading columns: OpenCV finds each title bar; each is cut out and read on its own,
    #    replacing what the whole-photo read listed there.
    strips: list[vision.ReadCard] = []
    if zoom and outlines:
        progress.stage("Looking for cascading columns")
        columns = await asyncio.to_thread(cascades.find, img, [f.corners for f in outlines])
        if columns:
            strips = await _read_strips(img, columns, crop_dir, stem, cfg, sem)
            covered = [np.float32(c.extra["strip"]["corners"]) for c in strips]
            before = len(cards)
            cards = [c for c in cards if not _inside_any(c.extra.get("bbox"), covered)]
            debug = {**(debug or {}), "columns": [{"front": outlines[col.front].bbox, "strips": len(col.strips)}
                                                  for col in columns],
                     "replaced_by_strips": before - len(cards)}

    if not photo_error:
        if zoom and box_scale:
            # The model's boxes are rough; line them up with the cards OpenCV can find (no model calls).
            progress.stage("Lining up boxes and counting pile edges")
            debug["snap"] = piles.snap_boxes(cards, [f.bbox for f in outlines], [f.corners for f in outlines])
            debug["snap"]["resized"] = piles.fit_sizes(cards, [f.bbox for f in outlines], (w, h))
            await asyncio.to_thread(piles.count_fans, img, cards)
            await piles.zoom_in(img, cards, crop_dir, stem, cfg, sem)
            if index is not None:
                await piles.settle(img, cards, index, cfg, sem)
        photo_cards = len(cards)
        reads.extend(_Read("photo", c.extra.get("crop_path"), c.extra.get("bbox"), c) for c in cards)
    reads.extend(_Read("strip", c.extra.get("crop_path"), c.extra.get("bbox"), c) for c in strips)

    # 4. Pre-pass only as a fallback: when the whole-photo read failed or gave no usable boxes,
    #    nothing got a close-up, so cut cards out with OpenCV and read them one at a time.
    has_boxes = any(r.bbox for r in reads)
    found: list[detection.FoundCard] = []
    if prepass and (photo_error or not has_boxes):
        reason = "whole-photo read failed" if photo_error else "no usable boxes from the model"
        debug = {**(debug or {}), "prepass": reason}
        found = outlines
        progress.stage("Reading cut-out cards one by one", len(found))
        crop_dir = Path(img_row.path).parent / "crops"
        crop_dir.mkdir(exist_ok=True)
        tasks = []
        for i, fc in enumerate(found):
            crop_path = crop_dir / f"{Path(img_row.path).stem}-{i:03d}.jpg"
            cv2.imwrite(str(crop_path), fc.image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            tasks.append((str(crop_path), fc.bbox, asyncio.create_task(ask(_jpeg_for_model(fc.image, 800), True))))
        crop_errors = []
        for crop_path, bbox, task in tasks:
            try:
                reads.extend(_Read("crop", crop_path, bbox, rc) for rc in await task)
            except vision.VisionError as e:
                crop_errors.append(e)
            progress.tick()
        if photo_error and (not tasks or len(crop_errors) == len(tasks)):
            raise vision.VisionError(photo_error)
    elif photo_error and not reads:
        raise vision.VisionError(photo_error)
    return _ImageResult(len(found), reads, photo_cards, photo_error, debug)


async def _read_strips(img: np.ndarray, columns: list[cascades.Column], crop_dir: Path, stem: str,
                       cfg: vision.VisionConfig, sem: asyncio.Semaphore) -> list[vision.ReadCard]:
    """Cut out each title strip of each column and read it on its own."""
    crop_dir.mkdir(exist_ok=True)
    progress.stage("Reading title strips", sum(len(col.strips) for col in columns))

    async def one(c: int, k: int, strip: cascades.Strip) -> vision.ReadCard:
        ok, buf = cv2.imencode(".jpg", cascades.cut(img, strip), [cv2.IMWRITE_JPEG_QUALITY, 92])
        path = crop_dir / f"{stem}-col{c:02d}-{k:02d}.jpg"
        path.write_bytes(buf.tobytes())
        async with sem:
            read = await vision.read_title(cfg, buf.tobytes())
        progress.tick()
        name = read.name if read and read.name else f"Unreadable title (column {c + 1}, card {k + 1})"
        return vision.ReadCard(name=name, count=1, confidence=read.confidence if read and read.name else 0.0,
                               extra={"bbox": strip.bbox, "crop_path": str(path),
                                      "strip": {"column": c + 1, "card": k + 1, "corners": strip.corners,
                                                "edge_score": strip.score, "reply": read.raw if read else None}})

    return list(await asyncio.gather(*(one(c, k, s) for c, col in enumerate(columns)
                                       for k, s in enumerate(col.strips))))


def _inside_any(bbox: list[int] | None, quads: list[np.ndarray]) -> bool:
    """Whether a box's centre lies on one of the title strips."""
    if not bbox:
        return False
    centre = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
    return any(cv2.pointPolygonTest(q.reshape(-1, 1, 2), centre, False) >= 0 for q in quads)


def _doubts(index: matching.CardIndex, card: vision.ReadCard, res: resolver.Resolution) -> list[str]:
    """Small reasons to trust a whole-photo entry's final name a little less (see evidence.py)."""
    out = []
    zoom: vision.ZoomRead | None = card.extra.get("zoom")
    if zoom and zoom.retried:
        out.append("title_retry")
    hits = [h[0] for h in (index.match(card.name, limit=1), index.match(zoom.name if zoom else "", limit=1)) if h]
    if len(hits) == 2 and min(h.score for h in hits) >= SURE_MATCH and hits[0].card_id != hits[1].card_id:
        out.append("reads_disagree")
    if card.extra.get("tiebreak") or res.method == "choice":
        out.append("model_chose")
    return out


async def _check_types(db: Session, job: ScanJob) -> None:
    """Compare the type line each close-up read with the matched card's real type."""
    dets = [d for img in job.images for d in img.detections if d.card_id and "type_read" in (d.raw or {})]
    if not dets:
        return
    await scryfall.fetch_details(db, list(db.scalars(select(Card).where(Card.id.in_({d.card_id for d in dets})))))
    for d in dets:
        result = evidence.type_check(d.raw["type_read"], db.get(Card, d.card_id).type_line)
        if result in evidence.DOUBTS:
            d.raw = {**d.raw, "doubts": [*d.raw.get("doubts", []), result]}


UNCONFIRMED_CONFIDENCE = 0.4  # name only from the whole-table read; the close-up saw no title
WEAK_CROP_SCORE = 85.0
WEAK_CROP_CONFIDENCE = 0.6


def _unconfirmed(det: Detection, photo_card_ids: set[int]) -> bool:
    """A shaky crop read for a card the whole-photo read didn't see is usually a bad crop
    (e.g. only the artwork was cut out) rather than a real extra card."""
    if det.source != "crop" or det.card_id in photo_card_ids:
        return False
    return det.card_id is None or det.match_score < WEAK_CROP_SCORE or det.model_confidence < WEAK_CROP_CONFIDENCE


SURE_MATCH = 85.0  # below this the "match" is only the nearest spelling
ACCEPT_MATCH = 80.0  # below this it isn't taken at all ("Force of One" is not Forest)

# Constructed decks allow at most 4 copies of a card, except basic lands and cards that
# say otherwise. Counts above the limit are almost always a miscounted pile.
ANY_NUMBER = {"Relentless Rats", "Shadowborn Apostle", "Rat Colony", "Persistent Petitioners",
              "Dragon's Approach", "Slime Against Humanity", "Hare Apparent", "Templar Knight",
              "Tempest Hawk"}
LIMITED_NUMBER = {"Seven Dwarves": 7, "Nazgûl": 9}


def copy_limit(name: str | None) -> int | None:
    """Most copies a deck may hold, or None when there is no limit."""
    if not name or name in BASIC_LANDS or name in ANY_NUMBER:
        return None
    return LIMITED_NUMBER.get(name, 4)


def _flags_for(line: dict, threshold: float) -> list[str]:
    flags = []
    if line["card_id"] is None:
        flags.append("unmatched")
    elif line["confidence"] < threshold or line.get("match_score", 100) < SURE_MATCH:
        flags.append("low_confidence")
    if line.get("type_mismatch"):
        flags.append("type_mismatch")
    if line.get("disagree"):
        flags.append("count_mismatch")
    if line.get("capped"):
        flags.append("over_limit")
    if line.get("name") in BASIC_LANDS and line["qty"] > 4:
        flags.append("check_basic_count")
    return flags


def build_lines(db: Session, job: ScanJob, threshold: float) -> list[ScanLine]:
    """Aggregate detections into one line per card.

    Within a photo the quantity is the larger of (crops showing the card, count read
    from the whole photo), so the two passes never double count. Photos are assumed
    to show different physical cards and are summed. Cards in a photo marked as another
    board (a photo of the sideboard) make lines of their own.
    """
    index = matching.get_index(db)
    names = {}
    agg: dict[tuple, dict] = {}
    for img in job.images:
        board = img.board or "main"
        crop_counts: dict[object, int] = defaultdict(int)
        photo_counts: dict[object, int] = defaultdict(int)
        per_key: dict[object, list[Detection]] = defaultdict(list)
        for d in img.detections:
            if d.match_method == "unconfirmed":
                continue
            key = d.card_id if d.card_id is not None else ("raw", matching.normalize(d.raw_name))
            (crop_counts if d.source == "crop" else photo_counts)[key] += d.qty
            per_key[key].append(d)
        for key, dets in per_key.items():
            c, p = crop_counts.get(key, 0), photo_counts.get(key, 0)
            line = agg.setdefault((key, board), {"qty": 0, "dets": [], "disagree": False})
            line["qty"] += max(c, p)
            line["dets"].extend(dets)
            if (c and p and c != p) or any((d.raw or {}).get("count_differs") for d in dets):
                line["disagree"] = True

    lines = []
    for (key, board), data in agg.items():
        dets: list[Detection] = data["dets"]
        best = max(dets, key=lambda d: d.model_confidence * d.match_score)
        conf = min(d.model_confidence * (d.match_score / 100 if d.card_id else 0)
                   * evidence.factor((d.raw or {}).get("doubts", [])) for d in dets)
        card_id = key if not isinstance(key, tuple) else None
        if card_id is not None and card_id not in names:
            names[card_id] = db.get(Card, card_id).name
        # Cards the rest of the row was read as come first: when a repeated whole-table name and
        # an unreadable close-up leave a card to the user, it's usually one of those.
        suggestions = [{"card_id": m.card_id, "name": m.name, "score": m.score}
                       for name in dict.fromkeys(n for d in dets for n in (d.raw or {}).get("suggest", []))
                       for m in index.match(name, limit=1) if m.card_id != card_id]
        suggestions += [
            {"card_id": m.card_id, "name": m.name, "score": m.score}
            for m in index.match(best.raw_name, limit=4)
            if m.card_id != card_id and m.score >= matching.MIN_SCORE and m.card_id not in {x["card_id"] for x in suggestions}
        ]
        suggestions = suggestions[:4]
        qty, capped = data["qty"], False
        limit = copy_limit(names.get(card_id)) if card_id is not None else None
        if limit is not None and qty > limit:
            qty, capped = limit, True
        line = {"card_id": card_id, "name": names.get(card_id), "qty": qty,
                "confidence": round(conf, 3), "disagree": data["disagree"], "capped": capped,
                "type_mismatch": any("type_mismatch" in (d.raw or {}).get("doubts", []) for d in dets),
                "match_score": max(d.match_score for d in dets)}
        lines.append(ScanLine(
            job_id=job.id,
            card_id=card_id,
            raw_name=best.raw_name,
            qty=qty,
            board=board,
            confidence=line["confidence"],
            flags=_flags_for(line, threshold),
            suggestions=suggestions,
            detection_ids=[d.id for d in dets],
        ))
    lines.sort(key=lambda ln: (not ln.flags, names.get(ln.card_id) or ln.raw_name))
    return lines


def line_facts(db: Session, lines: list[ScanLine]) -> list[dict]:
    return [{"card_id": ln.card_id, "name": db.get(Card, ln.card_id).name if ln.card_id else ln.raw_name,
             "qty": ln.qty} for ln in lines]


def record_score(db: Session, job: ScanJob, lines: list[ScanLine]) -> None:
    """Score this read against the real decklist the user gave (before any edits)."""
    result = decklists.score(job.expected, line_facts(db, lines))
    entry = {"at": datetime.now(UTC).isoformat(), "build": build_id(), "model": job.model,
             "right": result["right"], "total": result["total"],
             "missing": result["missing"][:40], "extra": result["extra"][:40]}
    job.score_history = [*(job.score_history or []), entry][-30:]


async def process_job(job_id: int) -> None:
    with session_scope() as db:
        job = db.get(ScanJob, job_id)
        if job is None:
            return
        vs = app_settings.get_group(db, "vision")
        cfg = vision.VisionConfig.from_settings(vs)
        if job.model:
            cfg.model = job.model
        job.model = cfg.model
        job.status = "processing"
        job.processed_images = 0
        job.error = None
        for img in job.images:
            img.detections.clear()
            img.crops_found = 0
            img.photo_cards = img.photo_error = img.debug = None
        job.lines.clear()
        db.commit()
        sem = asyncio.Semaphore(max(1, vs["max_concurrency"]))
        opts = resolver.Options.from_settings(vs)

        async def ask(coro):
            async with sem:
                return await coro

        try:
            index = matching.get_index(db)
            if len(index) == 0:
                job.step = "Downloading card names from Scryfall"
                db.commit()
                try:
                    await scryfall.sync_card_names(db)
                except httpx.HTTPError as e:
                    raise RuntimeError(
                        f"The card list is empty and Scryfall could not be reached ({e}). "
                        "Check the container's internet access, then use Settings → Card database."
                    ) from e
                index = matching.get_index(db)
            progress.start(job.id, len(job.images))
            for n, img in enumerate(job.images, start=1):
                job.step = f"Reading photo {n} of {len(job.images)}"
                progress.photo(n)
                db.commit()
                result = await _process_image(img, job.prepass, cfg, sem, vs["image_max_side"], job.layout,
                                              vs["pile_zoom"], matching.get_index(db))
                img.crops_found, img.photo_cards, img.photo_error = result.crops, result.photo_cards, result.photo_error
                img.debug = result.debug
                reads = result.reads
                job.step = f"Identifying cards in photo {n} of {len(job.images)}"
                db.commit()
                photo_reads = [r.card for r in reads if r.source == "photo"]
                progress.stage("Identifying cards", len(reads))
                for r in reads:
                    progress.tick()
                    card, raw = r.card, {"reply_entry": r.card.extra.get("raw")}
                    if r.source == "strip":
                        raw = {"strip": r.card.extra["strip"]}
                    if r.source == "photo":
                        choice = piles.decide(matching.get_index(db), r.card, photo_reads)
                        zoom = r.card.extra.get("zoom")
                        raw = {"photo": r.card.extra.get("raw"), "used": choice.used,
                               "count_differs": choice.count_differs,
                               "unsure": choice.unsure, "zoom_region": r.card.extra.get("zoom_region"),
                               "model_box": r.card.extra.get("model_bbox"), "snapped": r.card.extra.get("snapped"),
                               "resized": r.card.extra.get("resized"),
                               "shifted": r.card.extra.get("shifted"), "tiebreak": r.card.extra.get("tiebreak"),
                               "fan": (fan := r.card.extra.get("fan")) and {"count": fan.count, "direction": fan.direction,
                                                                            "edges": fan.edges},
                               "zoom": zoom and {"name": zoom.name, "count": zoom.count, "retried": zoom.retried,
                                                 "confidence": zoom.confidence, "reply": zoom.raw}}
                        conf = zoom.confidence if choice.used == "zoom" and zoom else r.card.confidence
                        if choice.unsure:
                            conf = min(conf, UNCONFIRMED_CONFIDENCE)
                        card = vision.ReadCard(name=choice.name, count=choice.count, confidence=conf,
                                               set=r.card.set, collector_number=r.card.collector_number)
                    res = await resolver.resolve(db, matching.get_index(db), card, r.crop_path, cfg, opts, ask)
                    if r.source == "photo" and res.score < SURE_MATCH and not (choice.repeated and choice.used == "zoom"):
                        # Neither read matched cleanly. Look the other name up too (the close-up may
                        # have read the type line, the whole-table read a foreign title) and keep
                        # whichever finds a card more convincingly.
                        zoom = r.card.extra.get("zoom")
                        other = r.card.name if choice.used == "zoom" else (zoom.name if zoom else "")
                        if other and other.casefold() != card.name.casefold():
                            alt = vision.ReadCard(name=other, count=card.count, confidence=card.confidence,
                                                  set=r.card.set, collector_number=r.card.collector_number)
                            alt_res = await resolver.resolve(db, matching.get_index(db), alt, None, cfg, opts, ask)
                            raw["both_names"] = {"kept": card.name, "other": other,
                                                 "scores": [res.score, alt_res.score]}
                            if alt_res.card_id and alt_res.score > res.score:
                                card, res = alt, alt_res
                                raw["both_names"]["kept"] = other
                                raw["used"] = "photo" if choice.used == "zoom" else "zoom"
                    if res.method in ("name", "choice") and res.score < ACCEPT_MATCH:
                        # Too far from any card name to guess: leave it for the user to pick.
                        res.card_id = None
                    if r.source == "photo" and choice.suggest:
                        raw["suggest"] = choice.suggest
                    if r.source == "photo":
                        raw["doubts"] = _doubts(matching.get_index(db), r.card, res)
                        if zoom:
                            raw["type_read"] = zoom.type_line
                    img.detections.append(Detection(
                        source=r.source, crop_path=r.crop_path, bbox=r.bbox,
                        raw_name=card.name[:255], raw_set=(r.card.set or "")[:20] or None,
                        raw_number=(r.card.collector_number or "")[:20] or None,
                        qty=card.count if r.source == "photo" else 1,
                        model_confidence=res.confidence if res.confidence is not None else card.confidence,
                        card_id=res.card_id, match_score=res.score, match_method=res.method, raw=raw,
                    ))
                progress.finish()
                img.debug = {**(img.debug or {}), "timings": progress.current().timings(n)}
                if result.photo_error is None:
                    seen = {d.card_id for d in img.detections if d.source == "photo" and d.card_id}
                    for d in img.detections:
                        if _unconfirmed(d, seen):
                            d.match_method = "unconfirmed"
                job.processed_images = n
                db.commit()
            job.step = "Checking card types"
            progress.photo(0)
            progress.stage("Checking card types")
            db.commit()
            await _check_types(db, job)
            job.step = "Building decklist"
            progress.stage("Building the decklist")
            db.flush()
            lines = build_lines(db, job, vs["confidence_threshold"])
            for line in lines:
                db.add(line)
            db.flush()
            if job.expected:
                record_score(db, job, lines)
            # Card pictures for the review screen (lines without a cut-out show these instead).
            job.step = "Fetching card images"
            progress.stage("Fetching card images")
            db.commit()
            card_ids = {ln.card_id for ln in lines if ln.card_id}
            await scryfall.fetch_details(db, list(db.scalars(select(Card).where(Card.id.in_(card_ids)))))
            job.status = "review"
            job.step = ""
        except Exception as e:
            log.exception("Scan %s failed", job_id)
            job.status = "failed"
            job.error = str(e) or e.__class__.__name__
        finally:
            progress.end(job.id)
        job.finished_at = datetime.now(UTC)


async def worker_loop() -> None:
    with session_scope() as db:
        # Anything interrupted by a restart goes back in the queue.
        for job in db.scalars(select(ScanJob).where(ScanJob.status == "processing")):
            job.status = "queued"
    while True:
        with session_scope() as db:
            job_id = db.scalar(select(ScanJob.id).where(ScanJob.status == "queued").order_by(ScanJob.id).limit(1))
        if job_id is None:
            _wakeup.clear()
            try:
                await asyncio.wait_for(_wakeup.wait(), timeout=30)
            except TimeoutError:
                pass
            continue
        try:
            await process_job(job_id)
        except Exception:
            log.exception("Worker error on scan %s", job_id)
            with session_scope() as db:
                job = db.get(ScanJob, job_id)
                if job and job.status in ("queued", "processing"):
                    job.status = "failed"
                    job.error = "Unexpected error; see the server log."
