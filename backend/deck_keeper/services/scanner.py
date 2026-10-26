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

from .. import app_settings
from ..db import session_scope
from ..models import Card, Detection, ScanImage, ScanJob, ScanLine
from . import detection, matching, piles, resolver, scryfall, vision

log = logging.getLogger(__name__)

BASIC_LANDS = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes",
               "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp",
               "Snow-Covered Mountain", "Snow-Covered Forest"}
STORED_MAX_SIDE = 2400

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
                         zoom: bool = True) -> _ImageResult:
    img = await asyncio.to_thread(cv2.imread, img_row.path)
    if img is None:
        raise vision.VisionError(f"Could not open {img_row.filename}")

    async def ask(jpeg: bytes, single: bool) -> list[vision.ReadCard]:
        async with sem:
            return await vision.read_cards(cfg, jpeg, single, layout)

    async def ask_photo(jpeg: bytes) -> tuple[list[vision.ReadCard], str]:
        async with sem:
            return await vision.read_cards_raw(cfg, jpeg, False, layout)

    tasks: list[tuple[str, str | None, list[int] | None, asyncio.Task]] = []
    found: list[detection.FoundCard] = []
    if prepass:
        found = await asyncio.to_thread(detection.find_cards, img)
        crop_dir = Path(img_row.path).parent / "crops"
        crop_dir.mkdir(exist_ok=True)
        for i, fc in enumerate(found):
            crop_path = crop_dir / f"{Path(img_row.path).stem}-{i:03d}.jpg"
            cv2.imwrite(str(crop_path), fc.image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            jpeg = _jpeg_for_model(fc.image, 800)
            tasks.append(("crop", str(crop_path), fc.bbox, asyncio.create_task(ask(jpeg, True))))
    # Always read the whole photo too: it catches cards the detector missed (fanned,
    # overlapping) and gives a second opinion on counts.
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    sent = (int(w * scale), int(h * scale))
    tasks.append(("photo", None, None, asyncio.create_task(ask_photo(_jpeg_for_model(img, max_side)))))

    reads: list[_Read] = []
    errors: list[Exception] = []
    photo_cards, photo_error, debug = None, None, None
    for source, crop_path, bbox, task in tasks:
        try:
            result = await task
        except vision.VisionError as e:
            errors.append(e)
            if source == "photo":
                photo_error = str(e)
                debug = {"error": str(e)}
                log.warning("Whole-photo read of %s failed: %s", img_row.filename, e)
            continue
        if source != "photo":
            reads.extend(_Read(source, crop_path, bbox, rc) for rc in result)
            continue
        cards, reply = result
        box_scale = piles.place_boxes(cards, sent, (w, h))
        cards, duplicates = piles.drop_duplicates(cards)
        if zoom and box_scale:
            await piles.zoom_in(img, cards, Path(img_row.path).parent / "crops", Path(img_row.path).stem, cfg, sem)
        photo_cards = len(cards)
        debug = {"reply": reply, "sent_size": list(sent), "box_scale": box_scale,
                 "duplicates": [{"name": d.name, "count": d.count, "box": d.extra.get("bbox")} for d in duplicates]}
        reads.extend(_Read("photo", c.extra.get("crop_path"), c.extra.get("bbox"), c) for c in cards)
    if errors and len(errors) == len(tasks):
        raise errors[0]
    return _ImageResult(len(found), reads, photo_cards, photo_error, debug)


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
    to show different physical cards and are summed.
    """
    index = matching.get_index(db)
    names = {}
    agg: dict[object, dict] = {}
    for img in job.images:
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
            line = agg.setdefault(key, {"qty": 0, "dets": [], "disagree": False})
            line["qty"] += max(c, p)
            line["dets"].extend(dets)
            if (c and p and c != p) or any((d.raw or {}).get("count_differs") for d in dets):
                line["disagree"] = True

    lines = []
    for key, data in agg.items():
        dets: list[Detection] = data["dets"]
        best = max(dets, key=lambda d: d.model_confidence * d.match_score)
        conf = min(d.model_confidence * (d.match_score / 100 if d.card_id else 0) for d in dets)
        card_id = key if not isinstance(key, tuple) else None
        if card_id is not None and card_id not in names:
            names[card_id] = db.get(Card, card_id).name
        suggestions = [
            {"card_id": m.card_id, "name": m.name, "score": m.score}
            for m in index.match(best.raw_name, limit=4) if m.card_id != card_id and m.score >= matching.MIN_SCORE
        ][:3]
        qty, capped = data["qty"], False
        limit = copy_limit(names.get(card_id)) if card_id is not None else None
        if limit is not None and qty > limit:
            qty, capped = limit, True
        line = {"card_id": card_id, "name": names.get(card_id), "qty": qty,
                "confidence": round(conf, 3), "disagree": data["disagree"], "capped": capped,
                "match_score": max(d.match_score for d in dets)}
        lines.append(ScanLine(
            job_id=job.id,
            card_id=card_id,
            raw_name=best.raw_name,
            qty=qty,
            board="main",
            confidence=line["confidence"],
            flags=_flags_for(line, threshold),
            suggestions=suggestions,
            detection_ids=[d.id for d in dets],
        ))
    lines.sort(key=lambda ln: (not ln.flags, names.get(ln.card_id) or ln.raw_name))
    return lines


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
            for n, img in enumerate(job.images, start=1):
                job.step = f"Reading photo {n} of {len(job.images)}"
                db.commit()
                result = await _process_image(img, job.prepass, cfg, sem, vs["image_max_side"], job.layout,
                                              vs["pile_zoom"])
                img.crops_found, img.photo_cards, img.photo_error = result.crops, result.photo_cards, result.photo_error
                img.debug = result.debug
                reads = result.reads
                job.step = f"Identifying cards in photo {n} of {len(job.images)}"
                db.commit()
                photo_reads = [r.card for r in reads if r.source == "photo"]
                for r in reads:
                    card, raw = r.card, {"reply_entry": r.card.extra.get("raw")}
                    if r.source == "photo":
                        choice = piles.decide(matching.get_index(db), r.card, photo_reads)
                        zoom = r.card.extra.get("zoom")
                        raw = {"photo": r.card.extra.get("raw"), "used": choice.used,
                               "count_differs": choice.count_differs,
                               "unsure": choice.unsure, "zoom_region": r.card.extra.get("zoom_region"),
                               "zoom": zoom and {"name": zoom.name, "count": zoom.count, "retried": zoom.retried,
                                                 "confidence": zoom.confidence, "reply": zoom.raw}}
                        conf = zoom.confidence if choice.used == "zoom" and zoom else r.card.confidence
                        if choice.unsure:
                            conf = min(conf, UNCONFIRMED_CONFIDENCE)
                        card = vision.ReadCard(name=choice.name, count=choice.count, confidence=conf,
                                               set=r.card.set, collector_number=r.card.collector_number)
                    res = await resolver.resolve(db, matching.get_index(db), card, r.crop_path, cfg, opts, ask)
                    img.detections.append(Detection(
                        source=r.source, crop_path=r.crop_path, bbox=r.bbox,
                        raw_name=card.name[:255], raw_set=(r.card.set or "")[:20] or None,
                        raw_number=(r.card.collector_number or "")[:20] or None,
                        qty=card.count if r.source == "photo" else 1,
                        model_confidence=res.confidence if res.confidence is not None else card.confidence,
                        card_id=res.card_id, match_score=res.score, match_method=res.method, raw=raw,
                    ))
                if result.photo_error is None:
                    seen = {d.card_id for d in img.detections if d.source == "photo" and d.card_id}
                    for d in img.detections:
                        if _unconfirmed(d, seen):
                            d.match_method = "unconfirmed"
                job.processed_images = n
                db.commit()
            job.step = "Building decklist"
            db.flush()
            lines = build_lines(db, job, vs["confidence_threshold"])
            for line in lines:
                db.add(line)
            db.flush()
            # Card pictures for the review screen (lines without a cut-out show these instead).
            job.step = "Fetching card images"
            db.commit()
            card_ids = {ln.card_id for ln in lines if ln.card_id}
            await scryfall.fetch_details(db, list(db.scalars(select(Card).where(Card.id.in_(card_ids)))))
            job.status = "review"
            job.step = ""
        except Exception as e:
            log.exception("Scan %s failed", job_id)
            job.status = "failed"
            job.error = str(e) or e.__class__.__name__
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
