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
from . import detection, matching, resolver, scryfall, vision

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


async def _process_image(img_row: ScanImage, prepass: bool, cfg: vision.VisionConfig,
                         sem: asyncio.Semaphore, max_side: int, layout: str = "piles") -> _ImageResult:
    img = await asyncio.to_thread(cv2.imread, img_row.path)
    if img is None:
        raise vision.VisionError(f"Could not open {img_row.filename}")

    async def ask(jpeg: bytes, single: bool) -> list[vision.ReadCard]:
        async with sem:
            return await vision.read_cards(cfg, jpeg, single, layout)

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
    tasks.append(("photo", None, None, asyncio.create_task(ask(_jpeg_for_model(img, max_side), False))))

    reads: list[_Read] = []
    errors: list[Exception] = []
    photo_cards, photo_error = None, None
    for source, crop_path, bbox, task in tasks:
        try:
            cards = await task
        except vision.VisionError as e:
            errors.append(e)
            if source == "photo":
                photo_error = str(e)
                log.warning("Whole-photo read of %s failed: %s", img_row.filename, e)
            continue
        if source == "photo":
            photo_cards = len(cards)
        reads.extend(_Read(source, crop_path, bbox, rc) for rc in cards)
    if errors and len(errors) == len(tasks):
        raise errors[0]
    return _ImageResult(len(found), reads, photo_cards, photo_error)


WEAK_CROP_SCORE = 85.0
WEAK_CROP_CONFIDENCE = 0.6


def _unconfirmed(det: Detection, photo_card_ids: set[int]) -> bool:
    """A shaky crop read for a card the whole-photo read didn't see is usually a bad crop
    (e.g. only the artwork was cut out) rather than a real extra card."""
    if det.source != "crop" or det.card_id in photo_card_ids:
        return False
    return det.card_id is None or det.match_score < WEAK_CROP_SCORE or det.model_confidence < WEAK_CROP_CONFIDENCE


def _flags_for(line: dict, threshold: float) -> list[str]:
    flags = []
    if line["card_id"] is None:
        flags.append("unmatched")
    elif line["confidence"] < threshold:
        flags.append("low_confidence")
    if line.get("disagree"):
        flags.append("count_mismatch")
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
            if c and p and c != p:
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
        line = {"card_id": card_id, "name": names.get(card_id), "qty": data["qty"],
                "confidence": round(conf, 3), "disagree": data["disagree"]}
        lines.append(ScanLine(
            job_id=job.id,
            card_id=card_id,
            raw_name=best.raw_name,
            qty=data["qty"],
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
            img.photo_cards = img.photo_error = None
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
                result = await _process_image(img, job.prepass, cfg, sem, vs["image_max_side"], job.layout)
                img.crops_found, img.photo_cards, img.photo_error = result.crops, result.photo_cards, result.photo_error
                reads = result.reads
                job.step = f"Identifying cards in photo {n} of {len(job.images)}"
                db.commit()
                for r in reads:
                    res = await resolver.resolve(db, matching.get_index(db), r.card, r.crop_path, cfg, opts, ask)
                    img.detections.append(Detection(
                        source=r.source, crop_path=r.crop_path, bbox=r.bbox,
                        raw_name=r.card.name[:255], raw_set=(r.card.set or "")[:20] or None,
                        raw_number=(r.card.collector_number or "")[:20] or None,
                        qty=r.card.count if r.source == "photo" else 1,
                        model_confidence=res.confidence if res.confidence is not None else r.card.confidence,
                        card_id=res.card_id, match_score=res.score, match_method=res.method,
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
