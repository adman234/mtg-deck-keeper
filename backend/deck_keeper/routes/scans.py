import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import app_settings
from ..config import get_env
from ..db import get_db
from ..deps import current_user
from ..models import Card, Deck, DeckCard, Detection, ScanImage, ScanJob, ScanLine, User
from ..services import scanner, scryfall
from .cards import card_out

router = APIRouter(prefix="/api/scans", tags=["scans"])

MAX_PHOTOS = 30
MAX_PHOTO_BYTES = 30 * 1024 * 1024
BOARDS = {"main", "side", "commander", "maybe"}
LAYOUTS = ("auto", "piles", "spread")  # all read the same way now; kept for older clients


def _job(db: Session, job_id: int, user: User) -> ScanJob:
    job = db.get(ScanJob, job_id)
    if job is None or (job.user_id != user.id and not user.is_admin):
        raise HTTPException(404, "Scan not found")
    return job


def _job_dir(job_id: int) -> Path:
    return get_env().scans_dir / str(job_id)


def job_summary(job: ScanJob) -> dict:
    return {
        "id": job.id, "name": job.name, "status": job.status, "step": job.step, "error": job.error,
        "prepass": job.prepass, "layout": job.layout, "model": job.model, "deck_id": job.deck_id,
        "total_images": job.total_images, "processed_images": job.processed_images,
        "created_at": job.created_at.isoformat(),
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


def line_out(line: ScanLine, dets: dict[int, Detection], job_id: int) -> dict:
    sources = []
    for did in line.detection_ids:
        d = dets.get(did)
        if d is None:
            continue
        sources.append({
            "detection_id": d.id, "image_id": d.image_id, "source": d.source, "raw_name": d.raw_name,
            "qty": d.qty, "model_confidence": d.model_confidence, "match_score": d.match_score,
            "match_method": d.match_method, "bbox": d.bbox,
            "used": (d.raw or {}).get("used"),
            "set": d.raw_set, "collector_number": d.raw_number,
            "crop_url": f"/api/scans/{job_id}/crops/{d.id}" if d.crop_path else None,
        })
    return {
        "id": line.id, "card": card_out(line.card), "raw_name": line.raw_name, "qty": line.qty,
        "board": line.board, "confidence": line.confidence, "flags": line.flags,
        "needs_review": bool(line.flags) and not line.accepted, "accepted": line.accepted,
        "user_edited": line.user_edited, "suggestions": line.suggestions, "sources": sources,
    }


def job_detail(db: Session, job: ScanJob) -> dict:
    dets = {d.id: d for img in job.images for d in img.detections}
    return {
        **job_summary(job),
        "threshold": app_settings.get(db, "vision.confidence_threshold"),
        "images": [{"id": i.id, "filename": i.filename, "width": i.width, "height": i.height,
                    "crops_found": i.crops_found, "photo_cards": i.photo_cards, "photo_error": i.photo_error,
                    "board": i.board,
                    "unconfirmed_crops": sum(1 for d in i.detections if d.match_method == "unconfirmed"),
                    "url": f"/api/scans/{job.id}/photos/{i.id}"} for i in job.images],
        "lines": [line_out(ln, dets, job.id) for ln in job.lines],
    }


@router.get("")
def list_scans(db: Session = Depends(get_db), user: User = Depends(current_user)):
    jobs = db.scalars(select(ScanJob).where(ScanJob.user_id == user.id).order_by(ScanJob.id.desc()).limit(200))
    return [job_summary(j) for j in jobs]


@router.post("", status_code=201)
async def create_scan(
    files: list[UploadFile] = File(...),
    name: str = Form(""),
    prepass: bool | None = Form(None),  # default: the fallback setting
    layout: str = Form("auto"),
    model: str = Form(""),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    if not app_settings.get(db, "vision.enabled"):
        raise HTTPException(400, "Photo scanning is turned off in Settings")
    if not files:
        raise HTTPException(400, "Add at least one photo")
    if layout not in LAYOUTS:
        raise HTTPException(400, f"Layout must be one of {', '.join(LAYOUTS)}")
    if len(files) > MAX_PHOTOS:
        raise HTTPException(400, f"Upload at most {MAX_PHOTOS} photos per scan")
    if prepass is None:
        prepass = app_settings.get(db, "vision.prepass_default")
    job = ScanJob(user_id=user.id, name=name.strip()[:200], prepass=prepass, layout=layout, model=model.strip(),
                  total_images=len(files))
    db.add(job)
    db.flush()
    job_dir = _job_dir(job.id)
    try:
        for i, f in enumerate(files, start=1):
            data = await f.read(MAX_PHOTO_BYTES + 1)
            if len(data) > MAX_PHOTO_BYTES:
                raise ValueError(f"{f.filename} is larger than 30 MB")
            info = scanner.store_upload(job_dir, i, f.filename or f"photo-{i}", data)
            job.images.append(ScanImage(**info))
    except ValueError as e:
        db.rollback()
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, str(e)) from e
    db.commit()
    scanner.notify()
    return job_summary(job)


@router.get("/{job_id}")
def get_scan(job_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    return job_detail(db, _job(db, job_id, user))


class RerunIn(BaseModel):
    prepass: bool | None = None
    layout: str | None = None
    model: str | None = None


@router.post("/{job_id}/rerun")
def rerun(job_id: int, body: RerunIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    job = _job(db, job_id, user)
    if job.status in ("queued", "processing"):
        raise HTTPException(409, "This scan is already running")
    job.prepass = body.prepass if body.prepass is not None else app_settings.get(db, "vision.prepass_default")
    if body.layout is not None:
        if body.layout not in LAYOUTS:
            raise HTTPException(400, f"Layout must be one of {', '.join(LAYOUTS)}")
        job.layout = body.layout
    if body.model is not None:
        job.model = body.model.strip()
    job.status = "queued"
    job.deck_id = None
    job.step = "Waiting in queue"
    shutil.rmtree(_job_dir(job.id) / "crops", ignore_errors=True)
    db.commit()
    scanner.notify()
    return job_summary(job)


@router.delete("/{job_id}", status_code=204)
def delete_scan(job_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    job = _job(db, job_id, user)
    if job.status == "processing":
        raise HTTPException(409, "Wait for the scan to finish before deleting it")
    db.delete(job)
    db.commit()
    shutil.rmtree(_job_dir(job_id), ignore_errors=True)


@router.get("/{job_id}/debug")
def debug(job_id: int, download: bool = False, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Everything the model said for this scan, for the raw-data view and bug reports."""
    job = _job(db, job_id, user)
    data = {
        **job_summary(job),
        "images": [{
            "id": img.id, "filename": img.filename, "size": [img.width, img.height],
            "crops_found": img.crops_found, "photo_cards": img.photo_cards, "photo_error": img.photo_error,
            **(img.debug or {}),
            "detections": [{
                "id": d.id, "source": d.source, "read_as": d.raw_name, "qty": d.qty, "bbox": d.bbox,
                "set": d.raw_set, "collector_number": d.raw_number,
                "model_confidence": d.model_confidence, "card": d.card_id and db.get(Card, d.card_id).name,
                "match_score": d.match_score, "match_method": d.match_method, "raw": d.raw,
                "crop_url": f"/api/scans/{job.id}/crops/{d.id}" if d.crop_path else None,
            } for d in img.detections],
        } for img in job.images],
    }
    headers = {"Content-Disposition": f'attachment; filename="scan-{job.id}-raw.json"'} if download else None
    return JSONResponse(data, headers=headers)


@router.get("/{job_id}/photos/{image_id}")
def photo(job_id: int, image_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    job = _job(db, job_id, user)
    img = next((i for i in job.images if i.id == image_id), None)
    if img is None:
        raise HTTPException(404, "Photo not found")
    return FileResponse(img.path, media_type="image/jpeg")


@router.get("/{job_id}/crops/{detection_id}")
def crop(job_id: int, detection_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    job = _job(db, job_id, user)
    det = db.get(Detection, detection_id)
    if det is None or not det.crop_path or det.image_id not in {i.id for i in job.images}:
        raise HTTPException(404, "Crop not found")
    return FileResponse(det.crop_path, media_type="image/jpeg")


# --- review editing --------------------------------------------------------

def _editable(job: ScanJob) -> None:
    if job.status not in ("review", "saved"):
        raise HTTPException(409, "This scan isn't ready for review")


def _line(job: ScanJob, line_id: int) -> ScanLine:
    line = next((ln for ln in job.lines if ln.id == line_id), None)
    if line is None:
        raise HTTPException(404, "Line not found")
    return line


class LineIn(BaseModel):
    card_id: int | None = None
    qty: int | None = Field(default=None, ge=1, le=999)
    board: str | None = None
    accepted: bool | None = None


def _apply(db: Session, line: ScanLine, body: LineIn) -> None:
    if body.board is not None:
        if body.board not in BOARDS:
            raise HTTPException(400, f"Board must be one of {', '.join(sorted(BOARDS))}")
        line.board = body.board
    if body.card_id is not None and body.card_id != line.card_id:
        card = db.get(Card, body.card_id)
        if card is None:
            raise HTTPException(400, "Unknown card")
        line.card_id = card.id
        line.card = card
        line.flags = [f for f in line.flags if f not in ("unmatched", "low_confidence")]
        line.user_edited = True
    if body.qty is not None and body.qty != line.qty:
        line.qty = body.qty
        line.flags = [f for f in line.flags if f not in ("count_mismatch", "check_basic_count")]
        line.user_edited = True
    if body.accepted is not None:
        line.accepted = body.accepted
    elif line.user_edited:
        line.accepted = True


@router.patch("/{job_id}/lines/{line_id}")
def update_line(job_id: int, line_id: int, body: LineIn, db: Session = Depends(get_db),
                user: User = Depends(current_user)):
    job = _job(db, job_id, user)
    _editable(job)
    line = _line(job, line_id)
    _apply(db, line, body)
    db.commit()
    return job_detail(db, job)


@router.post("/{job_id}/lines", status_code=201)
def add_line(job_id: int, body: LineIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    job = _job(db, job_id, user)
    _editable(job)
    if body.card_id is None:
        raise HTTPException(400, "Pick a card")
    card = db.get(Card, body.card_id)
    if card is None:
        raise HTTPException(400, "Unknown card")
    board = body.board or "main"
    existing = next((ln for ln in job.lines if ln.card_id == card.id and ln.board == board), None)
    if existing:
        existing.qty += body.qty or 1
        existing.user_edited = existing.accepted = True
    else:
        job.lines.append(ScanLine(card_id=card.id, raw_name=card.name, qty=body.qty or 1, board=board,
                                  confidence=1.0, flags=[], suggestions=[], detection_ids=[],
                                  accepted=True, user_edited=True))
    db.commit()
    return job_detail(db, job)


class BoardIn(BaseModel):
    board: str


def _photo_qty(dets: list[Detection]) -> int:
    """Cards one photo contributes to a line, counted as when the lines were built."""
    dets = [d for d in dets if d.match_method != "unconfirmed"]
    return max(sum(d.qty for d in dets if d.source == "photo"), sum(1 for d in dets if d.source == "crop"), 1)


@router.put("/{job_id}/images/{image_id}/board")
def set_photo_board(job_id: int, image_id: int, body: BoardIn, db: Session = Depends(get_db),
                    user: User = Depends(current_user)):
    """Move everything found in one photo to a board (say, a photo of the sideboard). A line
    that also has cards from other photos is split, so only this photo's share moves."""
    job = _job(db, job_id, user)
    _editable(job)
    img = next((i for i in job.images if i.id == image_id), None)
    if img is None:
        raise HTTPException(404, "Photo not found")
    if body.board not in BOARDS:
        raise HTTPException(400, f"Board must be one of {', '.join(sorted(BOARDS))}")
    img.board = body.board
    here = {d.id: d for d in img.detections}

    def target(card_id: int | None, exclude: ScanLine) -> ScanLine | None:
        if card_id is None:
            return None  # unmatched lines are told apart only by what was read: keep them separate
        return next((ln for ln in job.lines if ln is not exclude and ln.card_id == card_id
                     and ln.board == body.board), None)

    for line in list(job.lines):
        mine = [i for i in line.detection_ids if i in here]
        if not mine or line.board == body.board:
            continue
        if len(mine) == len(line.detection_ids):
            into = target(line.card_id, line)
            if into is None:
                line.board = body.board
            else:
                into.qty += line.qty
                into.detection_ids = into.detection_ids + mine
                job.lines.remove(line)
            continue
        qty = min(_photo_qty([here[i] for i in mine]), line.qty - 1)
        if qty < 1:
            continue
        line.qty -= qty
        line.detection_ids = [i for i in line.detection_ids if i not in here]
        into = target(line.card_id, line)
        if into is None:
            job.lines.append(ScanLine(card_id=line.card_id, raw_name=line.raw_name, qty=qty, board=body.board,
                                      confidence=line.confidence, flags=list(line.flags),
                                      suggestions=line.suggestions, detection_ids=mine,
                                      accepted=line.accepted, user_edited=line.user_edited))
        else:
            into.qty += qty
            into.detection_ids = into.detection_ids + mine
    db.commit()
    return job_detail(db, job)


@router.delete("/{job_id}/lines/{line_id}")
def delete_line(job_id: int, line_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    job = _job(db, job_id, user)
    _editable(job)
    job.lines.remove(_line(job, line_id))
    db.commit()
    return job_detail(db, job)


@router.post("/{job_id}/accept-all")
def accept_all(job_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Accept every flagged line that has a matched card. Unmatched lines still need a card."""
    job = _job(db, job_id, user)
    _editable(job)
    for line in job.lines:
        if line.card_id is not None:
            line.accepted = True
    db.commit()
    return job_detail(db, job)


class SaveIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    format: str = Field(default="", max_length=40)
    notes: str = ""


@router.post("/{job_id}/save")
async def save(job_id: int, body: SaveIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    job = _job(db, job_id, user)
    _editable(job)
    if any(ln.card_id is None for ln in job.lines):
        raise HTTPException(400, "Some lines have no card. Pick a card for them or remove them.")
    if not job.lines:
        raise HTTPException(400, "The list is empty")
    totals: dict[tuple[int, str], int] = {}
    for ln in job.lines:
        totals[(ln.card_id, ln.board)] = totals.get((ln.card_id, ln.board), 0) + ln.qty
    deck = Deck(name=body.name.strip(), format=body.format, notes=body.notes, owner_id=user.id,
                group_id=user.group_id)
    deck.cards = [DeckCard(card_id=c, board=b, qty=q) for (c, b), q in totals.items()]
    db.add(deck)
    db.flush()
    job.deck_id = deck.id
    job.status = "saved"
    db.commit()
    cards = list(db.scalars(select(Card).where(Card.id.in_([c for c, _ in totals]))))
    await scryfall.fetch_details(db, cards)
    return {"deck_id": deck.id}
