"""How long a scan should take, learned from this server's own finished scans.

The time goes almost all to the vision model, and grows with the number of photos (each is
outlined, straightened and read) and the number of cards in them (each title is read on its own).
So seconds ~ per_photo x photos + per_card x cards, fitted to the last scans' recorded stage
timings, which reflects this server's model and GPU. Before there are enough scans, defaults
measured on a 9B model with three parallel slots are used.
"""

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..models import ScanJob

DEFAULT = (20.0, 1.6)  # seconds per photo, per card
HISTORY = 20  # this many recent scans are learned from
MIN_HISTORY = 3
CARDS_UNKNOWN = 60  # cards a scan holds when its deck type doesn't say


def _seconds(job: ScanJob) -> float | None:
    """How long a finished scan took to read: its photos' recorded stage timings."""
    total = 0.0
    for img in job.images:
        timings = (img.debug or {}).get("timings")
        if not timings:
            return None
        total += sum(t.get("seconds") or 0 for t in timings)
    return total or None


def history(db: Session) -> list[tuple[int, int, float]]:
    """(photos, cards, seconds) of the most recent finished scans."""
    jobs = db.scalars(select(ScanJob).where(ScanJob.status.in_(("review", "saved")))
                      .options(selectinload(ScanJob.images), selectinload(ScanJob.lines))
                      .order_by(ScanJob.finished_at.desc().nullslast()).limit(HISTORY)).all()
    out = []
    for job in jobs:
        seconds = _seconds(job)
        cards = sum(line.qty for line in job.lines)
        if seconds and cards and job.images:
            out.append((len(job.images), cards, seconds))
    return out


def rates(db: Session) -> tuple[float, float, int]:
    """(seconds per photo, seconds per card, how many scans that comes from)."""
    rows = history(db)
    if len(rows) < MIN_HISTORY:
        return (*DEFAULT, len(rows))
    a = np.array([[p, c] for p, c, _ in rows], float)
    y = np.array([s for _, _, s in rows], float)
    (per_photo, per_card), *_ = np.linalg.lstsq(a, y, rcond=None)
    if per_photo < 0 or per_card <= 0:  # too few or too alike scans to tell the two apart
        per_photo, per_card = 0.0, float(y.sum() / a[:, 1].sum())
    return float(per_photo), float(per_card), len(rows)


def cards_for(deck_type: str) -> int:
    from .scanner import DECK_TYPES

    size = DECK_TYPES.get(deck_type or "other", {})
    return sum(size.get(k) or 0 for k in ("main", "side", "commander")) or CARDS_UNKNOWN


def seconds_for(db: Session, photos: int, deck_type: str) -> dict:
    per_photo, per_card, based_on = rates(db)
    cards = cards_for(deck_type)
    return {"seconds": round(per_photo * photos + per_card * cards), "cards": cards, "based_on": based_on}
