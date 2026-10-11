"""Turn what the model read into a card, trying progressively harder methods.

1. Fuzzy match against the English card list (instant, offline).
2. Set code + collector number -> exact printing on Scryfall (any language, any frame).
3. Search Scryfall for the name in any language, and alternate (flavor) names.
4. For single-card crops that are still unclear, show the model the card again with the
   closest candidates and let it choose.
Steps 2 and 3 are cached and skipped quietly when Scryfall can't be reached.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz import fuzz
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Card
from . import matching, scryfall, vision

log = logging.getLogger(__name__)

GOOD = 90.0  # a name match at least this good is trusted without further lookups
REASK_BELOW = 85.0
REASK_FLOOR = 65.0  # below this the read is probably not a real title at all
REASK_MARGIN = 10.0  # only candidates this close to the best match are offered


@dataclass
class Resolution:
    card_id: int | None = None
    score: float = 0.0
    method: str | None = None
    confidence: float | None = None  # replaces the model's confidence when set (re-ask)
    candidates: list[str] = field(default_factory=list)


def has_non_latin(text: str) -> bool:
    return any(ord(ch) > 0x24F and ch.isalpha() for ch in text)


def _card_id(db: Session, index: matching.CardIndex, name: str) -> int | None:
    hits = index.match(name, limit=1)
    if hits and hits[0].score == 100:
        return hits[0].card_id
    card = db.scalar(select(Card).where(Card.name == name))
    if card is None:  # printed after our last name sync
        card = Card(name=name)
        db.add(card)
        db.commit()
        matching.rebuild(db)
    return card.id


@dataclass
class Options:
    collector_lookup: bool = True
    foreign_lookup: bool = True
    candidate_reask: bool = True

    @classmethod
    def from_settings(cls, s: dict) -> "Options":
        return cls(s["collector_lookup"], s["foreign_lookup"], s["candidate_reask"])


async def resolve(db: Session, index: matching.CardIndex, read: vision.ReadCard, crop_path: str | None,
                  cfg: vision.VisionConfig, opts: Options, ask) -> Resolution:
    """``ask`` runs a coroutine under the scan's model concurrency limit."""
    hits = index.match(read.name, limit=5)
    best = hits[0] if hits and hits[0].score >= matching.MIN_SCORE else None
    res = Resolution(best.card_id, best.score, "name") if best else Resolution()
    if res.score >= GOOD:
        return res
    candidates = [h.name for h in hits if h.score >= 50]
    raw_loose = scryfall.loose(read.name)
    foreign_script = has_non_latin(read.name)

    if opts.collector_lookup:
        printing = await scryfall.lookup_printing(db, read.set, read.collector_number)
        if printing:
            # A misread number points at a random card, so the name must agree unless it's in a
            # script we can't compare (Japanese, Russian, ...).
            agree = max((fuzz.ratio(raw_loose, scryfall.loose(n)) for n in printing["names"]), default=0)
            if foreign_script or agree >= 60:
                return Resolution(_card_id(db, index, printing["name"]), 95.0, "printing")
            candidates.insert(0, printing["name"])

    if opts.foreign_lookup and (foreign_script or res.score < GOOD):
        found = await scryfall.lookup_foreign(db, read.name)
        if found and found["score"] > res.score:
            lang = found.get("lang", "en")
            method = "alias" if lang == "en" else f"foreign:{lang}"[:20]
            return Resolution(_card_id(db, index, found["name"]), found["score"], method)

    # Re-ask only breaks near-ties between real candidates. It must not overrule a clearly better
    # name match, and its pick keeps the candidate's own score so the line stays flagged.
    close = [h for h in hits if best and h.score >= best.score - REASK_MARGIN]
    if opts.candidate_reask and crop_path and best and REASK_FLOOR <= best.score < REASK_BELOW and len(close) > 1:
        try:
            jpeg = Path(crop_path).read_bytes()
        except OSError:
            jpeg = None
        if jpeg:
            choice = await ask(vision.choose_card(cfg, jpeg, [h.name for h in close]))
            if choice:
                name, conf = choice
                picked = next(h for h in close if h.name == name)
                return Resolution(picked.card_id, picked.score, "choice", confidence=min(conf, read.confidence))

    res.candidates = candidates
    return res
