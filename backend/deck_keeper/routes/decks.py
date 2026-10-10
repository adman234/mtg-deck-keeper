from collections import Counter

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..db import get_db
from ..deps import current_user
from ..models import Card, Deck, DeckCard, ScanJob, User
from ..services import exports, scryfall
from .cards import card_out
from .scans import BOARDS

router = APIRouter(prefix="/api/decks", tags=["decks"])


def _deck(db: Session, deck_id: int, user: User) -> Deck:
    deck = db.get(Deck, deck_id)
    if deck is None or (deck.owner_id != user.id and not user.is_admin):
        raise HTTPException(404, "Deck not found")
    return deck


def counts(deck: Deck) -> dict[str, int]:
    c: Counter[str] = Counter()
    for dc in deck.cards:
        c[dc.board] += dc.qty
    return {b: c.get(b, 0) for b in ("commander", "main", "side", "maybe")}


def deck_summary(deck: Deck) -> dict:
    cover = next((dc.card.image_uri for dc in sorted(deck.cards, key=lambda d: -d.qty) if dc.card.image_uri), None)
    return {"id": deck.id, "name": deck.name, "format": deck.format, "notes": deck.notes,
            "counts": counts(deck), "cover_image": cover,
            "created_at": deck.created_at.isoformat(), "updated_at": deck.updated_at.isoformat()}


@router.get("")
def list_decks(db: Session = Depends(get_db), user: User = Depends(current_user)):
    decks = db.scalars(select(Deck).where(Deck.owner_id == user.id)
                       .options(selectinload(Deck.cards).selectinload(DeckCard.card))
                       .order_by(Deck.updated_at.desc()))
    return [deck_summary(d) for d in decks]


class DeckIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    format: str = Field(default="", max_length=40)
    notes: str = ""


@router.post("", status_code=201)
def create_deck(body: DeckIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    deck = Deck(name=body.name.strip(), format=body.format, notes=body.notes, owner_id=user.id,
                group_id=user.group_id)
    db.add(deck)
    db.commit()
    return deck_summary(deck)


@router.get("/{deck_id}")
async def get_deck(deck_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    deck = _deck(db, deck_id, user)
    await scryfall.fetch_details(db, [dc.card for dc in deck.cards])
    scans = db.scalars(select(ScanJob).where(ScanJob.deck_id == deck.id))
    return {
        **deck_summary(deck),
        "cards": [{"card": card_out(dc.card), "qty": dc.qty, "board": dc.board}
                  for dc in sorted(deck.cards, key=lambda d: d.card.name)],
        "scans": [{"id": s.id, "name": s.name, "photos": [f"/api/scans/{s.id}/photos/{i.id}" for i in s.images]}
                  for s in scans],
    }


class DeckPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    format: str | None = Field(default=None, max_length=40)
    notes: str | None = None


@router.patch("/{deck_id}")
def update_deck(deck_id: int, body: DeckPatch, db: Session = Depends(get_db), user: User = Depends(current_user)):
    deck = _deck(db, deck_id, user)
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(deck, field, value.strip() if isinstance(value, str) and field == "name" else value)
    db.commit()
    return deck_summary(deck)


class DeckCardIn(BaseModel):
    card_id: int
    qty: int = Field(ge=0, le=999)
    board: str = "main"


@router.put("/{deck_id}/cards")
async def replace_cards(deck_id: int, body: list[DeckCardIn], db: Session = Depends(get_db),
                        user: User = Depends(current_user)):
    deck = _deck(db, deck_id, user)
    totals: dict[tuple[int, str], int] = {}
    for item in body:
        if item.board not in BOARDS:
            raise HTTPException(400, f"Unknown board {item.board}")
        if item.qty:
            totals[(item.card_id, item.board)] = totals.get((item.card_id, item.board), 0) + item.qty
    known = set(db.scalars(select(Card.id).where(Card.id.in_([c for c, _ in totals]))))
    if missing := {c for c, _ in totals} - known:
        raise HTTPException(400, f"Unknown card ids: {sorted(missing)}")
    deck.cards.clear()
    db.flush()
    deck.cards.extend(DeckCard(card_id=c, board=b, qty=q) for (c, b), q in totals.items())
    db.commit()
    db.refresh(deck)
    return await get_deck(deck_id, db, user)


@router.delete("/{deck_id}", status_code=204)
def delete_deck(deck_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    db.delete(_deck(db, deck_id, user))
    db.commit()


@router.get("/{deck_id}/export", response_class=PlainTextResponse)
def export(deck_id: int, format: str = Query("text"), db: Session = Depends(get_db),
           user: User = Depends(current_user)):
    fn = exports.FORMATS.get(format)
    if fn is None:
        raise HTTPException(400, f"Format must be one of {', '.join(exports.FORMATS)}")
    return fn(_deck(db, deck_id, user))
