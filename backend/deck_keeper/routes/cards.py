from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import current_user
from ..models import Card
from ..services import matching

router = APIRouter(prefix="/api/cards", tags=["cards"], dependencies=[Depends(current_user)])


def card_out(c: Card | None) -> dict | None:
    if c is None:
        return None
    return {"id": c.id, "name": c.name, "type_line": c.type_line, "mana_cost": c.mana_cost, "cmc": c.cmc,
            "colors": c.colors, "image_uri": c.image_uri, "scryfall_uri": c.scryfall_uri}


@router.get("/search")
def search(q: str = Query(min_length=1, max_length=100), limit: int = Query(10, le=50),
           db: Session = Depends(get_db)):
    return [{"id": m.card_id, "name": m.name, "score": m.score}
            for m in matching.get_index(db).search(q, limit)]


@router.get("/{card_id}")
def get_card(card_id: int, db: Session = Depends(get_db)):
    card = db.get(Card, card_id)
    if card is None:
        raise HTTPException(404, "Card not found")
    return card_out(card)
