"""Scryfall card-name sync and lazy card details.

Scryfall has no "changes since" feed, so the daily sync downloads the card-name
catalog (about 1 MB, every English card name) and inserts only names we have not
seen. Full card details are fetched on demand, 75 at a time, and cached.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import __version__, app_settings
from ..config import get_env
from ..db import session_scope
from ..models import Card
from . import matching

log = logging.getLogger(__name__)
HEADERS = {"User-Agent": f"MTGDeckKeeper/{__version__}", "Accept": "application/json"}

# Tests swap this for an httpx.MockTransport.
transport: httpx.AsyncBaseTransport | None = None


def _client(timeout: float = 60) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=get_env().scryfall_api, headers=HEADERS, timeout=timeout, transport=transport)


async def sync_card_names(db: Session) -> dict:
    async with _client() as client:
        r = await client.get("/catalog/card-names")
        r.raise_for_status()
        names: list[str] = r.json()["data"]
    existing = set(db.scalars(select(Card.name)))
    new = [n for n in dict.fromkeys(names) if n not in existing]
    db.add_all(Card(name=n) for n in new)
    db.commit()
    result = {
        "synced_at": datetime.now(UTC).isoformat(),
        "total": len(existing) + len(new),
        "added": len(new),
        "added_sample": new[:20],
    }
    app_settings.set_state(db, "scryfall_sync", result)
    matching.rebuild(db)
    log.info("Card sync: %d new names, %d total", len(new), result["total"])
    return result


def _apply_details(card: Card, data: dict) -> None:
    faces = data.get("card_faces") or []
    images = data.get("image_uris") or (faces[0].get("image_uris") if faces else None) or {}
    card.oracle_id = data.get("oracle_id")
    card.type_line = data.get("type_line") or (faces[0].get("type_line") if faces else None)
    card.mana_cost = data.get("mana_cost") or (faces[0].get("mana_cost") if faces else None)
    card.cmc = data.get("cmc")
    colors = data.get("colors") if "colors" in data else (faces[0].get("colors") if faces else [])
    card.colors = "".join(colors or [])
    card.image_uri = images.get("normal")
    card.scryfall_uri = data.get("scryfall_uri")
    card.details_fetched_at = datetime.now(UTC)


async def fetch_details(db: Session, cards: list[Card]) -> int:
    """Fill in type, cost and image for cards that don't have them yet. Never raises."""
    todo = [c for c in cards if c.details_fetched_at is None]
    if not todo:
        return 0
    by_name = {c.name.lower(): c for c in todo}
    filled = 0
    try:
        async with _client(timeout=30) as client:
            for i in range(0, len(todo), 75):
                chunk = todo[i:i + 75]
                r = await client.post("/cards/collection", json={"identifiers": [{"name": c.name} for c in chunk]})
                r.raise_for_status()
                for data in r.json().get("data", []):
                    card = by_name.get(data.get("name", "").lower())
                    if card is None:
                        continue
                    _apply_details(card, data)
                    filled += 1
                if i + 75 < len(todo):
                    await asyncio.sleep(0.1)  # Scryfall asks for <=10 requests per second
        db.commit()
    except (httpx.HTTPError, ValueError) as e:
        log.warning("Could not fetch card details from Scryfall: %s", e)
        db.rollback()
    return filled


def _due(db: Session) -> bool:
    state = app_settings.get_state(db, "scryfall_sync")
    if not state:
        return True
    last = datetime.fromisoformat(state["synced_at"])
    hour = app_settings.get(db, "scryfall.sync_hour")
    now = datetime.now().astimezone()
    if now - last < timedelta(hours=20):
        return False
    return now.hour >= hour


async def sync_loop() -> None:
    """Runs for the life of the app. First run seeds the database immediately."""
    while True:
        try:
            with session_scope() as db:
                never = not app_settings.get_state(db, "scryfall_sync")
                if never or (app_settings.get(db, "scryfall.auto_sync") and _due(db)):
                    await sync_card_names(db)
        except Exception as e:  # keep the loop alive through network outages
            log.warning("Scheduled card sync failed: %s", e)
        await asyncio.sleep(900)
