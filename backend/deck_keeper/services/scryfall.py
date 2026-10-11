"""Scryfall card-name sync and lazy card details.

Scryfall has no "changes since" feed, so the daily sync downloads the card-name
catalog (about 1 MB, every English card name) and inserts only names we have not
seen. Full card details are fetched on demand, 75 at a time, and cached.
"""

import asyncio
import logging
import re
import unicodedata
from datetime import UTC, datetime, timedelta

import httpx
from rapidfuzz import fuzz
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import __version__, app_settings
from ..config import get_env
from ..db import session_scope
from ..models import Card, Lookup
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


# --- lookups used to identify foreign, showcase and hard-to-read cards ---------------------

NOT_FOUND_RETRY = timedelta(days=7)
_gate = asyncio.Lock()
_last_call = 0.0


async def _get(path: str, params: dict | None = None) -> httpx.Response:
    """Rate-limited GET (Scryfall asks for at most ~10 requests per second)."""
    global _last_call
    async with _gate:
        wait = 0.1 - (asyncio.get_running_loop().time() - _last_call)
        if wait > 0:
            await asyncio.sleep(wait)
        try:
            async with _client(timeout=15) as client:
                return await client.get(path, params=params)
        finally:
            _last_call = asyncio.get_running_loop().time()


def _cached(db: Session, kind: str, key: str) -> tuple[bool, dict | None]:
    row = db.scalar(select(Lookup).where(Lookup.kind == kind, Lookup.key == key))
    if row is None:
        return False, None
    fetched = row.fetched_at if row.fetched_at.tzinfo else row.fetched_at.replace(tzinfo=UTC)
    if row.result is None and datetime.now(UTC) - fetched > NOT_FOUND_RETRY:
        return False, None  # retry misses after a while; new sets get added
    return True, row.result


def _store(db: Session, kind: str, key: str, result: dict | None) -> None:
    row = db.scalar(select(Lookup).where(Lookup.kind == kind, Lookup.key == key))
    if row is None:
        db.add(Lookup(kind=kind, key=key, result=result))
    else:
        row.result, row.fetched_at = result, datetime.now(UTC)
    db.commit()


def _names(card: dict) -> list[str]:
    """Every name a printing could show: English, printed (foreign) and flavor names, per face."""
    out = [card.get("name"), card.get("printed_name"), card.get("flavor_name")]
    for face in card.get("card_faces") or []:
        out += [face.get("name"), face.get("printed_name"), face.get("flavor_name")]
    return [n for n in out if n]


def _summary(card: dict) -> dict:
    return {"name": card["name"], "lang": card.get("lang", "en"), "names": _names(card)}


def clean_printing(set_code: str | None, number: str | None) -> tuple[str, str] | None:
    """'MID', '154/277' -> ('mid', '154'). Returns None when either part is unusable."""
    if not set_code or not number:
        return None
    s = re.search(r"[A-Za-z0-9]{2,6}", set_code)
    n = re.search(r"(\d+)([a-zA-Z★†]?)", number)
    if not s or not n:
        return None
    return s.group(0).lower(), f"{int(n.group(1))}{n.group(2).lower()}"


async def lookup_printing(db: Session, set_code: str | None, number: str | None) -> dict | None:
    """The card at a set code + collector number. Never raises."""
    parts = clean_printing(set_code, number)
    if parts is None:
        return None
    key = "/".join(parts)
    hit, result = _cached(db, "printing", key)
    if hit:
        return result
    try:
        r = await _get(f"/cards/{parts[0]}/{parts[1]}")
    except httpx.HTTPError as e:
        log.info("Printing lookup %s failed: %s", key, e)
        return None
    if r.status_code == 404:
        _store(db, "printing", key, None)
        return None
    if r.status_code != 200:
        return None
    result = _summary(r.json())
    _store(db, "printing", key, result)
    return result


def loose(text: str) -> str:
    """Normalise for comparing names in any script: case-folded, no spaces or punctuation."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in "LN")


def _foreign_queries(raw_name: str) -> list[str]:
    """Exact phrase first, then looser searches that survive a misread letter or accent."""
    text = re.sub(r'["()]', " ", raw_name).strip()
    words = [w for w in re.split(r"\s+", text) if w]
    queries = [f'"{text}"']
    if len(words) > 1:
        queries.append(" ".join(words))  # every word, any order
        # One distinctive word at a time, so a single misread word can't sink the search.
        distinct = list(dict.fromkeys(w for w in words if len(w) >= 4))
        queries += sorted(distinct, key=len, reverse=True)[:2]  # stable: ties keep word order
    return queries


async def lookup_foreign(db: Session, raw_name: str) -> dict | None:
    """Find a card by its printed name in any language (or a flavor name). Never raises."""
    key = loose(raw_name)
    if len(key) < 2:
        return None
    hit, result = _cached(db, "foreign", key)
    if hit:
        return result
    result = None
    for query in _foreign_queries(raw_name):
        try:
            r = await _get("/cards/search", {"q": f"{query} lang:any", "include_multilingual": "true",
                                             "unique": "prints"})
        except httpx.HTTPError as e:
            log.info("Foreign lookup %r failed: %s", raw_name, e)
            return None  # not cached: try again next scan
        if r.status_code == 404:
            continue
        if r.status_code != 200:
            return None
        best, best_score = None, 0.0
        for card in r.json().get("data", [])[:100]:
            for name in _names(card):
                score = fuzz.ratio(key, loose(name))
                if score > best_score:
                    best, best_score = card, score
        if best is not None and best_score >= 80:
            result = {**_summary(best), "score": round(best_score, 1)}
            break
    _store(db, "foreign", key, result)
    return result
