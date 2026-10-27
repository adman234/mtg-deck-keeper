"""Fuzzy-match names read by the vision model against the local card list."""

import re
import threading
import unicodedata
from dataclasses import dataclass

from rapidfuzz import fuzz, process
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Card

MIN_SCORE = 60.0
PREFIX_SCORE = 88.0  # a read that is the first words of one card's name, and nothing closer
PREFIX_MIN = 0.55  # ...and at least this much of it
TITLE_MIN = 8  # a legendary card's title after the comma, this long, that only one card has

_lock = threading.Lock()
_index: "CardIndex | None" = None


_LIGATURES = str.maketrans({"æ": "ae", "Æ": "Ae", "œ": "oe", "Œ": "Oe", "’": "'", "‘": "'"})


def normalize(name: str) -> str:
    name = name.translate(_LIGATURES)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = name.lower()
    name = re.sub(r"[^a-z0-9' /]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


@dataclass
class Match:
    card_id: int
    name: str
    score: float


class CardIndex:
    def __init__(self, cards: list[tuple[int, str]]):
        self.keys: list[str] = []
        self.targets: list[tuple[int, str]] = []
        self.exact: dict[str, tuple[int, str]] = {}
        self.titles: dict[str, set[tuple[int, str]]] = {}  # "lost but seeking": {Azusa, Lost but Seeking}
        for card_id, name in cards:
            # Index the full name and each face of split / double-faced cards,
            # since the model usually reads only the face it can see.
            variants = {name} | ({part.strip() for part in name.split("//")} if "//" in name else set())
            if "," in name and "//" not in name:
                self.titles.setdefault(normalize(name.split(",", 1)[1]), set()).add((card_id, name))
            for v in variants:
                key = normalize(v)
                if not key:
                    continue
                self.keys.append(key)
                self.targets.append((card_id, name))
                self.exact.setdefault(key, (card_id, name))

    def __len__(self) -> int:
        return len({t[0] for t in self.targets})

    def match(self, raw: str, limit: int = 4) -> list[Match]:
        key = normalize(raw)
        if not key or not self.keys:
            return []
        results: list[Match] = []
        if key in self.exact:
            card_id, name = self.exact[key]
            results.append(Match(card_id, name, 100.0))
        for _, score, idx in process.extract(key, self.keys, scorer=fuzz.ratio, limit=limit * 3):
            card_id, name = self.targets[idx]
            if all(m.card_id != card_id for m in results):
                results.append(Match(card_id, name, round(float(score), 1)))
        if (not results or results[0].score < PREFIX_SCORE) and "," in raw:
            # A legendary card's own name misread but its title after the comma read right (scan
            # 16: "Elspeth, Lost but Seeking" for Azusa, Lost but Seeking): enough when only one
            # card has that title.
            title = normalize(raw.split(",", 1)[1])
            same = self.titles.get(title, set())
            if len(title) >= TITLE_MIN and len(same) == 1:
                card_id, name = next(iter(same))
                results = [Match(card_id, name, PREFIX_SCORE)] + [m for m in results if m.card_id != card_id]
        if not results or results[0].score < PREFIX_SCORE:
            # The start of a name, cut off by something lying over the card (scan 18: "Pia and
            # Kiran" for Pia and Kiran Nalaar): good enough when only one card's name starts so.
            longer = {self.targets[i] for i, k in enumerate(self.keys) if k.startswith(key + " ")}
            if len(longer) == 1 and len(key) >= PREFIX_MIN * len(normalize(next(iter(longer))[1])):
                card_id, name = next(iter(longer))
                results = [Match(card_id, name, PREFIX_SCORE)] + [m for m in results if m.card_id != card_id]
        return results[:limit]

    def search(self, query: str, limit: int = 10) -> list[Match]:
        """Typeahead search: prefix and substring hits first, then fuzzy."""
        key = normalize(query)
        if not key:
            return []
        seen: set[int] = set()
        out: list[Match] = []
        for i, k in enumerate(self.keys):
            if k.startswith(key) or (len(key) >= 3 and key in k):
                card_id, name = self.targets[i]
                if card_id not in seen:
                    seen.add(card_id)
                    out.append(Match(card_id, name, 100.0 if k.startswith(key) else 90.0))
        out.sort(key=lambda m: (-m.score, len(m.name), m.name))
        if len(out) < limit:
            for _, score, idx in process.extract(key, self.keys, scorer=fuzz.WRatio, limit=limit * 2):
                card_id, name = self.targets[idx]
                if card_id not in seen and score >= 50:
                    seen.add(card_id)
                    out.append(Match(card_id, name, round(float(score), 1)))
        return out[:limit]


def rebuild(db: Session) -> CardIndex:
    global _index
    idx = CardIndex([(r.id, r.name) for r in db.execute(select(Card.id, Card.name))])
    with _lock:
        _index = idx
    return idx


def get_index(db: Session) -> CardIndex:
    if _index is None:
        return rebuild(db)
    return _index
