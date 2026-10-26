"""Read a pasted decklist: Arena, MTGO, Moxfield and plain "4 Lightning Bolt" lists."""

import re
from dataclasses import dataclass

from . import matching

SECTIONS = {"deck": "main", "main": "main", "mainboard": "main", "main deck": "main", "sideboard": "side",
            "side": "side", "commander": "commander", "commanders": "commander", "companion": "side",
            "maybeboard": "maybe", "maybe": "maybe", "considering": "maybe"}
LINE = re.compile(r"^\s*(?:(\d+)\s*x?\s+)?(.+?)\s*$", re.IGNORECASE)
# Trailing set and collector number: "(M10) 146", "[M10]", "(MH2) 179 *F*".
PRINTING = re.compile(r"\s+[(\[][A-Za-z0-9]{2,6}[)\]](?:\s+\S+)?(?:\s+\*\w+\*)?\s*$")
FUZZY_OK = 90.0


@dataclass
class Entry:
    qty: int
    name: str  # as written
    board: str
    card_id: int | None = None
    card_name: str | None = None


def parse(text: str, index: matching.CardIndex) -> list[Entry]:
    out: list[Entry] = []
    board = "main"
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue  # blank lines often just separate creatures, spells and lands
        if line.startswith(("#", "//")) or line.endswith(":"):
            key = line.lstrip("#/ ").rstrip(":").strip().lower()
            board = SECTIONS.get(key, board)
            continue
        if line.lower() in SECTIONS:
            board = SECTIONS[line.lower()]
            continue
        entry_board = board
        if line.upper().startswith("SB:"):  # MTGO
            entry_board, line = "side", line[3:].strip()
        m = LINE.match(line)
        if not m:
            continue
        qty, name = int(m.group(1) or 1), PRINTING.sub("", m.group(2)).strip()
        if not name:
            continue
        entry = Entry(qty, name, entry_board)
        hits = index.match(name, limit=1)
        if hits and hits[0].score >= FUZZY_OK:
            entry.card_id, entry.card_name = hits[0].card_id, hits[0].name
        out.append(entry)
    return out


def score(expected: list[dict], lines: list[dict]) -> dict:
    """Compare a scan's lines with the real list, by card name (boards aside).
    expected: [{"name", "card_name"?, "qty"}]; lines: [{"name", "qty"}] where name is the matched
    card's, or what was read for an unmatched line. Returns right/total and what was missing or
    extra."""
    want: dict[str, list] = {}
    for e in expected:
        name = e.get("card_name") or e["name"]
        want.setdefault(matching.normalize(name), [name, 0])[1] += e["qty"]
    # A two-faced card in the list counts when either face's name was found.
    faces = {matching.normalize(f): k for k, (n, _) in want.items() if "//" in n for f in n.split("//")}
    got: dict[str, list] = {}
    for ln in lines:
        key = matching.normalize(ln["name"])
        key = key if key in want else faces.get(key, key)
        got.setdefault(key, [ln["name"], 0])[1] += ln["qty"]
    right = sum(min(q, got.get(k, [None, 0])[1]) for k, (_, q) in want.items())
    missing = sorted(([n, q - got.get(k, [None, 0])[1]] for k, (n, q) in want.items()
                      if q > got.get(k, [None, 0])[1]), key=lambda x: x[0])
    extra = sorted(([n, q - want.get(k, [None, 0])[1]] for k, (n, q) in got.items()
                    if q > want.get(k, [None, 0])[1]), key=lambda x: x[0])
    return {"right": right, "total": sum(q for _, q in want.values()), "missing": missing, "extra": extra}
