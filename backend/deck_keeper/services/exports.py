"""Decklist text formats."""

from ..models import Deck

BOARD_ORDER = ["commander", "main", "side", "maybe"]


def _by_board(deck: Deck) -> dict[str, list[tuple[int, str]]]:
    boards: dict[str, list[tuple[int, str]]] = {b: [] for b in BOARD_ORDER}
    for dc in deck.cards:
        boards.setdefault(dc.board, []).append((dc.qty, dc.card.name))
    for rows in boards.values():
        rows.sort(key=lambda r: r[1])
    return boards


def to_text(deck: Deck) -> str:
    """Plain list, understood by Moxfield, Archidekt, MTGGoldfish and most tools."""
    b = _by_board(deck)
    parts = []
    if b["commander"]:
        parts.append("\n".join(f"{q} {n}" for q, n in b["commander"]))
    parts.append("\n".join(f"{q} {n}" for q, n in b["main"]))
    if b["side"]:
        parts.append("Sideboard\n" + "\n".join(f"{q} {n}" for q, n in b["side"]))
    return "\n\n".join(p for p in parts if p) + "\n"


def _arena_name(name: str) -> str:
    # Arena lists double-faced and split cards by front face only, e.g. "Delver of Secrets".
    return name.split(" // ")[0] if " // " in name else name


def to_arena(deck: Deck) -> str:
    b = _by_board(deck)
    parts = []
    if b["commander"]:
        parts.append("Commander\n" + "\n".join(f"{q} {_arena_name(n)}" for q, n in b["commander"]))
    parts.append("Deck\n" + "\n".join(f"{q} {_arena_name(n)}" for q, n in b["main"]))
    if b["side"]:
        parts.append("Sideboard\n" + "\n".join(f"{q} {_arena_name(n)}" for q, n in b["side"]))
    return "\n\n".join(parts) + "\n"


FORMATS = {"text": to_text, "arena": to_arena}
