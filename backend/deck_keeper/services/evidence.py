"""Doubts about a read, which lower its confidence a little, and checks that flag it outright.

Each doubt multiplies the line's confidence by its factor, so one small doubt leaves a line
alone while several push it under the review threshold. A type line that contradicts the
matched card flags the line on its own: a confidently wrong real name usually comes from rules
text or a neighbouring card, and then the type rarely matches.
"""

import re

DOUBTS = {
    "type_mismatch": 0.5,  # the type line read doesn't fit the card (flags the line too)
    "type_unchecked": 0.95,  # no type line could be read (covered, or not in English)
    "reads_disagree": 0.9,  # the whole-table read named a different real card
    "model_chose": 0.9,  # the model was asked to pick between candidates
    "title_retry": 0.92,  # the first close-up had no title in it
}

TYPES = {"artifact", "battle", "creature", "enchantment", "instant", "kindred", "land", "planeswalker", "sorcery"}


def card_types(type_line: str | None) -> set[str]:
    """The card types named in a type line ("Legendary Creature — Elf" -> {"creature"})."""
    words = re.findall(r"[a-z]+", (type_line or "").lower())
    return {"kindred" if w == "tribal" else w for w in words} & TYPES


def type_check(read: str | None, card_type_line: str | None) -> str | None:
    """"ok", "type_mismatch", "type_unchecked", or None when the card's own type isn't known.
    Only English type words are compared; anything else counts as unchecked."""
    if not card_type_line:
        return None
    seen = card_types(read)
    if not seen:
        return "type_unchecked"
    return "ok" if seen & card_types(card_type_line) else "type_mismatch"


def factor(doubts: list[str]) -> float:
    out = 1.0
    for d in doubts:
        out *= DOUBTS.get(d, 1.0)
    return out
