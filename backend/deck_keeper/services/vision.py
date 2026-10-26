"""Client for any OpenAI-compatible vision endpoint (llama-swap, Ollama, LM Studio, vLLM, OpenAI)."""

import base64
import json
import logging
import re
from dataclasses import dataclass, field

import httpx

log = logging.getLogger(__name__)

# Tests swap this for an httpx.MockTransport.
transport: httpx.AsyncBaseTransport | None = None

CARD_SCHEMA = {
    "type": "object",
    "properties": {
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "count": {"type": "integer", "minimum": 1},
                    "set": {"type": "string"},
                    "collector_number": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                },
                "required": ["name", "count", "confidence"],
            },
        }
    },
    "required": ["cards"],
}

TITLE_HINT = (
    "A card's name is the title on its TOP line, beside the mana cost. Never use the type line in the "
    "middle of the card (Creature, Sorcery, Instant, Land, Conjuro, Criatura, Rituel, Hexerei, ...) "
    "or any rules text as the name."
)

SYSTEM_PROMPT = (
    "You identify Magic: The Gathering cards in photos. Read each card's title line exactly as printed. "
    + TITLE_HINT + " "
    "Use the English card name when you recognise the card; otherwise copy the printed title exactly, "
    "in its own language and script. Never invent a name you cannot read. confidence is your "
    "probability (0-1) that the name is correct. Reply with JSON only."
)

BOX_HINT = (
    '"box" is where it is in the photo: [x1, y1, x2, y2] as numbers from 0 to 1000 across the '
    "image width and height (top-left to bottom-right). For a pile the box must cover the WHOLE pile, "
    "including the edges of every card behind the front one, not just the front card."
)
FOUR_HINT = "Only if you truly can't tell 4 from 5, answer 4."
COUNT_HINT = (
    "Count cards, not edges: a card in a sleeve shows two edges (sleeve and card)."
)

# One prompt for every layout: decks are often a mix of single cards and piles of copies.
TABLE_PROMPT = (
    "These Magic cards are laid out on a table. Some may lie on their own and some may be in piles: "
    "copies of one card overlapped or fanned so the front card's title is fully visible and the cards "
    "behind show only an edge or part of their title bar. The cards in a pile can be other printings "
    "or languages of the same card. Make one entry for every separate card and every pile, even when "
    "the same card appears several times side by side. For each entry give the name of the front "
    "card and count = how many cards are in it (1 for a card on its own). " + COUNT_HINT + " "
    + FOUR_HINT + " Never read or guess names from the slivers of cards behind the front one. "
    + BOX_HINT + ' Format: {"cards": [{"name": ..., "count": ..., "confidence": ..., "box": [x1, y1, x2, y2]}]}'
)

# Kept for scans saved with a layout; every layout now uses the same prompt.
LAYOUTS = {"auto": TABLE_PROMPT, "piles": TABLE_PROMPT, "spread": TABLE_PROMPT}

ZOOM_PROMPT = (
    "This is a close-up of one spot on a table of Magic cards: a single card or a pile of copies of "
    "one card, overlapped so the front card is on top. Read the title of the FRONT card in the middle "
    "of the image (English name if you recognise it, otherwise as printed). " + TITLE_HINT + " Count the cards in "
    "this one pile, including the front card. " + COUNT_HINT + " Only count cards stacked under the "
    "front card; cards lying beside it belong to other piles, so ignore them and anything cut off at "
    "the edges. " + FOUR_HINT + " If no title is visible, return an empty name. "
    'Format: {"name": ..., "count": ..., "confidence": ...}'
)
ZOOM_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "count": {"type": "integer", "minimum": 0},
                   "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
    "required": ["name", "count", "confidence"],
}

CROP_PROMPT = (
    "This image is a single Magic card. Return exactly one entry with count 1. If you can't see a "
    'title line (for example only artwork is visible) or can\'t read it, return {"cards": []}. Newer '
    "cards print a set code (3-5 letters, e.g. MID) and collector number (e.g. 154/277) in small text "
    "at the bottom-left; include them if legible, they identify the card in any language. "
    'Format: {"cards": [{"name": ..., "count": 1, "set": ..., "collector_number": ..., "confidence": ...}]}'
)


class VisionError(Exception):
    pass


@dataclass
class VisionConfig:
    base_url: str
    model: str
    api_key: str = ""
    timeout: float = 300
    disable_thinking: bool = True

    @classmethod
    def from_settings(cls, s: dict) -> "VisionConfig":
        return cls(base_url=s["base_url"], model=s["model"], api_key=s["api_key"], timeout=s["timeout"],
                   disable_thinking=s["disable_thinking"])


@dataclass
class ReadCard:
    name: str
    count: int = 1
    set: str | None = None
    collector_number: str | None = None
    confidence: float = 0.5
    extra: dict = field(default_factory=dict)


def _client(cfg: VisionConfig) -> httpx.AsyncClient:
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
    return httpx.AsyncClient(
        base_url=cfg.base_url.rstrip("/"), headers=headers, timeout=cfg.timeout, transport=transport
    )


async def list_models(cfg: VisionConfig) -> list[str]:
    async with _client(cfg) as client:
        r = await client.get("/models")
        r.raise_for_status()
        return sorted(m["id"] for m in r.json().get("data", []))


def _salvage(text: str) -> list:
    """Recover the complete card objects from a reply that was cut off mid-way."""
    items, decoder, i = [], json.JSONDecoder(), text.find("[")
    while i >= 0:
        i = text.find("{", i + 1)
        if i < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            items.append(obj)
            i += end - 1
    return items


def _box(value) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1 or min(x1, y1) < 0:
        return None
    return [x1, y1, x2, y2]


def parse_cards(text: str, keep_repeats: bool = False) -> list[ReadCard]:
    """Parse the model's reply, tolerating code fences, chatter and replies that were cut off."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
    if start < 0:
        raise VisionError(f"Model reply had no JSON: {text[:200]!r}")
    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as e:
        data = _salvage(text[start:])
        if not data:
            raise VisionError(f"Model reply was not valid JSON: {e}") from e
        log.warning("Model reply was cut off; recovered %d cards from it", len(data))
    items = data.get("cards", []) if isinstance(data, dict) else data
    out: list[ReadCard] = []
    seen: dict[str, ReadCard] = {}
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        # Some models write names like "Wirewood_Symbiote".
        name = re.sub(r"\s+", " ", str(item.get("name") or "").replace("_", " ")).strip()
        if not name:
            continue
        try:
            count = max(1, int(item.get("count") or 1))
        except (TypeError, ValueError):
            count = 1
        try:
            conf = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        if conf > 1:  # some models answer in percent
            conf = conf / 100
        key = name.casefold()
        if key in seen and not keep_repeats:
            # A repeat is the same card: small models sometimes repeat entries until they run out
            # of tokens. (In piles each entry is a separate pile, kept apart and added up later.)
            seen[key].count = max(seen[key].count, count)
            continue
        card = ReadCard(
            name=name,
            count=count,
            set=(str(item["set"]).strip() or None) if item.get("set") else None,
            collector_number=(str(item["collector_number"]).strip() or None) if item.get("collector_number") else None,
            confidence=min(1.0, max(0.0, conf)),
            extra={"raw": item, "box": _box(item.get("box"))},
        )
        seen[key] = card
        out.append(card)
    return out


# Request fields some servers don't know. On a 400 that names one, it is dropped and the
# request retried, so OpenAI-strict servers still work.
OPTIONAL_FIELDS = ("chat_template_kwargs", "response_format")


def _message_text(message: dict) -> tuple[str, str]:
    """Return (answer, reasoning) text from an OpenAI-style message."""
    content = message.get("content") or ""
    if isinstance(content, list):  # some servers return content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    return content, reasoning if isinstance(reasoning, str) else ""


async def read_cards(cfg: VisionConfig, jpeg: bytes, single_card: bool, layout: str = "piles") -> list[ReadCard]:
    return (await read_cards_raw(cfg, jpeg, single_card, layout))[0]


async def read_cards_raw(cfg: VisionConfig, jpeg: bytes, single_card: bool,
                         layout: str = "piles") -> tuple[list[ReadCard], str]:
    """Like read_cards, but also returns the model's reply text for the scan's raw-data view."""
    if not cfg.model:
        raise VisionError("No vision model is configured. Choose one in Settings → Vision model.")
    image_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    body = {
        "model": cfg.model,
        "temperature": 0,
        # Generous so a model that thinks anyway still has room left for the answer.
        "max_tokens": 1024 if single_card else 4096,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": CROP_PROMPT if single_card else LAYOUTS.get(layout, TABLE_PROMPT)},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ],
        "response_format": {"type": "json_schema", "json_schema": {"name": "cards", "schema": CARD_SCHEMA}},
    }
    if cfg.disable_thinking:
        # Honoured by llama-server / llama-swap and vLLM chat templates (Qwen3, etc.).
        body["chat_template_kwargs"] = {"enable_thinking": False}
    async with _client(cfg) as client:
        try:
            r = await client.post("/chat/completions", json=body)
            while r.status_code in (400, 422):
                field = next((f for f in OPTIONAL_FIELDS if f in body and f in r.text), None)
                if field is None:
                    break
                body.pop(field)
                r = await client.post("/chat/completions", json=body)
            r.raise_for_status()
        except httpx.TimeoutException as e:
            raise VisionError(f"The vision server timed out after {cfg.timeout:.0f}s") from e
        except httpx.HTTPStatusError as e:
            raise VisionError(f"Vision server returned {e.response.status_code}: {e.response.text[:300]}") from e
        except httpx.HTTPError as e:
            raise VisionError(f"Could not reach the vision server at {cfg.base_url}: {e}") from e
    try:
        choice = r.json()["choices"][0]
        content, reasoning = _message_text(choice["message"])
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise VisionError(f"Unexpected reply from vision server: {r.text[:300]}") from e
    keep_repeats = not single_card  # each entry is a separate card or pile
    if content.strip():
        try:
            return parse_cards(content, keep_repeats), content
        except VisionError as e:
            raise VisionError(f"{e}. Reply: {content[:300]!r}") from e
    if reasoning.strip():
        # Thinking models sometimes put the whole answer in the reasoning text.
        try:
            return parse_cards(reasoning, keep_repeats), reasoning
        except VisionError:
            pass
    log.warning("Empty reply from %s (finish_reason=%s): %s", cfg.model, choice.get("finish_reason"), r.text[:500])
    if reasoning or choice.get("finish_reason") == "length":
        raise VisionError(
            "The model used its whole reply on reasoning and returned no answer. It looks like a "
            "thinking model: add --reasoning-budget 0 to its llama-server command in llama-swap, "
            "or keep 'Turn off model thinking' enabled in Settings → Vision model."
        )
    raise VisionError(
        "The model returned an empty reply. Check that it can see images (llama-server needs "
        "--mmproj) and look at the llama-swap log for errors."
    )


CHOICE_PROMPT = (
    "This image is a single Magic card whose title was hard to read. Which of these cards is it? "
    "Compare the title, art, mana cost and text box. Answer with one name exactly as listed, or "
    '"none" if it is none of them. Format: {"name": ..., "confidence": ...}\n\nCandidates:\n'
)


async def choose_card(cfg: VisionConfig, jpeg: bytes, candidates: list[str]) -> tuple[str, float] | None:
    """Ask the model to pick which candidate a single-card image shows. Returns (name, confidence)."""
    options = list(dict.fromkeys(candidates))[:8]
    if not options:
        return None
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string", "enum": [*options, "none"]},
                       "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
        "required": ["name", "confidence"],
    }
    body = {
        "model": cfg.model,
        "temperature": 0,
        "max_tokens": 512,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": CHOICE_PROMPT + "\n".join(f"- {o}" for o in options)},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}},
        ]}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "choice", "schema": schema}},
    }
    if cfg.disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        async with _client(cfg) as client:
            r = await client.post("/chat/completions", json=body)
            while r.status_code in (400, 422):
                field = next((f for f in OPTIONAL_FIELDS if f in body and f in r.text), None)
                if field is None:
                    break
                body.pop(field)
                r = await client.post("/chat/completions", json=body)
            r.raise_for_status()
        content, reasoning = _message_text(r.json()["choices"][0]["message"])
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as e:
        log.info("Candidate re-ask failed: %s", e)
        return None
    for text in (content, reasoning):
        start = text.find("{")
        if start < 0:
            continue
        try:
            data, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        name = str(data.get("name", "")).strip() if isinstance(data, dict) else ""
        match = next((o for o in options if o.casefold() == name.casefold()), None)
        if match:
            try:
                conf = float(data.get("confidence", 0.6))
            except (TypeError, ValueError):
                conf = 0.6
            return match, min(1.0, max(0.0, conf / 100 if conf > 1 else conf))
        return None
    return None


@dataclass
class ZoomRead:
    name: str
    count: int | None
    confidence: float
    raw: str
    retried: bool = False


async def read_pile(cfg: VisionConfig, jpeg: bytes) -> ZoomRead | None:
    """Re-read one pile from a close-up: front card's name and how many cards are in it.

    Deliberately not told what the whole-table read said: given a name, small models repeat it.
    """
    body = {
        "model": cfg.model,
        "temperature": 0,
        "max_tokens": 512,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": [
            {"type": "text", "text": ZOOM_PROMPT},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}},
        ]}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "pile", "schema": ZOOM_SCHEMA}},
    }
    if cfg.disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        async with _client(cfg) as client:
            r = await client.post("/chat/completions", json=body)
            while r.status_code in (400, 422):
                field = next((f for f in OPTIONAL_FIELDS if f in body and f in r.text), None)
                if field is None:
                    break
                body.pop(field)
                r = await client.post("/chat/completions", json=body)
            r.raise_for_status()
        content, reasoning = _message_text(r.json()["choices"][0]["message"])
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as e:
        log.info("Pile close-up read failed: %s", e)
        return None
    text = content if content.strip() else reasoning
    start = text.find("{")
    if start < 0:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    name = re.sub(r"\s+", " ", str(data.get("name") or "").replace("_", " ")).strip()
    try:
        count = int(data["count"]) if data.get("count") not in (None, "") else None
    except (TypeError, ValueError):
        count = None
    try:
        conf = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    conf = conf / 100 if conf > 1 else conf
    return ZoomRead(name, count if count and count > 0 else None, min(1.0, max(0.0, conf)), text)
