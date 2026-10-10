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
                },
                "required": ["name", "count", "confidence"],
            },
        }
    },
    "required": ["cards"],
}

SYSTEM_PROMPT = (
    "You identify Magic: The Gathering cards in photos. Read each card's title line exactly as printed. "
    "Use the English card name when you recognise the card; otherwise copy the printed text. "
    "Never invent a name you cannot read. If the set code or collector number at the bottom-left "
    "is clearly legible, include it, otherwise leave it out. confidence is your probability (0-1) "
    "that the name is correct. Reply with JSON only."
)

PHOTO_PROMPT = (
    "List every distinct card visible in this photo. For each one, count how many copies of that "
    "card are visible (count each physical card once). Include cards that are partly covered if the "
    'title is readable. Format: {"cards": [{"name": ..., "count": ..., "set": ..., '
    '"collector_number": ..., "confidence": ...}]}'
)

CROP_PROMPT = (
    "This image is a single Magic card. Return exactly one entry with count 1. If the image is not a "
    'card or the title is unreadable, return {"cards": []}. Format: {"cards": [{"name": ..., '
    '"count": 1, "set": ..., "collector_number": ..., "confidence": ...}]}'
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


def parse_cards(text: str) -> list[ReadCard]:
    """Parse the model's reply, tolerating code fences and chatter around the JSON."""
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
        raise VisionError(f"Model reply was not valid JSON: {e}") from e
    items = data.get("cards", []) if isinstance(data, dict) else data
    out = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
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
        out.append(ReadCard(
            name=name,
            count=count,
            set=(str(item["set"]).strip() or None) if item.get("set") else None,
            collector_number=(str(item["collector_number"]).strip() or None) if item.get("collector_number") else None,
            confidence=min(1.0, max(0.0, conf)),
        ))
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


async def read_cards(cfg: VisionConfig, jpeg: bytes, single_card: bool) -> list[ReadCard]:
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
                {"type": "text", "text": CROP_PROMPT if single_card else PHOTO_PROMPT},
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
    if content.strip():
        return parse_cards(content)
    if reasoning.strip():
        # Thinking models sometimes put the whole answer in the reasoning text.
        try:
            return parse_cards(reasoning)
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
