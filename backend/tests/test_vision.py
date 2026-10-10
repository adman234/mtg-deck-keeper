import asyncio
import json

import httpx
import pytest

from deck_keeper.services import vision
from deck_keeper.services.vision import VisionConfig, VisionError, parse_cards


def test_parse_plain_json():
    cards = parse_cards('{"cards": [{"name": "Lightning Bolt", "count": 4, "confidence": 0.9}]}')
    assert cards[0].name == "Lightning Bolt" and cards[0].count == 4


def test_parse_fenced_json_with_chatter_and_percent_confidence():
    text = 'Sure!\n```json\n{"cards":[{"name":"Counterspell","count":"2","confidence":85}]}\n```'
    c = parse_cards(text)[0]
    assert (c.count, c.confidence) == (2, 0.85)


def test_parse_bare_list_and_skips_blank_names():
    cards = parse_cards('[{"name": ""}, {"name": "Island"}]')
    assert [c.name for c in cards] == ["Island"]


def test_parse_garbage_raises():
    with pytest.raises(VisionError):
        parse_cards("I can't see any cards")


def test_falls_back_when_structured_output_unsupported():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append("response_format" in body)
        if "response_format" in body:
            return httpx.Response(400, text="response_format not supported")
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"cards":[{"name":"Island",'
                                                                             '"count":1,"confidence":1}]}'}}]})

    vision.transport = httpx.MockTransport(handler)
    try:
        cfg = VisionConfig(base_url="http://llama-swap:8080/v1", model="daily")
        cards = asyncio.run(vision.read_cards(cfg, b"jpeg", single_card=True))
    finally:
        vision.transport = None
    assert calls == [True, False]
    assert cards[0].name == "Island"


def test_missing_model_is_a_clear_error():
    with pytest.raises(VisionError, match="No vision model"):
        asyncio.run(vision.read_cards(VisionConfig(base_url="http://x/v1", model=""), b"", True))
