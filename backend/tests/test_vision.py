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


def _run(handler, **cfg):
    vision.transport = httpx.MockTransport(handler)
    try:
        return asyncio.run(vision.read_cards(VisionConfig(base_url="http://x/v1", model="daily", **cfg),
                                             b"jpeg", single_card=True))
    finally:
        vision.transport = None


def _reply(message: dict, finish_reason="stop") -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": finish_reason}]})


def test_asks_server_to_turn_off_thinking_by_default():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return _reply({"content": '{"cards":[{"name":"Island","count":1,"confidence":1}]}'})

    _run(handler)
    _run(handler, disable_thinking=False)
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in seen[1]
    assert seen[0]["max_tokens"] >= 1024


def test_strict_server_rejecting_template_kwargs_keeps_structured_output():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(sorted(k for k in body if k in vision.OPTIONAL_FIELDS))
        if "chat_template_kwargs" in body:
            return httpx.Response(400, text="Unrecognized request argument supplied: chat_template_kwargs")
        return _reply({"content": '{"cards":[{"name":"Island","count":1,"confidence":1}]}'})

    assert _run(handler)[0].name == "Island"
    assert seen == [["chat_template_kwargs", "response_format"], ["response_format"]]


def test_answer_found_in_reasoning_when_content_empty():
    reply = _reply({"content": "", "reasoning_content": 'Okay. {"cards":[{"name":"Shock","count":1,"confidence":0.9}]}'})
    assert _run(lambda r: reply)[0].name == "Shock"


def test_thinking_ran_out_of_tokens_gives_actionable_error():
    reply = _reply({"content": "", "reasoning_content": "Let me look at the title line carefully..."}, "length")
    with pytest.raises(VisionError, match="reasoning-budget 0"):
        _run(lambda r: reply)


def test_plain_empty_reply_mentions_mmproj():
    with pytest.raises(VisionError, match="mmproj"):
        _run(lambda r: _reply({"content": ""}))


def test_underscores_in_names_become_spaces():
    assert parse_cards('{"cards":[{"name":"Wirewood_Symbiote","count":3,"confidence":0.9}]}')[0].name == "Wirewood Symbiote"


def test_piles_stay_separate_but_repeats_otherwise_do_not_add_up():
    reply = '{"cards":[{"name":"Forest","count":4,"confidence":1,"box":[0,0,100,100]},{"name":"Forest","count":3,"confidence":1}]}'
    piles = parse_cards(reply, keep_repeats=True)
    assert [c.count for c in piles] == [4, 3] and piles[0].extra["box"] == [0, 0, 100, 100]
    assert [c.count for c in parse_cards(reply)] == [4]


def test_layout_picks_the_whole_photo_prompt():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content)["messages"][1]["content"][0]["text"])
        return _reply({"content": '{"cards":[]}'})

    vision.transport = httpx.MockTransport(handler)
    try:
        cfg = VisionConfig(base_url="http://x/v1", model="daily")
        for layout in ("piles", "spread"):
            asyncio.run(vision.read_cards(cfg, b"jpeg", single_card=False, layout=layout))
        asyncio.run(vision.read_cards(cfg, b"jpeg", single_card=True, layout="piles"))
    finally:
        vision.transport = None
    assert "FRONT card" in seen[0] and "slivers" in seen[0]
    assert "FRONT card" not in seen[1]
    assert "single Magic card" in seen[2]
