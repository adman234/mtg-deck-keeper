import asyncio
import json

import httpx
import pytest

from deck_keeper.db import session_scope
from deck_keeper.models import Card, Lookup
from deck_keeper.services import matching, resolver, scryfall, vision
from deck_keeper.services.vision import ReadCard, VisionConfig

CFG = VisionConfig(base_url="http://llama/v1", model="daily")
ALL_ON = resolver.Options()


def _scryfall(routes: dict, calls: list):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path + ("?" + request.url.query.decode() if request.url.query else ""))
        for prefix, reply in routes.items():
            if request.url.path.startswith(prefix):
                return reply
        return httpx.Response(404, json={"object": "error"})
    return httpx.MockTransport(handler)


async def _direct(coro):
    return await coro


def _resolve(read, crop_path=None, opts=ALL_ON):
    with session_scope() as db:
        return asyncio.run(resolver.resolve(db, matching.get_index(db), read, crop_path, CFG, opts, _direct))


def _name(card_id):
    with session_scope() as db:
        return db.get(Card, card_id).name


@pytest.fixture(autouse=True)
def _reset():
    yield
    scryfall.transport = None
    vision.transport = None


def test_good_english_name_needs_no_lookups(client):
    calls = []
    scryfall.transport = _scryfall({}, calls)
    res = _resolve(ReadCard("Lightning Bolf", set="sta", collector_number="42"))
    assert (_name(res.card_id), res.method) == ("Lightning Bolt", "name") and calls == []


def test_japanese_title_resolved_by_collector_number(client):
    calls = []
    scryfall.transport = _scryfall({"/cards/sta/42": httpx.Response(200, json={
        "name": "Lightning Bolt", "printed_name": "稲妻", "lang": "ja"})}, calls)
    res = _resolve(ReadCard("稲妻", set="STA", collector_number="042/063"))
    assert (_name(res.card_id), res.method) == ("Lightning Bolt", "printing")
    # Cached: a second card from the same printing makes no request
    _resolve(ReadCard("稲妻", set="sta", collector_number="42"))
    assert calls == ["/cards/sta/42"]


def test_misread_collector_number_is_not_trusted(client):
    """The number points at a different card and the names don't agree."""
    scryfall.transport = _scryfall({"/cards/mid/154": httpx.Response(200, json={
        "name": "Play with Fire", "lang": "en"})}, [])
    res = _resolve(ReadCard("Counterspel", set="mid", collector_number="154"))
    assert _name(res.card_id) == "Counterspell" and res.method == "name"


def test_german_name_found_by_foreign_search(client):
    calls = []
    scryfall.transport = _scryfall({"/cards/search": httpx.Response(200, json={"data": [
        {"name": "Lightning Strike", "printed_name": "Blitzeinschlag", "lang": "de"},
        {"name": "Lightning Bolt", "printed_name": "Blitzschlag", "lang": "de"},
    ]})}, calls)
    res = _resolve(ReadCard("Blitzschlag"))
    assert (_name(res.card_id), res.method) == ("Lightning Bolt", "foreign:de")
    assert "lang%3Aany" in calls[0] and "include_multilingual=true" in calls[0]


def test_unknown_card_name_from_scryfall_is_added(client):
    scryfall.transport = _scryfall({"/cards/search": httpx.Response(200, json={"data": [
        {"name": "Brand New Card", "printed_name": "Brandneue Karte", "lang": "de"}]})}, [])
    res = _resolve(ReadCard("Brandneue Karte"))
    assert _name(res.card_id) == "Brand New Card"


def test_scryfall_offline_falls_back_quietly(client):
    def down(request):
        raise httpx.ConnectError("offline")

    scryfall.transport = httpx.MockTransport(down)
    res = _resolve(ReadCard("稲妻", set="sta", collector_number="42"))
    assert res.card_id is None
    with session_scope() as db:
        assert db.query(Lookup).count() == 0  # failures aren't cached as "not found"


def _crop(tmp_path):
    crop = tmp_path / "crop.jpg"
    crop.write_bytes(b"jpeg")
    return str(crop)


def _model_picks(name, seen=None):
    def handler(request):
        body = json.loads(request.content)
        if seen is not None:
            seen.append(body["response_format"]["json_schema"]["schema"]["properties"]["name"]["enum"])
        return httpx.Response(200, json={"choices": [{"message": {
            "content": json.dumps({"name": name, "confidence": 0.9})}}]})
    return httpx.MockTransport(handler)


def test_reask_breaks_a_near_tie(client, tmp_path):
    seen = []
    vision.transport = _model_picks("Lightning Bolt", seen)
    scryfall.transport = _scryfall({}, [])
    # "Lightning Stb": Strike 83%, Bolt 82% -- too close to call by spelling alone
    res = _resolve(ReadCard("Lightning Stb", confidence=0.7), crop_path=_crop(tmp_path))
    assert (_name(res.card_id), res.method) == ("Lightning Bolt", "choice")
    assert set(seen[0]) == {"Lightning Strike", "Lightning Bolt", "none"}
    # The pick keeps its own spelling score and the lower confidence, so it still gets reviewed
    assert res.score < resolver.GOOD and res.confidence == 0.7


def test_reask_cannot_overrule_a_clearly_better_match(client, tmp_path):
    """Regression: an art-only crop read as "Terror of the Tithes" was re-asked and the model
    picked Terror of the Fairgrounds (71%) over Terror of the Peaks (82%)."""
    with session_scope() as db:
        db.add_all([Card(name="Terror of the Peaks"), Card(name="Terror of the Fairgrounds")])
    with session_scope() as db:
        matching.rebuild(db)
    calls = []
    vision.transport = _model_picks("Terror of the Fairgrounds", calls)
    scryfall.transport = _scryfall({}, [])
    res = _resolve(ReadCard("Terror of the Tithes", confidence=0.5), crop_path=_crop(tmp_path))
    assert (_name(res.card_id), res.method) == ("Terror of the Peaks", "name")
    assert calls == []  # only one candidate is close enough, so there was nothing to ask


def test_reask_none_leaves_it_for_review(client, tmp_path):
    vision.transport = _model_picks("none")
    scryfall.transport = _scryfall({}, [])
    res = _resolve(ReadCard("Lightning Stb"), crop_path=_crop(tmp_path))
    assert res.method == "name" and res.score < resolver.GOOD


def test_foreign_search_survives_a_misread_letter(client):
    calls = []

    card = {"name": "Lightning Bolt", "printed_name": "Foudre éclair", "lang": "fr"}

    def handler(request):
        # Like Scryfall: every word must appear in the name (accents ignored).
        q = request.url.params["q"].replace(" lang:any", "")
        calls.append(q)
        words = [q.strip('"')] if q.startswith('"') else q.split()
        name = scryfall.loose(card["printed_name"]).replace("é", "e")
        if all(scryfall.loose(w) in name for w in words):
            return httpx.Response(200, json={"data": [card]})
        return httpx.Response(404, json={})

    scryfall.transport = httpx.MockTransport(handler)
    res = _resolve(ReadCard("Foudre eclalr"))
    assert (_name(res.card_id), res.method) == ("Lightning Bolt", "foreign:fr")
    assert calls == ['"Foudre eclalr"', "Foudre eclalr", "Foudre"]


def test_everything_off_is_plain_fuzzy_matching(client):
    calls = []
    scryfall.transport = _scryfall({}, calls)
    res = _resolve(ReadCard("稲妻", set="sta", collector_number="42"), opts=resolver.Options(False, False, False))
    assert res.card_id is None and calls == []
