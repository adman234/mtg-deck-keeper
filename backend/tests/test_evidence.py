import asyncio
import json

import httpx

from deck_keeper.services import evidence, scanner, scryfall, vision

from .conftest import chat_reply
from .test_scan_flow import _table_jpeg


def test_type_check():
    assert evidence.type_check("Land", "Basic Land — Forest") == "ok"
    assert evidence.type_check("Land", "Instant") == "type_mismatch"
    assert evidence.type_check("Creature — Elf", "Artifact Creature — Elf Warrior") == "ok"
    assert evidence.type_check("Tribal Instant — Elf", "Kindred Instant — Elf") == "ok"
    assert evidence.type_check("", "Instant") == "type_unchecked"  # covered by another card
    assert evidence.type_check("Conjuro", "Sorcery") == "type_unchecked"  # foreign: not compared
    assert evidence.type_check("Land", None) is None  # card details unknown


def test_doubts_add_up():
    assert evidence.factor([]) == 1.0
    assert evidence.factor(["type_unchecked"]) > 0.9  # one small doubt alone doesn't flag a line
    assert evidence.factor(["type_unchecked", "reads_disagree", "title_retry"]) < 0.8


def _scan(admin, zoom_reply: dict, type_line: str):
    def model(request):
        text = json.loads(request.content)["messages"][-1]["content"][0]["text"]
        if "close-up" in text:
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(zoom_reply)}}]})
        return chat_reply([{"name": "Force of Will", "count": 1, "confidence": 1, "box": [130, 160, 250, 420]}])

    def scryfall_api(request):
        if request.url.path == "/cards/collection":
            return httpx.Response(200, json={"data": [{"name": "Force of Will", "type_line": type_line}]})
        return httpx.Response(404, json={})

    vision.transport = httpx.MockTransport(model)
    scryfall.transport = httpx.MockTransport(scryfall_api)
    admin.put("/api/admin/settings", json={"values": {"vision.model": "daily"}})
    from deck_keeper.db import session_scope
    from deck_keeper.models import Card
    from deck_keeper.services import matching
    with session_scope() as db:
        db.add(Card(name="Force of Will"))
        db.flush()
        matching.rebuild(db)
    job_id = admin.post("/api/scans", files=[("files", ("a.jpg", _table_jpeg([(300, 300)]), "image/jpeg"))]).json()["id"]
    asyncio.run(scanner.process_job(job_id))
    [line] = admin.get(f"/api/scans/{job_id}").json()["lines"]
    return line


def test_name_with_the_wrong_type_line_is_flagged(admin):
    """Lands photo: a Forest read as "Force of Will". The close-up saw a Land type line."""
    line = _scan(admin, {"name": "Force of Will", "type": "Land", "count": 1, "confidence": 1}, "Instant")
    assert "type_mismatch" in line["flags"] and line["needs_review"]
    assert line["sources"][0]["type_read"] == "Land" and "type_mismatch" in line["sources"][0]["doubts"]


def test_matching_type_line_raises_no_flag(admin):
    line = _scan(admin, {"name": "Force of Will", "type": "Instant", "count": 1, "confidence": 1}, "Instant")
    assert not line["flags"] and line["confidence"] == 1.0


def test_unreadable_type_line_costs_a_little(admin):
    line = _scan(admin, {"name": "Force of Will", "type": "", "count": 1, "confidence": 1}, "Instant")
    assert not line["flags"] and line["confidence"] == 0.95
