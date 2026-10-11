import asyncio
import io
import json

import httpx
from PIL import Image

from deck_keeper.services import scanner, scryfall, vision

from .conftest import chat_reply


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (800, 600), (30, 80, 40)).save(buf, "JPEG")
    return buf.getvalue()


def _fake_model(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert body["model"] == "daily"
    return chat_reply([
        {"name": "Lightning Bolf", "count": 4, "confidence": 0.95},
        {"name": "Monastery Swiftspear", "count": 4, "confidence": 0.97},
        {"name": "Delver of Secrets", "count": 1, "confidence": 0.5},
        {"name": "Mountain", "count": 20, "confidence": 0.9},
        {"name": "Totally Made Up Card", "count": 1, "confidence": 0.3},
    ])


def test_photo_to_saved_deck(admin):
    admin.put("/api/admin/settings", json={"values": {"vision.model": "daily"}})
    vision.transport = httpx.MockTransport(_fake_model)
    scryfall.transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"data": [
        {"name": "Lightning Bolt", "type_line": "Instant", "cmc": 1, "mana_cost": "{R}", "colors": ["R"],
         "image_uris": {"normal": "https://cards.scryfall.io/x.jpg"}}]}))

    r = admin.post("/api/scans", data={"name": "Burn", "prepass": "false"},
                   files=[("files", ("a.jpg", _jpeg(), "image/jpeg")), ("files", ("b.jpg", _jpeg(), "image/jpeg"))])
    assert r.status_code == 201, r.text
    job_id = r.json()["id"]
    asyncio.run(scanner.process_job(job_id))

    scan = admin.get(f"/api/scans/{job_id}").json()
    assert scan["status"] == "review", scan["error"]
    lines = {(ln["card"] or {}).get("name") or ln["raw_name"]: ln for ln in scan["lines"]}
    # Two photos of the same list are summed
    assert lines["Lightning Bolt"]["qty"] == 8
    assert lines["Lightning Bolt"]["raw_name"] == "Lightning Bolf"
    assert "unmatched" in lines["Totally Made Up Card"]["flags"]
    assert "low_confidence" in lines["Delver of Secrets // Insectile Aberration"]["flags"]
    assert "check_basic_count" in lines["Mountain"]["flags"]
    assert not lines["Monastery Swiftspear"]["needs_review"]

    # Saving is blocked while a line has no card
    assert admin.post(f"/api/scans/{job_id}/save", json={"name": "Burn"}).status_code == 400

    # Override: swap the made-up card for a suggestion, fix the land count, sideboard the Delver
    unknown = lines["Totally Made Up Card"]
    cards = admin.get("/api/cards/search", params={"q": "counterspell"}).json()
    admin.patch(f"/api/scans/{job_id}/lines/{unknown['id']}", json={"card_id": cards[0]["id"]})
    admin.patch(f"/api/scans/{job_id}/lines/{lines['Mountain']['id']}", json={"qty": 18})
    scan = admin.patch(f"/api/scans/{job_id}/lines/{lines['Delver of Secrets // Insectile Aberration']['id']}",
                       json={"board": "side", "accepted": True}).json()
    assert not any(ln["needs_review"] for ln in scan["lines"])

    r = admin.post(f"/api/scans/{job_id}/save", json={"name": "Burn", "format": "modern"})
    assert r.status_code == 200, r.text
    deck = admin.get(f"/api/decks/{r.json()['deck_id']}").json()
    assert deck["counts"] == {"commander": 0, "main": 8 + 8 + 18 + 2, "side": 2, "maybe": 0}
    bolt = next(c for c in deck["cards"] if c["card"]["name"] == "Lightning Bolt")
    assert bolt["card"]["type_line"] == "Instant"

    arena = admin.get(f"/api/decks/{deck['id']}/export", params={"format": "arena"}).text
    assert "Deck\n" in arena and "8 Lightning Bolt" in arena and "Sideboard\n2 Delver of Secrets" in arena


def test_vision_server_down_marks_scan_failed(admin):
    admin.put("/api/admin/settings", json={"values": {"vision.model": "daily"}})

    def down(request):
        raise httpx.ConnectError("connection refused")

    vision.transport = httpx.MockTransport(down)
    job_id = admin.post("/api/scans", data={"prepass": "true"},
                        files=[("files", ("a.jpg", _jpeg(), "image/jpeg"))]).json()["id"]
    asyncio.run(scanner.process_job(job_id))
    scan = admin.get(f"/api/scans/{job_id}").json()
    assert scan["status"] == "failed" and "Could not reach the vision server" in scan["error"]


def test_rejects_non_images(admin):
    r = admin.post("/api/scans", files=[("files", ("a.txt", b"hello", "text/plain"))])
    assert r.status_code == 400 and "not an image" in r.json()["detail"]


def test_empty_card_list_and_no_scryfall_gives_clear_error(admin):
    from deck_keeper.db import session_scope
    from deck_keeper.models import Card
    from deck_keeper.services import matching

    with session_scope() as db:
        db.query(Card).delete()
    with session_scope() as db:
        matching.rebuild(db)
    admin.put("/api/admin/settings", json={"values": {"vision.model": "daily"}})

    def offline(request):
        raise httpx.ConnectError("Name or service not known")

    scryfall.transport = httpx.MockTransport(offline)
    job_id = admin.post("/api/scans", files=[("files", ("a.jpg", _jpeg(), "image/jpeg"))]).json()["id"]
    asyncio.run(scanner.process_job(job_id))
    scan = admin.get(f"/api/scans/{job_id}").json()
    assert scan["status"] == "failed" and "Scryfall could not be reached" in scan["error"]


def test_foreign_card_identified_in_full_scan(admin):
    admin.put("/api/admin/settings", json={"values": {"vision.model": "daily"}})
    vision.transport = httpx.MockTransport(lambda r: chat_reply([
        {"name": "稲妻", "count": 4, "set": "STA", "collector_number": "042/063", "confidence": 0.9}]))

    def scry(request):
        if request.url.path == "/cards/sta/42":
            return httpx.Response(200, json={"name": "Lightning Bolt", "printed_name": "稲妻", "lang": "ja"})
        return httpx.Response(404, json={})

    scryfall.transport = httpx.MockTransport(scry)
    job_id = admin.post("/api/scans", data={"prepass": "false"},
                        files=[("files", ("a.jpg", _jpeg(), "image/jpeg"))]).json()["id"]
    asyncio.run(scanner.process_job(job_id))
    scan = admin.get(f"/api/scans/{job_id}").json()
    line = scan["lines"][0]
    assert (line["card"]["name"], line["qty"], line["needs_review"]) == ("Lightning Bolt", 4, False)
    assert line["sources"][0]["match_method"] == "printing" and line["raw_name"] == "稲妻"
