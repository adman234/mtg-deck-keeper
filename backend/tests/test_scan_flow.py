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


def _table_jpeg(cards):
    """A photo of cards laid on a mat, so the OpenCV pre-pass has something to cut out."""
    import cv2
    import numpy as np
    img = np.full((1200, 1600, 3), (40, 90, 50), np.uint8)
    for cx, cy in cards:
        for size, color in (((252, 352), (20, 20, 20)), ((226, 326), (180, 200, 220)), ((190, 140), (90, 60, 140))):
            cv2.fillPoly(img, [cv2.boxPoints(((cx, cy), size, 0)).astype(np.int32)], color)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _by_prompt(crop_reply, photo_reply):
    def handler(request):
        text = json.loads(request.content)["messages"][-1]["content"][0]["text"]
        reply = crop_reply if "single Magic card" in text else photo_reply
        return reply() if callable(reply) else reply
    return httpx.MockTransport(handler)


def _scan(admin, jpeg, prepass=True):
    admin.put("/api/admin/settings", json={"values": {"vision.model": "daily"}})
    scryfall.transport = httpx.MockTransport(lambda r: httpx.Response(404, json={}))
    job_id = admin.post("/api/scans", data={"prepass": "true" if prepass else "false"},
                        files=[("files", ("a.jpg", jpeg, "image/jpeg"))]).json()["id"]
    asyncio.run(scanner.process_job(job_id))
    return admin.get(f"/api/scans/{job_id}").json()


def test_failed_whole_photo_read_is_reported_not_hidden(admin):
    vision.transport = _by_prompt(chat_reply([{"name": "Counterspell", "count": 1, "confidence": 0.95}]),
                                  httpx.Response(500, text="model crashed"))
    scan = _scan(admin, _table_jpeg([(300, 300), (700, 300), (1100, 300)]))
    assert scan["status"] == "review"
    img = scan["images"][0]
    assert img["crops_found"] == 3 and img["photo_cards"] is None and "500" in img["photo_error"]
    assert scan["lines"][0]["qty"] == 3  # crops still count when the whole-photo read failed


def test_weak_crop_the_whole_photo_did_not_see_is_not_counted(admin):
    crops = iter([
        chat_reply([{"name": "Counterspell", "count": 1, "confidence": 0.95}]),
        chat_reply([{"name": "Counterspell", "count": 1, "confidence": 0.95}]),
        chat_reply([{"name": "Mountian Thing", "count": 1, "confidence": 0.4}]),  # an art-only crop
    ])
    vision.transport = _by_prompt(lambda: next(crops),
                                  chat_reply([{"name": "Counterspell", "count": 3, "confidence": 0.9}]))
    scan = _scan(admin, _table_jpeg([(300, 300), (700, 300), (1100, 300)]))
    names = {(ln["card"] or {}).get("name") or ln["raw_name"]: ln["qty"] for ln in scan["lines"]}
    assert names == {"Counterspell": 3}
    assert scan["images"][0]["unconfirmed_crops"] == 1


def test_cut_off_photo_reply_still_counts(admin):
    truncated = '{"cards":[{"name":"Mountain","count":4,"confidence":0.9},{"name":"Shock","count":2,"confidence":0.9},{"na'
    vision.transport = _by_prompt(chat_reply([]), httpx.Response(200, json={
        "choices": [{"message": {"content": truncated}, "finish_reason": "length"}]}))
    scan = _scan(admin, _jpeg(), prepass=False)
    names = {(ln["card"] or {}).get("name") or ln["raw_name"]: ln["qty"] for ln in scan["lines"]}
    assert names["Mountain"] == 4 and names["Shock"] == 2


def test_layout_is_stored_and_can_change_on_rerun(admin):
    vision.transport = _by_prompt(chat_reply([]), chat_reply([{"name": "Counterspell", "count": 4, "confidence": 0.9}]))
    scan = _scan(admin, _jpeg(), prepass=False)
    assert scan["layout"] == "piles"
    r = admin.post(f"/api/scans/{scan['id']}/rerun", json={"layout": "spread"})
    assert r.status_code == 200 and r.json()["layout"] == "spread"
    assert admin.post(f"/api/scans/{scan['id']}/rerun", json={"layout": "heap"}).status_code in (400, 409)
