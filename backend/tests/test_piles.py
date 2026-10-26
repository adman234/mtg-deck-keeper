import asyncio
import json

import httpx

from deck_keeper.services import matching, piles, scanner, scryfall, vision
from deck_keeper.services.vision import ReadCard

from .conftest import chat_reply
from .test_scan_flow import _table_jpeg


def _card(name, count=4, box=None, conf=0.9):
    return ReadCard(name=name, count=count, confidence=conf, extra={"box": box})


def test_boxes_on_0_to_1000_scale():
    c = _card("Shock", box=[100, 200, 300, 600])
    assert piles.place_boxes([c], sent=(1600, 1200), photo=(2400, 1800)) == "0-1000"
    assert c.extra["bbox"] == [240, 360, 480, 720]


def test_boxes_in_pixels_of_the_sent_image():
    a, b = _card("Shock", box=[100, 100, 400, 500]), _card("Island", box=[1200, 100, 1500, 500])
    assert piles.place_boxes([a, b], sent=(1600, 1200), photo=(2400, 1800)) == "pixels"
    assert a.extra["bbox"] == [150, 150, 450, 600]


def test_tiny_or_missing_boxes_are_ignored():
    a, b = _card("Shock", box=[100, 100, 105, 105]), _card("Island")
    piles.place_boxes([a, b], sent=(1600, 1200), photo=(1600, 1200))
    assert "bbox" not in a.extra and "bbox" not in b.extra


def test_same_spot_listed_twice_is_one_pile():
    a, b, c = _card("Natural Order", conf=0.9), _card("Natural Mage", conf=0.5), _card("Shock")
    a.extra["bbox"], b.extra["bbox"], c.extra["bbox"] = [100, 100, 300, 400], [110, 105, 290, 400], [600, 100, 300, 400]
    kept, dropped = piles.drop_duplicates([a, b, c])
    assert [k.name for k in kept] == ["Natural Order", "Shock"] and [d.name for d in dropped] == ["Natural Mage"]


def test_close_up_read_wins_and_fixes_the_count(client):
    c = _card("Lightnin Blt", count=5)
    c.extra["zoom"] = vision.ZoomRead("Lightning Bolt", 4, 0.9, "{}")
    from deck_keeper.db import session_scope
    with session_scope() as db:
        d = piles.decide(matching.get_index(db), c, [c])
    assert (d.name, d.count, d.used, d.count_differs) == ("Lightning Bolt", 4, "zoom", True)


def test_close_up_of_the_wrong_pile_is_not_trusted(client):
    """The box landed on the Counterspell pile, which the photo read also listed separately."""
    bolt, counter = _card("Lightning Bolt"), _card("Counterspell")
    bolt.extra["zoom"] = vision.ZoomRead("Counterspell", 4, 0.9, "{}")
    from deck_keeper.db import session_scope
    with session_scope() as db:
        d = piles.decide(matching.get_index(db), bolt, [bolt, counter])
    assert (d.name, d.used) == ("Lightning Bolt", "photo")


def test_scan_zooms_into_piles(admin):
    """Whole photo: Bolt pile of 5 (miscounted), and the same Bolt pile listed again as a misread.
    Close-ups count 4. Result: one line of 4 with a cut-out thumbnail and a raw-data trail."""
    calls = {"zoom": 0}

    def model(request):
        text = json.loads(request.content)["messages"][-1]["content"][0]["text"]
        if "close-up" in text:
            calls["zoom"] += 1
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(
                {"name": "Lightning Bolt", "count": 4, "confidence": 0.95})}}]})
        return chat_reply([
            {"name": "Lightning Bolt", "count": 5, "confidence": 0.9, "box": [130, 160, 250, 420]},
            {"name": "Lightnin Mage", "count": 2, "confidence": 0.4, "box": [135, 165, 255, 425]},
        ])

    vision.transport = httpx.MockTransport(model)
    scryfall.transport = httpx.MockTransport(lambda r: httpx.Response(404, json={}))
    admin.put("/api/admin/settings", json={"values": {"vision.model": "daily"}})
    job_id = admin.post("/api/scans", data={"prepass": "false", "layout": "piles"},
                        files=[("files", ("a.jpg", _table_jpeg([(300, 300)]), "image/jpeg"))]).json()["id"]
    asyncio.run(scanner.process_job(job_id))
    scan = admin.get(f"/api/scans/{job_id}").json()
    assert calls["zoom"] == 1  # the duplicate entry was dropped before zooming
    [line] = scan["lines"]
    assert (line["card"]["name"], line["qty"]) == ("Lightning Bolt", 4)
    assert "count_mismatch" in line["flags"]  # 5 vs 4: still worth a look
    src = line["sources"][0]
    assert src["crop_url"] and src["bbox"] and src["used"] == "zoom"
    assert admin.get(src["crop_url"]).status_code == 200

    raw = admin.get(f"/api/scans/{job_id}/debug").json()
    img = raw["images"][0]
    assert img["box_scale"] == "0-1000" and "Lightnin Mage" in img["reply"]
    assert img["duplicates"][0]["name"] == "Lightnin Mage"
    det = img["detections"][0]
    assert det["raw"]["photo"]["count"] == 5 and det["raw"]["zoom"]["count"] == 4
    dl = admin.get(f"/api/scans/{job_id}/debug", params={"download": "true"})
    assert "attachment" in dl.headers["content-disposition"]


def test_zoom_can_be_turned_off(admin):
    calls = {"zoom": 0}

    def model(request):
        if "close-up" in json.loads(request.content)["messages"][-1]["content"][0]["text"]:
            calls["zoom"] += 1
        return chat_reply([{"name": "Shock", "count": 3, "confidence": 0.9, "box": [130, 160, 250, 420]}])

    vision.transport = httpx.MockTransport(model)
    scryfall.transport = httpx.MockTransport(lambda r: httpx.Response(404, json={}))
    admin.put("/api/admin/settings", json={"values": {"vision.model": "daily", "vision.pile_zoom": False}})
    job_id = admin.post("/api/scans", data={"prepass": "false"},
                        files=[("files", ("a.jpg", _table_jpeg([(300, 300)]), "image/jpeg"))]).json()["id"]
    asyncio.run(scanner.process_job(job_id))
    assert calls["zoom"] == 0


def test_copy_limit():
    assert scanner.copy_limit("Lightning Bolt") == 4
    assert scanner.copy_limit("Forest") is None and scanner.copy_limit("Relentless Rats") is None
    assert scanner.copy_limit("Seven Dwarves") == 7
