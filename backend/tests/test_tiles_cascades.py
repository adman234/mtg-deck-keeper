import asyncio
import json
import sys
from pathlib import Path

import cv2
import httpx
import numpy as np

from deck_keeper.services import cascades, detection, scanner, scryfall, tiles, vision

from .conftest import chat_reply

sys.path.insert(0, str(Path(__file__).parent))
from cascade_photo import column  # noqa: E402


def _card(name, box):
    return vision.ReadCard(name=name, count=1, confidence=0.9, extra={"bbox": box})


def test_tile_grid_covers_the_photo_with_overlap():
    grid = tiles.plan(2400, 1800, 100)
    assert len(grid) >= 6
    assert min(t.x for t in grid) == 0 and max(t.x + t.w for t in grid) == 2400
    assert min(t.y for t in grid) == 0 and max(t.y + t.h for t in grid) == 1800
    xs = sorted({t.x for t in grid})
    w = grid[0].w
    assert xs[1] < w  # neighbours overlap


def test_when_to_tile():
    assert tiles.reason(10, False, None, 300, 2400) is None
    assert "stuck" in tiles.reason(10, False, {"name": "Coral Reef", "dropped": 50}, 300, 2400)
    assert tiles.reason(40, False, None, 300, 2400) == "40 cards listed"
    assert tiles.reason(10, True, None, 300, 2400) == "the model's list was cut off"
    assert tiles.reason(10, False, None, 150, 2400) == "the cards are small in the photo"


def test_cards_seen_by_two_tiles_are_kept_once():
    left, right = tiles.Tile(0, 0, 1000, 1000), tiles.Tile(800, 0, 1000, 1000)
    # On the seam: the left tile saw it whole (further from its edge), the right tile only just.
    a, b = _card("Scapeshift", [700, 100, 200, 100]), _card("Scapeshlft", [705, 102, 196, 98])
    c = _card("Sol Ring", [1300, 500, 200, 100])
    kept = tiles.merge([(a, left), (b, right), (c, right)], (1800, 1000))
    assert [k.name for k in kept] == ["Scapeshift", "Sol Ring"]


def test_columns_found_and_counted_even_photographed_at_an_angle():
    for n, tilt, drift in ((8, False, -4), (12, True, -4), (6, False, 6)):
        img, _ = column(n, drift=drift, tilt=tilt)
        found = detection.find_cards(img)
        cols = cascades.find(img, [f.corners for f in found])
        assert [len(c.strips) for c in cols] == [n - 1], (n, tilt)
        strip = cascades.cut(img, cols[0].strips[0])
        assert strip.shape[1] == cascades.READ_W and 40 < strip.shape[0] < 200


def test_separate_cards_are_not_a_column():
    img = np.full((1200, 1600, 3), (120, 70, 40), np.uint8)
    from cascade_photo import card
    rng = np.random.default_rng(1)
    for x, y in ((100, 100), (450, 100), (800, 100), (100, 600), (450, 600)):
        img[y:y + 349, x:x + 250] = card(rng)
    found = detection.find_cards(img)
    assert len(found) == 5 and cascades.find(img, [f.corners for f in found]) == []


def test_scan_reads_each_title_in_a_column(admin):
    """A column of 7 cards: the whole-photo read is hopeless, so each title strip is read on its
    own and replaces what the whole-photo read listed there."""
    img, n = column(7, tilt=True)
    jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    titles = iter([f"Card {k}" for k in range(1, 20)])

    def model(request):
        text = json.loads(request.content)["messages"][-1]["content"][0]["text"]
        if "top strip" in text:
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(
                {"name": next(titles), "confidence": 0.9})}}]})
        if "close-up" in text:
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(
                {"name": "Front Card", "type": "", "count": 1, "confidence": 0.9})}}]})
        return chat_reply([{"name": "Coral Reef", "count": 1, "confidence": 1, "box": [380, 300, 620, 720]},
                           {"name": "Front Card", "count": 1, "confidence": 1, "box": [390, 660, 610, 940]}])

    vision.transport = httpx.MockTransport(model)
    scryfall.transport = httpx.MockTransport(lambda r: httpx.Response(404, json={}))
    admin.put("/api/admin/settings", json={"values": {"vision.model": "daily"}})
    job_id = admin.post("/api/scans", files=[("files", ("a.jpg", jpeg, "image/jpeg"))]).json()["id"]
    asyncio.run(scanner.process_job(job_id))
    scan = admin.get(f"/api/scans/{job_id}").json()
    names = sorted(ln["raw_name"] for ln in scan["lines"])
    assert names == sorted(["Front Card", *[f"Card {k}" for k in range(1, n)]]), names
    raw = admin.get(f"/api/scans/{job_id}/debug").json()["images"][0]
    assert raw["columns"][0]["strips"] == n - 1 and raw["replaced_by_strips"] == 1
    strip = next(d for d in raw["detections"] if d["source"] == "strip")
    assert strip["crop_url"] and strip["raw"]["strip"]["column"] == 1


def test_scan_with_many_cards_is_read_in_tiles(admin):
    """The whole-photo read lists 35 entries (too many to trust): the photo is read again in
    tiles, and a card seen by two tiles on a seam counts once."""
    from .test_scan_flow import _jpeg
    whole = [{"name": f"Blur {k}", "count": 1, "confidence": 0.5, "box": [20 * k, 10, 20 * k + 18, 60]} for k in range(35)]

    def model(request):
        text = json.loads(request.content)["messages"][-1]["content"][0]["text"]
        if "one part of a larger photo" in text:
            # Every tile sees a card in its own top-left corner, and the first tile's card again.
            return chat_reply([{"name": "Sol Ring", "count": 1, "confidence": 0.9, "box": [100, 100, 300, 400]}])
        if "close-up" in text:
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(
                {"name": "Sol Ring", "type": "Artifact", "count": 1, "confidence": 0.9})}}]})
        return chat_reply(whole)

    vision.transport = httpx.MockTransport(model)
    scryfall.transport = httpx.MockTransport(lambda r: httpx.Response(404, json={}))
    admin.put("/api/admin/settings", json={"values": {"vision.model": "daily"}})
    job_id = admin.post("/api/scans", files=[("files", ("a.jpg", _jpeg(), "image/jpeg"))]).json()["id"]
    asyncio.run(scanner.process_job(job_id))
    raw = admin.get(f"/api/scans/{job_id}/debug").json()["images"][0]
    assert raw["tiled"] == "35 cards listed" and len(raw["tiles"]) >= 4
    lines = admin.get(f"/api/scans/{job_id}").json()["lines"]
    assert all(ln["raw_name"] == "Sol Ring" for ln in lines)
    assert sum(ln["qty"] for ln in lines) == len(raw["tiles"])  # one per tile: none of them overlap
