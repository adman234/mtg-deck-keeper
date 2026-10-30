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


def test_names_cut_off_by_a_tile_edge_are_left_to_the_tile_that_saw_them_whole():
    """Scan 18: the left tile's edge cut "Genesis Chamber" to "Genesis", a real card."""
    left, right = tiles.Tile(0, 0, 1000, 1000), tiles.Tile(800, 0, 1000, 1000)
    cut = _card("Genesis", [880, 300, 120, 60])  # boxed right up to the left tile's edge
    whole = _card("Genesis Chamber", [880, 330, 300, 60])
    edge = _card("Mountain", [1700, 600, 100, 60])  # at the photo's edge, not a seam: kept
    kept = tiles.merge([(cut, left), (whole, right), (edge, right)], (1800, 1000))
    assert [k.name for k in kept] == ["Genesis Chamber", "Mountain"]


def test_columns_found_and_counted_even_photographed_at_an_angle():
    for n, tilt, drift in ((8, False, -4), (12, True, -4), (6, False, 6)):
        img, _ = column(n, drift=drift, tilt=tilt)
        found = detection.find_cards(img)
        cols = cascades.find(img, [f.corners for f in found])
        assert [len(c.strips) for c in cols] == [n - 1], (n, tilt)
        front = np.float32(found[cols[0].front].corners).reshape(-1, 1, 2)
        for s in cols[0].strips:  # no strip is the front card's own title
            assert cv2.pointPolygonTest(front, tuple(float(v) for v in np.mean(s.title, 0)), False) < 0
        strip = cascades.cut(img, cols[0].strips[0])
        assert strip.shape[1] == cascades.READ_W and 40 < strip.shape[0] < 200


def test_column_found_when_its_front_card_sits_apart_or_looks_different():
    img, n = column(9, front_shift=55)
    cols = cascades.find(img, [f.corners for f in detection.find_cards(img)])
    assert [len(c.strips) for c in cols] == [n - 1]

    # The front card's own top edge matches nothing behind it: the edges of the other column are used.
    img, n = column(9, odd_front=True, beside=8)
    found = detection.find_cards(img)
    cols = cascades.find(img, [f.corners for f in found])
    assert sorted(len(c.strips) for c in cols) == [7, n - 1]


def test_separate_cards_are_not_a_column():
    img = np.full((1200, 1600, 3), (120, 70, 40), np.uint8)
    from cascade_photo import card
    rng = np.random.default_rng(1)
    for x, y in ((100, 100), (450, 100), (800, 100), (100, 600), (450, 600)):
        img[y:y + 349, x:x + 250] = card(rng)
    found = detection.find_cards(img)
    assert len(found) == 5 and cascades.find(img, [f.corners for f in found]) == []


def test_scan_reads_each_title_in_a_column(admin, monkeypatch):
    """A column of 7 cards: the whole-photo read is hopeless, so each title strip is read on its
    own and replaces what the whole-photo read listed there."""
    monkeypatch.setattr(scanner.titles, "in_strip", lambda img, corners: None)  # one read per strip
    img, n = column(7, tilt=True)
    jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    titles = iter([f"Card {k}" for k in range(1, 20)])
    strips = iter(range(100))

    def model(request):
        text = json.loads(request.content)["messages"][-1]["content"][0]["text"]
        if "top strip" in text:
            # One strip has no readable title (the table past the column's end): it is dropped.
            name = "" if next(strips) == 3 else next(titles)
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(
                {"name": name, "confidence": 0.9})}}]})
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
    assert names == sorted(["Front Card", *[f"Card {k}" for k in range(1, n - 1)]]), names
    raw = admin.get(f"/api/scans/{job_id}/debug").json()["images"][0]
    assert raw["columns"][0]["strips"] == n - 1 and raw["replaced_by_strips"] == 1
    assert raw["unreadable_strips"] == 1
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


def test_progress_reports_stages_and_counts():
    from deck_keeper.services import progress
    p = progress.start(99, photos=2)
    progress.photo(1)
    progress.stage("Reading title strips", 3)
    progress.tick()
    progress.tick()
    snap = progress.get(99)
    assert snap["photo"] == 1 and snap["photos"] == 2
    assert (snap["stage"]["name"], snap["stage"]["done"], snap["stage"]["total"]) == ("Reading title strips", 2, 3)
    progress.stage("Close-ups of each card and pile", 1)
    assert progress.get(99)["finished"][-1]["name"] == "Reading title strips"
    progress.finish()
    assert [t["stage"] for t in p.timings(1)] == ["Reading title strips", "Close-ups of each card and pile"]
    progress.end(99)
    assert progress.get(99) is None
    progress.tick()  # outside a scan: nothing happens


def test_scan_keeps_how_long_each_stage_took(admin):
    from .test_scan_flow import _jpeg
    vision.transport = httpx.MockTransport(lambda r: chat_reply([{"name": "Shock", "count": 1, "confidence": 0.9}]))
    scryfall.transport = httpx.MockTransport(lambda r: httpx.Response(404, json={}))
    admin.put("/api/admin/settings", json={"values": {"vision.model": "daily"}})
    job_id = admin.post("/api/scans", data={"prepass": "false"},
                        files=[("files", ("a.jpg", _jpeg(), "image/jpeg"))]).json()["id"]
    asyncio.run(scanner.process_job(job_id))
    timings = admin.get(f"/api/scans/{job_id}/debug").json()["images"][0]["timings"]
    stages = [t["stage"] for t in timings]
    assert stages[:2] == ["Finding card outlines", "Reading the whole photo"] and "Identifying cards" in stages
    assert admin.get(f"/api/scans/{job_id}").json()["progress"] is None  # done


def test_tiles_sized_from_the_cards():
    grid = tiles.plan(4904, 4088, None, (380, 530))  # scan 16: a straightened 100-card photo
    cols = len({t.x for t in grid})
    rows = len({t.y for t in grid})
    assert (cols, rows) == (5, 4) and max(t.w for t in grid) < 3.2 * 380 * 1.3
    assert len(tiles.plan(20000, 20000, None, (100, 140))) <= tiles.MAX_TILES


def test_commander_decks_allow_one_copy():
    assert scanner.copy_limit("Sol Ring", "100") == 1 and scanner.copy_limit("Forest", "100") is None
    assert scanner.copy_limit("Sol Ring", "60+15") == 4 and scanner.copy_limit("Seven Dwarves", "99") == 7


def test_deck_type_is_kept_and_can_be_changed(admin, pile_counts):
    from .test_scan_flow import _jpeg
    vision.transport = httpx.MockTransport(lambda r: chat_reply([{"name": "Lightning Bolt", "count": 2, "confidence": 0.9}]))
    scryfall.transport = httpx.MockTransport(lambda r: httpx.Response(404, json={}))
    admin.put("/api/admin/settings", json={"values": {"vision.model": "daily"}})
    assert admin.post("/api/scans", data={"deck_type": "nope"},
                      files=[("files", ("a.jpg", _jpeg(), "image/jpeg"))]).status_code == 400
    job_id = admin.post("/api/scans", data={"deck_type": "100", "prepass": "false"},
                        files=[("files", ("a.jpg", _jpeg(), "image/jpeg"))]).json()["id"]
    asyncio.run(scanner.process_job(job_id))
    scan = admin.get(f"/api/scans/{job_id}").json()
    [line] = scan["lines"]
    assert scan["deck_type"] == "100" and line["qty"] == 1 and "over_limit" in line["flags"]  # singleton
    assert admin.put(f"/api/scans/{job_id}/deck-type", json={"deck_type": "60"}).json()["deck_type"] == "60"


def test_many_title_bars_in_columns_skip_the_whole_photo_read(admin):
    img, n = column(18, step=30)
    jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    prompts = []

    def model(request):
        text = json.loads(request.content)["messages"][-1]["content"][0]["text"]
        prompts.append("whole" if "laid out on a table" in text and "one part" not in text else
                       "tile" if "one part" in text else "other")
        if "top strip" in text:
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"name": "Forest", "confidence": 0.9})}}]})
        if "close-up" in text:
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(
                {"name": "Forest", "type": "Basic Land — Forest", "count": 1, "confidence": 0.9})}}]})
        return chat_reply([])

    vision.transport = httpx.MockTransport(model)
    scryfall.transport = httpx.MockTransport(lambda r: httpx.Response(404, json={}))
    admin.put("/api/admin/settings", json={"values": {"vision.model": "daily"}})

    def scan(deck_type="other", photos=1):
        prompts.clear()
        job_id = admin.post("/api/scans", data={"deck_type": deck_type},
                            files=[("files", (f"{k}.jpg", jpeg, "image/jpeg")) for k in range(photos)]).json()["id"]
        asyncio.run(scanner.process_job(job_id))
        return admin.get(f"/api/scans/{job_id}").json(), admin.get(f"/api/scans/{job_id}/debug").json()["images"][0]

    # A 60-card deck in 4 photos: the 18 cards found here are about this photo's share, so the
    # titles in the columns and the outlined front card are all that's read.
    detail, raw = scan("60", photos=4)
    assert "whole" not in prompts and "tile" not in prompts
    assert raw["whole_photo"].startswith("skipped") and raw["columns"][0]["strips"] == n - 1
    assert sum(ln["qty"] for ln in detail["lines"]) == 4 * n  # each front card got its close-up

    # A 100-card deck in one photo: most of it wasn't found in columns, so tiles are read too.
    detail, raw = scan("100")
    assert "tile" in prompts and raw["tiled"] == f"the columns hold {n} of about 100 cards"

    # The setting reads the photo in tiles whenever there are columns.
    admin.put("/api/admin/settings", json={"values": {"vision.tiles_with_columns": True}})
    detail, raw = scan("60", photos=4)
    assert "whole" not in prompts and "tile" in prompts


def test_entry_next_to_a_strip_reading_the_same_name_is_one_card():
    """Scan 19: the tile listed the card behind each front card with its box over the front
    card's title, and the strip read it too."""
    strip = vision.ReadCard(name="Siege-Gang Commander", count=1, confidence=0.9, extra={"bbox": [1478, 1738, 446, 128]})
    tile = _card("Siege-Gang Commander", [1388, 1880, 519, 203])
    other = _card("Forgotten Cave", [1388, 1880, 519, 203])
    assert scanner._read_by_strip(tile, [strip], 675)
    assert not scanner._read_by_strip(other, [strip], 675)
    assert not scanner._read_by_strip(_card("Siege-Gang Commander", [1388, 2600, 519, 203]), [strip], 675)


def test_a_strip_is_read_from_its_title_bar_then_whole(tmp_path, monkeypatch):
    """The levelled title bar is read first; when it gives no name, the whole strip is."""
    img, _ = column(4, step=60)  # a whole title bar shows above the next card, as in real columns
    cols = cascades.find(img, [f.corners for f in detection.find_cards(img)])
    seen = []

    def model(request):
        content = json.loads(request.content)["messages"][-1]["content"]
        seen.append(len(content[1]["image_url"]["url"]))
        name = "" if len(seen) == 1 else "Lightning Bolt"
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"name": name, "confidence": 0.9})}}]})

    vision.transport = httpx.MockTransport(model)
    cols[0].strips = cols[0].strips[:1]
    cfg = vision.VisionConfig(base_url="http://x/v1", api_key="", model="m", timeout=5)
    out = asyncio.run(scanner._read_strips(img, cols, tmp_path, "p", cfg, asyncio.Semaphore(1)))
    assert [c.name for c in out] == ["Lightning Bolt"] and len(seen) == 2
    assert out[0].extra["strip"]["bar"] is not None  # the bar was found, and is the name box



def test_a_card_two_tiles_both_listed_counts_once(client):
    """Scan 16: two tiles listed Reap and Sow with boxes a card's step apart; the second close-up
    read the card below. Basic lands next to each other are separate cards."""
    from deck_keeper.db import session_scope
    from deck_keeper.models import Card, Detection

    with session_scope() as db:
        bolt = db.query(Card).filter_by(name="Lightning Bolt").one().id
        forest = db.query(Card).filter_by(name="Mountain").one().id

        def det(card_id, box, tile, used):
            return Detection(source="photo", bbox=box, raw_name="x", qty=1, card_id=card_id,
                             match_score=100, raw={"tile": tile, "used": used})

        a, b = det(bolt, [1121, 1514, 395, 80], 0, "zoom"), det(bolt, [1121, 1577, 395, 80], 1, "photo")
        far = det(bolt, [1121, 1900, 395, 80], 1, "zoom")  # a second copy further down the column
        f1, f2 = det(forest, [500, 1514, 395, 80], 0, "zoom"), det(forest, [500, 1577, 395, 80], 1, "zoom")
        assert scanner._tile_twins(db, [a, b, far, f1, f2]) == [b]


def test_type_lines_are_not_names():
    assert scanner._type_line("Land Creature — Forest Dryad") and scanner._type_line("Sorcery")
    assert scanner._type_line("Legendary Creature - Elf Druid")
    assert not scanner._type_line("Artifact Mutation") and not scanner._type_line("Natural Order")


def test_a_column_without_an_outlined_front_is_read_bar_by_bar(tmp_path, monkeypatch):
    """Scan 16: the model listed a column's cards, boxed roughly, but OpenCV couldn't outline its
    front card. Each title bar down the column is read on its own instead, and the model's
    entries it read again are replaced; a type line read off the front card is not a card."""
    img, n = column(7, step=60, drift=0)
    names = iter(["Sorcery"] + [f"Card {k}" for k in range(1, 40)])
    calls = []

    async def read_title(cfg, jpeg):
        calls.append(jpeg)
        return vision.ZoomRead(next(names), 1, 0.9, "{}")

    monkeypatch.setattr(vision, "read_title", read_title)
    fy = 1500 - 349 - 120
    entries = [vision.ReadCard(name=f"Card {k}", count=1, confidence=0.9,
                               extra={"bbox": [470, fy - (n - k) * 60 - 30, 270, 60]}) for k in range(1, n + 1)]
    reads, replaced = asyncio.run(scanner._read_bar_columns(img, entries, [], [], 250, 349, 0, tmp_path, "p", None,
                                                            asyncio.Semaphore(1)))
    assert len(calls) >= n and "Sorcery" not in [r.name for r in reads]
    assert all(r.extra["strip"]["bar"] and r.extra["strip"]["found"] == "bars" for r in reads)
    read_names = [r.name for r in reads]
    assert {c.name for c in entries if c.name in read_names} <= {c.name for c in replaced}


def test_time_estimate_for_a_new_scan_and_while_it_runs(admin):
    from deck_keeper.services import estimate, progress

    r = admin.get("/api/scans/estimate", params={"photos": 2, "deck_type": "100"}).json()
    assert r["cards"] == 100 and r["based_on"] == 0
    assert r["seconds"] == round(2 * estimate.DEFAULT[0] + 100 * estimate.DEFAULT[1])

    p = progress.Progress(1, photo=1, photos=1, estimate=100.0)
    p.started -= 30
    assert p.remaining(p.started + 30) == 70  # on track: the estimate less the time taken
    p.stage = progress.Stage("Reading title strips", total=100, done=10, started=p.started)
    assert p.remaining(p.started + 30) == 270 + progress.TAIL  # slower than estimated: the stage's own pace
    snap = p.snapshot()
    assert 0 < snap["fraction"] < 1 and snap["remaining"] >= snap["elapsed"]


def test_time_estimate_learns_from_finished_scans(monkeypatch):
    from deck_keeper.services import estimate

    # Scans taking 10 s a photo and 2 s a card.
    monkeypatch.setattr(estimate, "history", lambda db: [(1, 100, 210.0), (2, 60, 140.0), (1, 40, 90.0), (3, 120, 270.0)])
    per_photo, per_card, n = estimate.rates(None)
    assert n == 4 and abs(per_photo - 10) < 0.01 and abs(per_card - 2) < 0.01


def test_pile_counts_are_off_unless_turned_on(admin, monkeypatch):
    """By default every card found counts once, whatever count the model gives a pile; PILE_COUNTS
    turns pile counting back on."""
    from .test_scan_flow import _fake_model, _jpeg

    admin.put("/api/admin/settings", json={"values": {"vision.model": "daily"}})
    vision.transport = httpx.MockTransport(_fake_model)  # Lightning Bolf x4, Swiftspear x2, Mountain x20
    scryfall.transport = httpx.MockTransport(lambda r: httpx.Response(404, json={}))

    def quantities():
        job_id = admin.post("/api/scans", data={"prepass": "false"},
                            files=[("files", ("a.jpg", _jpeg(), "image/jpeg"))]).json()["id"]
        asyncio.run(scanner.process_job(job_id))
        lines = admin.get(f"/api/scans/{job_id}").json()["lines"]
        return {(ln["card"] or {}).get("name") or ln["raw_name"]: ln["qty"] for ln in lines}

    off = quantities()
    assert off["Lightning Bolt"] == 1 and off["Monastery Swiftspear"] == 1 and off["Mountain"] == 1

    monkeypatch.setenv("PILE_COUNTS", "true")
    on = quantities()
    assert on["Lightning Bolt"] == 4 and on["Monastery Swiftspear"] == 2
