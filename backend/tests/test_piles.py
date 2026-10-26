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
    """The Bolt box was drawn over the Counterspell pile, which the photo read listed separately."""
    bolt, counter = _card("Lightning Bolt"), _card("Counterspell")
    bolt.extra["bbox"], counter.extra["bbox"] = [100, 100, 300, 400], [180, 100, 300, 400]
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
    # The close-up area is the model's box plus room above, and for a pile (5 here) a quarter of
    # the box width on each side so the cards fanned out behind the front one are in view
    zx, zy, zw, zh = det["raw"]["zoom_region"]
    bx, by, bw, bh = det["bbox"]
    assert zy < by and zx + zw >= bx + bw
    assert abs(bx - zx - bw * piles.PAD_SIDE_PILE) <= 1 or zx == 0
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


def test_close_up_beats_a_made_up_name_even_when_the_card_is_elsewhere(client):
    """Scan 8: four separate Gaea's Cradles side by side. The whole-table read called them
    "Orchin Martyr"; each close-up read Gaea's Cradle. Boxes don't overlap, so trust the close-ups."""
    from deck_keeper.db import session_scope
    entries = [_card("Orchin Martyr", count=1) for _ in range(3)] + [_card("Counterspell", count=1)]
    for i, e in enumerate(entries):
        e.extra["bbox"] = [i * 340, 0, 330, 500]
        e.extra["zoom"] = vision.ZoomRead("Counterspell", 1, 0.95, "{}")
    with session_scope() as db:
        decisions = [piles.decide(matching.get_index(db), e, entries) for e in entries]
    assert [d.name for d in decisions] == ["Counterspell"] * 4


def test_disagreeing_counts_take_the_lower_one():
    c = _card("Misty Rainforest", count=2)
    c.extra["zoom"] = vision.ZoomRead("Misty Rainforest", 4, 0.9, "{}")
    from deck_keeper.services.matching import CardIndex
    d = piles.decide(CardIndex([(1, "Misty Rainforest")]), c, [c])
    assert (d.count, d.count_differs) == (2, True)


def test_no_title_in_close_up_leaves_the_name_unconfirmed():
    c = _card("Meticulous Artisan", count=1)
    c.extra["zoom"] = vision.ZoomRead("", 1, 1.0, "{}")
    from deck_keeper.services.matching import CardIndex
    d = piles.decide(CardIndex([(1, "Meticulous Artisan")]), c, [c])
    assert (d.name, d.used, d.unsure) == ("Meticulous Artisan", "photo", True)


def test_close_up_without_a_title_retries_taller(tmp_path):
    import numpy as np
    replies = iter([{"name": "", "count": 1, "confidence": 1}, {"name": "Forest", "count": 1, "confidence": 0.9}])
    sizes = []

    def model(request):
        import base64
        import cv2
        url = json.loads(request.content)["messages"][-1]["content"][1]["image_url"]["url"]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
        sizes.append(img.shape[0])
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(next(replies))}}]})

    vision.transport = httpx.MockTransport(model)
    try:
        card = _card("Meticulous Artisan", count=1)
        card.extra["bbox"] = [400, 600, 300, 400]
        img = np.full((1800, 2400, 3), 90, np.uint8)
        asyncio.run(piles.zoom_in(img, [card], tmp_path, "photo-001",
                                  vision.VisionConfig(base_url="http://x/v1", model="daily"), asyncio.Semaphore(1)))
    finally:
        vision.transport = None
    zoom = card.extra["zoom"]
    assert (zoom.name, zoom.retried) == ("Forest", True)
    assert sizes[1] > sizes[0]  # second crop reaches further up
    assert card.extra["crop_path"].endswith("-tall.jpg")
    assert card.extra["zoom_region"][1] < 600 - 400 * 0.5  # the taller crop's area is the one recorded


def test_close_up_is_not_told_the_whole_table_name():
    assert "thought this was" not in vision.ZOOM_PROMPT and "{hint}" not in vision.ZOOM_PROMPT


def test_scan_8_regression_made_up_row_names_fixed_by_close_ups(admin):
    """Four single Cradles in a row the whole-table read called "Orchin Martyr"."""
    def model(request):
        text = json.loads(request.content)["messages"][-1]["content"][0]["text"]
        if "close-up" in text:
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(
                {"name": "Counterspell", "count": 1, "confidence": 0.95})}}]})
        return chat_reply([{"name": "Orchin Martyr", "count": 1, "confidence": 1.0, "box": [x, 33, x + 140, 333]}
                           for x in (211, 363, 501, 635)])

    vision.transport = httpx.MockTransport(model)
    scryfall.transport = httpx.MockTransport(lambda r: httpx.Response(404, json={}))
    admin.put("/api/admin/settings", json={"values": {"vision.model": "daily"}})
    job_id = admin.post("/api/scans", data={"prepass": "false"},
                        files=[("files", ("a.jpg", _table_jpeg([(300, 300)]), "image/jpeg"))]).json()["id"]
    asyncio.run(scanner.process_job(job_id))
    lines = admin.get(f"/api/scans/{job_id}").json()["lines"]
    assert [(ln["card"]["name"], ln["qty"]) for ln in lines] == [("Counterspell", 4)]


def test_combine_counts():
    assert piles.combine_counts(4, 4) == (4, False)
    assert piles.combine_counts(5, 1) == (5, True)  # close-up only saw the front card
    assert piles.combine_counts(2, 4) == (2, True)  # close-up over-counted sleeves / next pile
    assert piles.combine_counts(4, 3) == (3, True)
    assert piles.combine_counts(1, 1) == (1, False)
    assert piles.combine_counts(3, None) == (3, False)


def test_single_card_close_up_stays_narrow_and_pile_reaches_sideways():
    single = piles.crop_region((2400, 1800), [1000, 500, 300, 400], pile=False)
    pile = piles.crop_region((2400, 1800), [1000, 500, 300, 400], pile=True)
    assert single[0] == 1000 - int(300 * piles.PAD_SIDE) and pile[0] == 1000 - int(300 * piles.PAD_SIDE_PILE)
    assert pile[2] > single[2] and pile[1] == single[1]


def test_prompts_say_title_is_the_top_line_and_box_the_whole_pile():
    for prompt in (vision.ZOOM_PROMPT, vision.SYSTEM_PROMPT):
        assert "TOP line" in prompt and "type line" in prompt and "Conjuro" in prompt
    assert "WHOLE pile" in vision.TABLE_PROMPT


def test_close_up_reading_the_type_line_loses_to_the_foreign_title(admin):
    """Scan 11: the close-up read "Conjuro" (Spanish for Sorcery), which fuzzy-matched Conquer;
    the whole-table read "Vislumbrar en la naturaleza" is Glimpse of Nature in Spanish."""
    from deck_keeper.db import session_scope
    from deck_keeper.models import Card
    with session_scope() as db:
        db.add_all([Card(name="Glimpse of Nature"), Card(name="Conquer")])
    with session_scope() as db:
        matching.rebuild(db)

    def model(request):
        text = json.loads(request.content)["messages"][-1]["content"][0]["text"]
        if "close-up" in text:
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(
                {"name": "Conjuro", "count": 1, "confidence": 1.0})}}]})
        if "Which of these" in text:  # candidate re-ask: the model happily picks one
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(
                {"name": "Conquer", "confidence": 0.9})}}]})
        return chat_reply([{"name": "Vislumbrar en la naturaleza", "count": 4, "confidence": 1,
                            "box": [130, 160, 250, 420]}])

    def scry(request):
        if request.url.path == "/cards/search" and "naturaleza" in request.url.params["q"] and \
                not request.url.params["q"].startswith('"'):
            return httpx.Response(200, json={"data": [
                {"name": "Glimpse of Nature", "printed_name": "Vislumbrar la naturaleza", "lang": "es"}]})
        return httpx.Response(404, json={})

    vision.transport = httpx.MockTransport(model)
    scryfall.transport = httpx.MockTransport(scry)
    admin.put("/api/admin/settings", json={"values": {"vision.model": "daily"}})
    job_id = admin.post("/api/scans", files=[("files", ("a.jpg", _table_jpeg([(300, 300)]), "image/jpeg"))]).json()["id"]
    asyncio.run(scanner.process_job(job_id))
    [line] = admin.get(f"/api/scans/{job_id}").json()["lines"]
    assert (line["card"]["name"], line["qty"]) == ("Glimpse of Nature", 4)  # close-up's 1 = front card only
    raw = admin.get(f"/api/scans/{job_id}/debug").json()["images"][0]["detections"][0]["raw"]
    assert raw["both_names"]["kept"] == "Vislumbrar en la naturaleza"


def _count_prompts(seen):
    def model(request):
        text = json.loads(request.content)["messages"][-1]["content"][0]["text"]
        kind = "crop" if "single Magic card" in text else "zoom" if "close-up" in text else "photo"
        seen.append(kind)
        if kind == "crop":
            return chat_reply([{"name": "Shock", "count": 1, "confidence": 0.95}])
        if kind == "zoom":
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(
                {"name": "Shock", "count": 1, "confidence": 0.95})}}]})
        return seen.photo_reply
    return model


class _Seen(list):
    photo_reply = None


def test_pre_pass_skipped_when_the_model_gives_boxes(admin):
    seen = _Seen()
    seen.photo_reply = chat_reply([{"name": "Shock", "count": 1, "confidence": 0.9, "box": [130, 160, 250, 420]}])
    vision.transport = httpx.MockTransport(_count_prompts(seen))
    scryfall.transport = httpx.MockTransport(lambda r: httpx.Response(404, json={}))
    admin.put("/api/admin/settings", json={"values": {"vision.model": "daily"}})
    job_id = admin.post("/api/scans", files=[("files", ("a.jpg", _table_jpeg([(300, 300), (700, 300)]), "image/jpeg"))]).json()["id"]
    asyncio.run(scanner.process_job(job_id))
    # The listed card gets a close-up, and so does the card OpenCV outlined that the model missed.
    assert "crop" not in seen and seen.count("zoom") == 2
    raw = admin.get(f"/api/scans/{job_id}/debug").json()["images"][0]
    assert raw["crops_found"] == 0 and "prepass" not in raw and raw["snap"]["unlisted"] == 1


def test_pre_pass_runs_when_the_model_gives_no_boxes(admin):
    seen = _Seen()
    seen.photo_reply = chat_reply([{"name": "Shock", "count": 2, "confidence": 0.9}])
    vision.transport = httpx.MockTransport(_count_prompts(seen))
    scryfall.transport = httpx.MockTransport(lambda r: httpx.Response(404, json={}))
    admin.put("/api/admin/settings", json={"values": {"vision.model": "daily"}})
    job_id = admin.post("/api/scans", files=[("files", ("a.jpg", _table_jpeg([(300, 300), (700, 300)]), "image/jpeg"))]).json()["id"]
    asyncio.run(scanner.process_job(job_id))
    assert seen.count("crop") == 2
    raw = admin.get(f"/api/scans/{job_id}/debug").json()["images"][0]
    assert raw["crops_found"] == 2 and raw["prepass"] == "no usable boxes from the model"


# Scan 12, lands photo (2000 px wide): the model's boxes and the card outlines OpenCV found.
LANDS_FOUND = [[413, 64, 297, 364], [701, 65, 287, 363], [989, 69, 281, 361], [1272, 70, 284, 362],
               [986, 430, 294, 402], [1592, 944, 318, 417]]


def _lands_entries():
    rows = [("Ota Halcs", 1, [436, 50, 270, 423]), ("Ota Halcs", 1, [726, 50, 274, 423]),
            ("Ota Halcs", 1, [1002, 50, 274, 423]), ("Ota Halcs", 1, [1276, 50, 274, 423]),
            ("Misty Rainforest", 1, [422, 509, 284, 377]), ("Misty Rainforest", 1, [706, 509, 284, 377]),
            ("Misty Rainforest", 1, [990, 509, 276, 377]), ("Misty Rainforest", 1, [1266, 509, 300, 377]),
            ("Windswept Heath", 2, [150, 901, 320, 218]), ("Dryad Arbor", 2, [1516, 901, 370, 218])]
    out = []
    for name, count, box in rows:
        c = _card(name, count=count)
        c.extra["bbox"] = box
        out.append(c)
    return out


def test_scan_12_boxes_are_lined_up_with_the_cards_opencv_found():
    """The second row's boxes sat ~80 px below the cards. One Bayou there was found by OpenCV, so
    the other three move up by the same amount; the piles keep their own boxes."""
    entries = _lands_entries()
    info = piles.snap_boxes(entries, LANDS_FOUND)
    assert info["snapped"] == 6 and info["shifted"] == 3 and info["missed"] == []
    cradle, forest, bayou, heath, dryad = entries[0], entries[4], entries[6], entries[8], entries[9]
    assert cradle.extra["bbox"] == LANDS_FOUND[0] and cradle.extra["model_bbox"] == [436, 50, 270, 423]
    assert bayou.extra["bbox"] == LANDS_FOUND[4] and bayou.extra["snapped"]
    assert not forest.extra.get("snapped") and forest.extra["shifted"][1] == -79
    assert abs(forest.extra["bbox"][1] - 430) <= 1  # level with the Bayou it sits beside
    assert "model_bbox" not in heath.extra and heath.extra["bbox"] == [150, 901, 320, 218]
    assert dryad.extra["bbox"] == [1516, 901, 394, 460]  # pile widened to take in its front card


def test_one_found_card_moves_its_own_row_only():
    entries = _lands_entries()
    info = piles.snap_boxes(entries, LANDS_FOUND[4:5])
    assert info["matched_by"] == "overlap"  # 3 rows of boxes, 1 row of found cards
    assert entries[6].extra["model_bbox"] == [990, 509, 276, 377]
    assert [e.extra.get("shift_source") for e in entries[4:8]] == ["row", "row", None, "row"]
    assert not any(e.extra.get("shifted") for e in entries[:4])


def test_scan_13_squashed_rows_are_matched_row_by_row():
    """Scan 13, 15 sleeved singles in 3 rows: the model's rows were squashed into the middle of
    the photo, so its third row overlapped the real second row more than the third."""
    found = [[1014, 184, 387, 498], [1779, 204, 369, 472], [1021, 707, 375, 507], [1414, 699, 399, 521],
             [1800, 703, 396, 517], [1034, 1260, 395, 568], [619, 1269, 375, 546], [1846, 1254, 400, 548]]
    xs = [(225, 387), (619, 391), (1017, 391), (1413, 324), (1744, 394)]
    entries = []
    for y in (352, 630, 910):
        for x, w in xs:
            c = _card("x", count=1)
            c.extra["bbox"] = [x, y, w, 259]
            entries.append(c)
    info = piles.snap_boxes(entries, found)
    assert info["matched_by"] == "rows" and info["snapped"] == 8 and info["missed"] == []
    assert entries[12].extra["bbox"] == [1034, 1260, 395, 568]  # row 3 box on the row 3 card
    for row, (top, bottom) in enumerate([(184, 682), (699, 1220), (1254, 1828)]):
        for e in entries[row * 5:row * 5 + 5]:
            y, h = e.extra["bbox"][1], e.extra["bbox"][3]
            assert top - 15 <= y and y + h <= bottom + 15, (row, e.extra)
    assert entries[0].extra["shift_source"] == "row" and piles.pads(entries[0]) == (piles.PAD_TOP, piles.PAD_BOTTOM)


def test_row_with_no_found_card_is_estimated_from_the_others():
    found = [[100, 100, 200, 280], [400, 700, 200, 280]]
    entries = []
    for y in (150, 450, 750):
        c = _card("x", count=1)
        c.extra["bbox"] = [100, y, 200, 200]
        entries.append(c)
    entries[2].extra["bbox"] = [400, 750, 200, 200]
    piles.snap_boxes(entries, found)
    middle = entries[1]
    assert middle.extra["shift_source"] == "estimate" and middle.extra["bbox"][1] == 400
    assert piles.pads(middle) == (piles.PAD_TOP_LOOSE, -piles.TRIM_BOTTOM_LOOSE)


def test_close_up_of_a_loose_box_reaches_up_and_leaves_out_the_next_title():
    loose, snapped = _card("Choke", count=1), _card("Choke", count=1)
    snapped.extra["snapped"] = True
    box = [1740, 352, 415, 461]  # scan 12: a third of a card below the real card
    lx, ly, lw, lh = piles.crop_region((2400, 1946), box, *piles.pads(loose)[:1], bottom=piles.pads(loose)[1])
    sx, sy, sw, sh = piles.crop_region((2400, 1946), box, *piles.pads(snapped)[:1], bottom=piles.pads(snapped)[1])
    assert ly < 352 - 461 * 0.4 and ly + lh < 352 + 461 * 0.75  # title above in, next row's title out
    assert sy > 352 - 461 * 0.1 and sy + sh > 352 + 461


def test_repeated_whole_table_name_gives_way_to_the_close_ups():
    """Scan 12: the whole-table read called a Forest and three Bayous "Misty Rainforest"."""
    from deck_keeper.services.matching import CardIndex
    index = CardIndex([(1, "Misty Rainforest"), (2, "Forest"), (3, "Bayou"), (4, "Gaea's Cradle")])
    entries = [_card("Misty Rainforest", count=1) for _ in range(4)]
    for e, (i, z) in zip(entries, enumerate(["Forest", "Horizon s Edge", "Bayou", "Bayou"])):
        e.extra["bbox"] = [i * 300, 500, 290, 400]
        e.extra["zoom"] = vision.ZoomRead(z, 1, 0.9, "{}")
    d = [piles.decide(index, e, entries) for e in entries]
    assert [x.name for x in d] == ["Forest", "Horizon s Edge", "Bayou", "Bayou"]
    assert d[1].unsure and d[1].repeated and not d[0].unsure
    assert d[1].suggest == ["Forest", "Bayou"]  # what the rest of the row was read as


def test_close_up_reading_the_card_below_keeps_the_whole_table_name():
    """Scan 12: Craterhoof's box sat low and the close-up read Once Upon a Time, the card below."""
    from deck_keeper.services.matching import CardIndex
    index = CardIndex([(1, "Craterhoof Behemoth"), (2, "Once Upon a Time"), (3, "Boseiju, Who Endures")])
    hoof, once, boseiju = _card("Craterhoof Behemoth", 1), _card("Once Upon a Time", 1), _card("Boseiju, Who Endures", 1)
    hoof.extra["bbox"], once.extra["bbox"], boseiju.extra["bbox"] = [634, 466, 439, 506], [670, 986, 439, 511], [152, 986, 506, 511]
    hoof.extra["zoom"] = vision.ZoomRead("Once Upon a Time", 1, 0.9, "{}")
    d = piles.decide(index, hoof, [hoof, once, boseiju])
    assert (d.name, d.used, d.unsure) == ("Craterhoof Behemoth", "photo", True)


def test_two_real_names_are_settled_by_asking_the_model(tmp_path):
    """Scan 12: whole table "Force of Vigor" (right), close-up "Nature's Claim" (read from the
    rules text when the box missed the title). The model is shown a taller close-up and asked."""
    import numpy as np
    from deck_keeper.services.matching import CardIndex
    index = CardIndex([(1, "Force of Vigor"), (2, "Nature's Claim")])
    asked = []

    def model(request):
        body = json.loads(request.content)
        asked.append(body["response_format"]["json_schema"]["schema"]["properties"]["name"]["enum"])
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(
            {"name": "Force of Vigor", "confidence": 0.9})}}]})

    card = _card("Force of Vigor", count=1)
    card.extra["bbox"] = [626, 352, 384, 461]
    card.extra["zoom"] = vision.ZoomRead("Nature's Claim", 1, 0.9, "{}")
    assert piles.decide(index, card, [card]).unsure  # undecided without asking
    vision.transport = httpx.MockTransport(model)
    try:
        asyncio.run(piles.settle(np.full((1946, 2400, 3), 90, np.uint8), [card], index,
                                 vision.VisionConfig(base_url="http://x/v1", model="daily"), asyncio.Semaphore(1)))
    finally:
        vision.transport = None
    assert asked == [["Force of Vigor", "Nature's Claim", "none"]]
    d = piles.decide(index, card, [card])
    assert (d.name, d.used, d.unsure) == ("Force of Vigor", "photo", False)


def test_agreeing_reads_are_not_asked_again():
    import numpy as np
    from deck_keeper.services.matching import CardIndex
    card = _card("Thoughtize", count=1)
    card.extra["bbox"] = [1394, 352, 346, 461]
    card.extra["zoom"] = vision.ZoomRead("Thoughtseize", 1, 0.9, "{}")
    vision.transport = httpx.MockTransport(lambda r: (_ for _ in ()).throw(AssertionError("asked")))
    try:
        asyncio.run(piles.settle(np.full((1946, 2400, 3), 90, np.uint8), [card], CardIndex([(1, "Thoughtseize")]),
                                 vision.VisionConfig(base_url="http://x/v1", model="daily"), asyncio.Semaphore(1)))
    finally:
        vision.transport = None
    assert "tiebreak" not in card.extra


def test_model_unable_to_choose_keeps_the_whole_table_name():
    """Scan 13: whole table "Abrupt Decay" (right); the close-up, still over the card above, read
    "Naturalize". Asked to choose, the model said neither, so the close-up wasn't of this card."""
    from deck_keeper.services.matching import CardIndex
    index = CardIndex([(1, "Abrupt Decay"), (2, "Naturalize")])
    card = _card("Abrupt Decay", count=1)
    card.extra["bbox"] = [602, 911, 394, 261]
    card.extra["zoom"] = vision.ZoomRead("Naturalize", 1, 1.0, "{}")
    card.extra["tiebreak"] = {"options": ["Abrupt Decay", "Naturalize"], "name": "none", "confidence": 0.5}
    d = piles.decide(index, card, [card])
    assert (d.name, d.used, d.unsure) == ("Abrupt Decay", "photo", True)


def test_repeated_name_with_unreadable_close_up_is_left_to_the_user(admin):
    """Scan 13: four "Meticulous Artisan" for a Forest and three Bayous. One close-up misread the
    Forest as "Force of One", which is far from any card: the line is left unmatched, with
    Forest and Bayou offered first, and never falls back to the repeated name."""
    from deck_keeper.db import session_scope
    from deck_keeper.models import Card
    with session_scope() as db:
        for name in ("Meticulous Artisan", "Forest", "Bayou", "Force of Will", "Force of Vigor"):
            db.add(Card(name=name))
        db.flush()
        matching.rebuild(db)
    reads = iter(["Force of One", "Bayou", "Bayou", "Bayou"])

    def model(request):
        text = json.loads(request.content)["messages"][-1]["content"][0]["text"]
        if "close-up" in text:
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(
                {"name": next(reads), "count": 1, "confidence": 1})}}]})
        if "Candidates" in text:
            raise AssertionError("the model must not be asked to choose here")
        return chat_reply([{"name": "Meticulous Artisan", "count": 1, "confidence": 1, "box": [x, 300, x + 150, 600]}
                           for x in (100, 300, 500, 700)])

    vision.transport = httpx.MockTransport(model)
    scryfall.transport = httpx.MockTransport(lambda r: httpx.Response(404, json={}))
    admin.put("/api/admin/settings", json={"values": {"vision.model": "daily"}})
    job_id = admin.post("/api/scans", files=[("files", ("a.jpg", _table_jpeg([(300, 300)]), "image/jpeg"))]).json()["id"]
    asyncio.run(scanner.process_job(job_id))
    lines = admin.get(f"/api/scans/{job_id}").json()["lines"]
    unmatched = [ln for ln in lines if ln["card"] is None]
    assert len(unmatched) == 1 and "unmatched" in unmatched[0]["flags"]
    assert [s["name"] for s in unmatched[0]["suggestions"]][:1] == ["Bayou"]
    assert not any(ln["card"] and ln["card"]["name"] == "Meticulous Artisan" for ln in lines)


def test_stuck_model_repeating_one_name_is_cut_off():
    """Scan 15 (a 100-card photo): the reply ended in ~70 "Coral Reef", some without boxes, the
    rest stepping diagonally across the photo and off its edge."""
    real = [_card(n, count=1) for n in ("Scapeshift", "Edge of Autumn", "Coral Reef")]
    for i, c in enumerate(real):
        c.extra["bbox"] = [100, 100 + i * 120, 320, 110]
    stuck = []
    for i in range(30):
        c = _card("Coral Reef", count=1)
        if i < 12:
            x = 1248 + i * 82
            c.extra["bbox"] = [x, 734 + i * 47, min(312, 2400 - x), 181]
        stuck.append(c)
    cards = real + stuck
    out = piles.drop_runaway(cards)
    assert out == {"name": "Coral Reef", "dropped": 30}  # the run includes the real one: it stays
    assert [c.name for c in cards] == ["Scapeshift", "Edge of Autumn", "Coral Reef"]


def test_a_row_of_basic_lands_is_not_a_runaway():
    row = [_card("Forest", count=1) for _ in range(8)]
    for i, c in enumerate(row):
        c.extra["bbox"] = [100 + i * 260, 300, 250, 350]
    assert piles.drop_runaway(row) is None and len(row) == 8


def test_boxes_are_given_a_cards_size():
    """Lands photo: the fetchland piles were boxed to their top half."""
    found = [[413, 64, 297, 364], [701, 65, 287, 363], [989, 69, 281, 361], [986, 430, 294, 402]]
    pile, single, fine = _card("Windswept Heath", count=2), _card("Forest", count=1), _card("Bayou", count=1)
    pile.extra["bbox"], single.extra["bbox"], fine.extra["bbox"] = [150, 900, 320, 180], [700, 430, 500, 600], [1272, 430, 290, 370]
    assert piles.fit_sizes([pile, single, fine], found, (2000, 1506)) == 2
    assert pile.extra["bbox"][1] == 900 and pile.extra["bbox"][3] >= 363  # grown down to a card
    assert pile.extra["model_bbox"] == [150, 900, 320, 180]
    x, y, w, h = single.extra["bbox"]
    assert abs(w - 290) <= 1 and abs(h - 363) <= 1  # a typical card's size...
    assert abs(x + w / 2 - 950) <= 1 and abs(y + h / 2 - 730) <= 1  # ...around the same centre
    assert "resized" not in fine.extra
