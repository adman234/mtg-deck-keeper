"""Scoring scans against a real decklist."""
import asyncio
import io
import zipfile

import httpx

from deck_keeper.services import decklists, matching, scanner, scryfall, vision

from .conftest import chat_reply
from .test_scan_flow import _jpeg

TITANIA = """1 Titania, Protector of Argoth

1 Aftermath Analyst
1 Azusa, Lost but Seeking

1 Beast Within
1 Sol Ring (CMM) 395
SB: 1 Lightning Bolt

11 Forest
"""


def _index():
    return matching.CardIndex([(i, n) for i, n in enumerate(
        ["Titania, Protector of Argoth", "Aftermath Analyst", "Azusa, Lost but Seeking", "Beast Within",
         "Sol Ring", "Lightning Bolt", "Forest"], start=1)])


def test_parse_decklist_with_blank_lines_between_groups():
    entries = decklists.parse(TITANIA, _index())
    assert [(e.qty, e.card_name, e.board) for e in entries] == [
        (1, "Titania, Protector of Argoth", "main"), (1, "Aftermath Analyst", "main"),
        (1, "Azusa, Lost but Seeking", "main"), (1, "Beast Within", "main"), (1, "Sol Ring", "main"),
        (1, "Lightning Bolt", "side"), (11, "Forest", "main")]


def test_parse_sections_and_unknown_names():
    entries = decklists.parse("Deck\n4 Lightning Bolt\n2x Forest\nSideboard\n1 Not A Real Card\n", _index())
    assert [(e.qty, e.card_name, e.board) for e in entries] == [(4, "Lightning Bolt", "main"), (2, "Forest", "main"),
                                                              (1, None, "side")]


def test_score():
    expected = [{"card_id": 1, "name": "Titania", "qty": 1}, {"card_id": 7, "name": "Forest", "qty": 11}]
    got = [{"card_id": 7, "name": "Forest", "qty": 9}, {"card_id": None, "name": "Gate's Crackle", "qty": 1},
           {"card_id": 5, "name": "Sol Ring", "qty": 1}]
    s = decklists.score(expected, got)
    assert (s["right"], s["total"]) == (9, 12)
    assert s["missing"] == [["Forest", 2], ["Titania", 1]] and s["extra"] == [["Gate's Crackle", 1], ["Sol Ring", 1]]


def test_score_counts_either_face_of_a_two_faced_card():
    expected = [{"name": "Naktamun Lorespinner // Wheel of Fortune", "qty": 1}]
    s = decklists.score(expected, [{"name": "Wheel of Fortune", "qty": 1}])
    assert s["right"] == 1 and s["extra"] == []


def test_scan_scored_against_the_real_list_on_every_read(admin, pile_counts):
    admin.put("/api/admin/settings", json={"values": {"vision.model": "daily"}})
    vision.transport = httpx.MockTransport(lambda r: chat_reply([
        {"name": "Lightning Bolt", "count": 3, "confidence": 0.95}, {"name": "Shock", "count": 1, "confidence": 0.9}]))
    scryfall.transport = httpx.MockTransport(lambda r: httpx.Response(404, json={}))
    job_id = admin.post("/api/scans", data={"prepass": "false"},
                        files=[("files", ("a.jpg", _jpeg(), "image/jpeg"))]).json()["id"]
    asyncio.run(scanner.process_job(job_id))
    scan = admin.put(f"/api/scans/{job_id}/expected", json={"text": "4 Lightning Bolt\n2 Counterspell"}).json()
    assert scan["test"]["now"]["right"] == 3 and scan["test"]["now"]["total"] == 6 and scan["test"]["history"] == []

    admin.post(f"/api/scans/{job_id}/rerun", json={})
    asyncio.run(scanner.process_job(job_id))
    test = admin.get(f"/api/scans/{job_id}").json()["test"]
    [run] = test["history"]
    assert (run["right"], run["total"]) == (3, 6) and run["missing"] == [["Counterspell", 2], ["Lightning Bolt", 1]]
    assert run["extra"] == [["Shock", 1]] and run["build"]

    z = zipfile.ZipFile(io.BytesIO(admin.get(f"/api/scans/{job_id}/test-files").content))
    assert sorted(z.namelist()) == ["expected.txt", "photo-1.jpg", "scan.json"]
    assert "4 Lightning Bolt" in z.read("expected.txt").decode()

    assert admin.put(f"/api/scans/{job_id}/expected", json={"text": "\n\n"}).status_code == 400
    assert admin.delete(f"/api/scans/{job_id}/expected").json()["test"] is None
