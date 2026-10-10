import asyncio

import httpx

from deck_keeper.db import session_scope
from deck_keeper.services import matching, scryfall


def test_sync_adds_only_new_names(client):
    scryfall.transport = httpx.MockTransport(lambda r: httpx.Response(200, json={
        "data": ["Lightning Bolt", "Counterspell", "Brand New Card", "Another New One"]}))
    with session_scope() as db:
        result = asyncio.run(scryfall.sync_card_names(db))
        assert result["added"] == 2 and set(result["added_sample"]) == {"Brand New Card", "Another New One"}
        assert matching.get_index(db).match("Brand New Card")[0].score == 100
        again = asyncio.run(scryfall.sync_card_names(db))
        assert again["added"] == 0
