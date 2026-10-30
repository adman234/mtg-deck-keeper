import json
import os
import tempfile

import httpx
import pytest

_tmp = tempfile.mkdtemp(prefix="deck-keeper-test-")
os.environ["DATA_DIR"] = _tmp
os.environ["SECRET_KEY"] = "test-secret-key-that-is-long-enough-for-hs256"
for k in list(os.environ):
    if k.startswith(("OIDC_", "VISION_")) or k in ("ALLOW_PASSWORD_LOGIN", "DATABASE_URL"):
        del os.environ[k]

from fastapi.testclient import TestClient  # noqa: E402

from deck_keeper.db import Base, get_engine, session_scope  # noqa: E402
from deck_keeper.main import create_app  # noqa: E402
from deck_keeper.models import Card  # noqa: E402
from deck_keeper.services import matching, scryfall, vision  # noqa: E402

CARD_NAMES = [
    "Lightning Bolt", "Monastery Swiftspear", "Bonecrusher Giant // Stomp", "Play with Fire",
    "Mountain", "Lightning Strike", "Delver of Secrets // Insectile Aberration", "Counterspell",
]


@pytest.fixture
def client():
    engine = get_engine()
    Base.metadata.drop_all(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP TABLE IF EXISTS alembic_version")
    app = create_app(background=False)
    with TestClient(app) as c:
        with session_scope() as db:
            db.add_all(Card(name=n) for n in CARD_NAMES)
        with session_scope() as db:
            matching.rebuild(db)
        yield c
    vision.transport = None
    scryfall.transport = None


@pytest.fixture
def admin(client):
    r = client.post("/api/auth/setup", json={"username": "admin", "email": "admin@example.com",
                                              "password": "password123"})
    assert r.status_code == 200, r.text
    return client


def chat_reply(cards: list[dict]) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"cards": cards})}}]})


@pytest.fixture
def pile_counts(monkeypatch):
    """Pile counting on (PILE_COUNTS=true): it is off by default."""
    monkeypatch.setenv("PILE_COUNTS", "true")
