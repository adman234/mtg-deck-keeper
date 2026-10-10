from deck_keeper import app_settings
from deck_keeper.db import session_scope
from deck_keeper.routes.oidc import OIDCError, resolve_user

import pytest


def test_first_run_setup_then_login(client):
    assert client.get("/api/auth/status").json()["setup_required"] is True
    assert client.get("/api/decks").status_code == 401
    client.post("/api/auth/setup", json={"username": "admin", "email": "a@example.com", "password": "password123"})
    assert client.post("/api/auth/setup", json={"username": "xx", "email": "x@example.com",
                                                "password": "password123"}).status_code == 409
    client.post("/api/auth/logout")
    assert client.post("/api/auth/login", json={"username": "admin", "password": "nope"}).status_code == 401
    r = client.post("/api/auth/login", json={"username": "A@example.com", "password": "password123"})
    assert r.status_code == 200 and r.json()["is_admin"]


def _cfg(**over):
    with session_scope() as db:
        cfg = app_settings.get_group(db, "oidc")
    return {**cfg, **over}


def test_oidc_creates_user_and_applies_admin_group(admin):
    cfg = _cfg(user_group="mtg", admin_group="mtg-admins")
    with session_scope() as db:
        user = resolve_user(db, cfg, {"email": "Pat@Example.com", "preferred_username": "pat",
                                      "groups": ["mtg", "mtg-admins"]})
        assert (user.username, user.email, user.is_admin, user.auth_method) == ("pat", "pat@example.com", True, "oidc")
        # Removed from the admin group upstream -> loses admin on next login
        user = resolve_user(db, cfg, {"email": "pat@example.com", "groups": ["mtg"]})
        assert user.is_admin is False


def test_oidc_group_gate_and_signup_off(admin):
    with session_scope() as db:
        with pytest.raises(OIDCError, match="not in the group"):
            resolve_user(db, _cfg(user_group="mtg"), {"email": "x@example.com", "groups": ["other"]})
        with pytest.raises(OIDCError, match="sign-up is turned off"):
            resolve_user(db, _cfg(signup_enabled=False), {"email": "new@example.com"})


def test_oidc_links_existing_local_user_by_email(admin):
    with session_scope() as db:
        user = resolve_user(db, _cfg(), {"email": "admin@example.com"})
        assert user.username == "admin"
