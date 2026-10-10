import time
from urllib.parse import parse_qs, urlparse

import httpx
from joserfc import jwt
from joserfc.jwk import RSAKey

from deck_keeper.routes import oidc

ISSUER = "https://auth.example.com/application/o/deck-keeper/"
KEY = RSAKey.generate_key(2048, parameters={"kid": "k1"})


def _provider(nonce_holder: dict, groups: list[str]):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/.well-known/openid-configuration"):
            return httpx.Response(200, json={
                "issuer": ISSUER, "authorization_endpoint": ISSUER + "authorize",
                "token_endpoint": ISSUER + "token", "jwks_uri": ISSUER + "jwks",
                "userinfo_endpoint": ISSUER + "userinfo"})
        if url.endswith("/jwks"):
            return httpx.Response(200, json={"keys": [KEY.as_dict(private=False)]})
        if url.endswith("/token"):
            assert b"code_verifier=" in request.content and b"code=abc" in request.content
            now = int(time.time())
            id_token = jwt.encode({"alg": "RS256", "kid": "k1"}, {
                "iss": ISSUER, "aud": "deck-keeper", "sub": "123", "iat": now, "exp": now + 300,
                "nonce": nonce_holder["nonce"], "email": "sam@example.com", "preferred_username": "sam",
            }, KEY)
            return httpx.Response(200, json={"access_token": "at", "id_token": id_token, "token_type": "Bearer"})
        if url.endswith("/userinfo"):
            return httpx.Response(200, json={"sub": "123", "groups": groups})
        return httpx.Response(404)
    return handler


def _configure(admin):
    r = admin.put("/api/admin/settings", json={"values": {
        "oidc.enabled": True, "oidc.configuration_url": ISSUER + ".well-known/openid-configuration",
        "oidc.client_id": "deck-keeper", "oidc.client_secret": "s3cret",
        "oidc.user_group": "mtg", "oidc.admin_group": "mtg-admins", "oidc.provider_name": "Authentik"}})
    assert r.status_code == 200, r.text
    admin.post("/api/auth/logout")


def _login(client, holder):
    r = client.get("/api/auth/oidc/login", follow_redirects=False)
    assert r.status_code == 302, r.text
    q = parse_qs(urlparse(r.headers["location"]).query)
    assert q["code_challenge_method"] == ["S256"] and q["redirect_uri"][0].endswith("/api/auth/oidc/callback")
    holder["nonce"] = q["nonce"][0]
    return client.get("/api/auth/oidc/callback", params={"code": "abc", "state": q["state"][0]},
                      follow_redirects=False)


def test_full_oidc_login(admin):
    _configure(admin)
    holder: dict = {}
    oidc.transport = httpx.MockTransport(_provider(holder, ["mtg"]))
    oidc._discovery_cache.clear()
    try:
        status = admin.get("/api/auth/status").json()
        assert status["oidc_enabled"] and status["oidc_provider_name"] == "Authentik"
        r = _login(admin, holder)
        assert r.status_code == 302 and r.headers["location"] == "/"
        me = admin.get("/api/auth/me").json()
        assert (me["username"], me["auth_method"], me["is_admin"]) == ("sam", "oidc", False)
    finally:
        oidc.transport = None


def test_oidc_rejects_user_outside_group(admin):
    _configure(admin)
    holder: dict = {}
    oidc.transport = httpx.MockTransport(_provider(holder, ["someone-else"]))
    oidc._discovery_cache.clear()
    try:
        r = _login(admin, holder)
        assert r.status_code == 302 and r.headers["location"].startswith("/login?error=")
        assert admin.get("/api/auth/me").status_code == 401
    finally:
        oidc.transport = None


def test_callback_with_wrong_state_is_refused(admin):
    _configure(admin)
    r = admin.get("/api/auth/oidc/callback", params={"code": "abc", "state": "forged"}, follow_redirects=False)
    assert r.headers["location"].startswith("/login?error=")
