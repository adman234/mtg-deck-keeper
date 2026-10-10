"""OpenID Connect login (authorization code + PKCE), configured from env vars or the GUI.

Behaviour follows Mealie: users are matched by OIDC_USER_CLAIM, optionally created on
first login, gated by OIDC_USER_GROUP, and made admin by OIDC_ADMIN_GROUP.
"""

import base64
import hashlib
import logging
import re
import secrets
import time
from urllib.parse import urlencode

import httpx
from joserfc import jwt
from joserfc.errors import JoseError
from joserfc.jwk import KeySet
from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import app_settings
from ..config import get_env
from ..db import get_db
from ..models import User
from ..security import create_token, decode_token
from .auth import default_group, set_session

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth/oidc", tags=["auth"])

STATE_COOKIE = "dk_oidc"
DEFAULT_SCOPES = "openid profile email groups"
_discovery_cache: dict[str, tuple[float, dict]] = {}

# Tests swap this for an httpx.MockTransport.
transport: httpx.AsyncBaseTransport | None = None


class OIDCError(Exception):
    pass


def _http() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=15, transport=transport, follow_redirects=True)


async def discover(url: str, force: bool = False) -> dict:
    cached = _discovery_cache.get(url)
    if cached and not force and time.time() - cached[0] < 3600:
        return cached[1]
    async with _http() as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
    for key in ("issuer", "authorization_endpoint", "token_endpoint", "jwks_uri"):
        if key not in data:
            raise OIDCError(f"Discovery document is missing {key}")
    _discovery_cache[url] = (time.time(), data)
    return data


def redirect_uri(request: Request) -> str:
    base = get_env().base_url.rstrip("/") or str(request.base_url).rstrip("/")
    return f"{base}/api/auth/oidc/callback"


def _fail(message: str) -> RedirectResponse:
    resp = RedirectResponse("/login?" + urlencode({"error": message, "direct": "1"}), status_code=302)
    resp.delete_cookie(STATE_COOKIE, path="/api/auth/oidc")
    return resp


@router.get("/login")
async def login(request: Request, db: Session = Depends(get_db)):
    cfg = app_settings.get_group(db, "oidc")
    if not (cfg["enabled"] and cfg["configuration_url"] and cfg["client_id"]):
        return _fail("Single sign-on is not configured")
    try:
        meta = await discover(cfg["configuration_url"])
    except (httpx.HTTPError, ValueError, OIDCError) as e:
        log.warning("OIDC discovery failed: %s", e)
        return _fail("Could not reach the sign-on provider")
    state, nonce, verifier = (secrets.token_urlsafe(24) for _ in range(3))
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    params = {
        "response_type": "code",
        "client_id": cfg["client_id"],
        "redirect_uri": redirect_uri(request),
        "scope": cfg["scopes_override"] or DEFAULT_SCOPES,
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    resp = RedirectResponse(meta["authorization_endpoint"] + "?" + urlencode(params), status_code=302)
    cookie = create_token({"typ": "oidc", "state": state, "nonce": nonce, "verifier": verifier}, hours=0.25)
    resp.set_cookie(STATE_COOKIE, cookie, max_age=900, httponly=True, samesite="lax",
                    secure=request.url.scheme == "https", path="/api/auth/oidc")
    return resp


async def _exchange(cfg: dict, meta: dict, code: str, verifier: str, request: Request) -> dict:
    data = {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri(request),
            "code_verifier": verifier, "client_id": cfg["client_id"]}
    methods = meta.get("token_endpoint_auth_methods_supported") or ["client_secret_basic"]
    auth = None
    if cfg["client_secret"]:
        if "client_secret_post" in methods and "client_secret_basic" not in methods:
            data["client_secret"] = cfg["client_secret"]
        else:
            auth = (cfg["client_id"], cfg["client_secret"])
    async with _http() as client:
        r = await client.post(meta["token_endpoint"], data=data, auth=auth,
                              headers={"Accept": "application/json"})
        if r.status_code >= 400:
            raise OIDCError(f"Token exchange failed ({r.status_code}): {r.text[:200]}")
        tokens = r.json()
        if "id_token" not in tokens:
            raise OIDCError("Provider did not return an id_token; is the openid scope enabled?")
        jwks = (await client.get(meta["jwks_uri"])).json()
        try:
            token = jwt.decode(tokens["id_token"], KeySet.import_key_set(jwks),
                               algorithms=["RS256", "RS384", "RS512", "ES256", "ES384", "PS256", "EdDSA"])
            jwt.JWTClaimsRegistry(
                leeway=120,
                iss={"essential": True, "value": meta["issuer"]},
                aud={"essential": True, "value": cfg["client_id"]},
            ).validate(token.claims)
        except (JoseError, ValueError) as e:
            raise OIDCError(f"Invalid id_token: {e}") from e
        claims = token.claims
        merged = dict(claims)
        if meta.get("userinfo_endpoint") and tokens.get("access_token"):
            ui = await client.get(meta["userinfo_endpoint"],
                                  headers={"Authorization": f"Bearer {tokens['access_token']}"})
            if ui.status_code == 200:
                # Groups often only appear in userinfo; id_token claims win on conflicts.
                merged = {**ui.json(), **merged}
        return merged


def _groups(claims: dict, claim: str) -> set[str]:
    value = claims.get(claim) or []
    if isinstance(value, str):
        value = [value]
    return {str(g) for g in value}


def _unique_username(db: Session, wanted: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9_.-]", "", wanted)[:90] or "user"
    name, n = base, 1
    while db.scalar(select(User.id).where(func.lower(User.username) == name.lower())):
        n += 1
        name = f"{base}{n}"
    return name


def resolve_user(db: Session, cfg: dict, claims: dict) -> User:
    """Find or create the user for these claims and apply group rules. Raises OIDCError."""
    groups = _groups(claims, cfg["groups_claim"])
    if cfg["user_group"] and cfg["user_group"] not in groups and cfg["admin_group"] not in groups:
        raise OIDCError("Your account is not in the group allowed to use this app")
    claim = cfg["user_claim"] or "email"
    ident = claims.get(claim)
    if not ident:
        raise OIDCError(f"The provider did not send the '{claim}' claim")
    ident = str(ident)
    if claim == "email":
        user = db.scalar(select(User).where(User.email == ident.lower()))
    else:
        user = db.scalar(select(User).where(func.lower(User.username) == ident.lower()))
    if user is None:
        if not cfg["signup_enabled"]:
            raise OIDCError("No account exists for you and sign-up is turned off")
        email = str(claims.get("email") or (ident if "@" in ident else f"{ident}@oidc.invalid")).lower()
        if db.scalar(select(User.id).where(User.email == email)):
            raise OIDCError("An account with your email already exists under a different username")
        user = User(
            username=_unique_username(db, ident if claim != "email"
                                      else str(claims.get("preferred_username") or email.split("@")[0])),
            email=email, full_name=str(claims.get("name") or ""), password_hash=None,
            auth_method="oidc", group_id=default_group(db).id,
        )
        db.add(user)
    if cfg["admin_group"]:
        user.is_admin = cfg["admin_group"] in groups
    db.commit()
    return user


@router.get("/callback")
async def callback(request: Request, db: Session = Depends(get_db)):
    if err := request.query_params.get("error"):
        return _fail(request.query_params.get("error_description") or err)
    saved = decode_token(request.cookies.get(STATE_COOKIE, "")) or {}
    if saved.get("typ") != "oidc" or saved.get("state") != request.query_params.get("state"):
        return _fail("Sign-in expired or was started in another browser. Please try again.")
    cfg = app_settings.get_group(db, "oidc")
    try:
        meta = await discover(cfg["configuration_url"])
        claims = await _exchange(cfg, meta, request.query_params.get("code", ""), saved["verifier"], request)
        if claims.get("nonce") != saved.get("nonce"):
            raise OIDCError("Nonce mismatch")
        user = resolve_user(db, cfg, claims)
    except (OIDCError, httpx.HTTPError, ValueError) as e:
        log.warning("OIDC login failed: %s", e)
        return _fail(str(e))
    resp = RedirectResponse("/", status_code=302)
    resp.delete_cookie(STATE_COOKIE, path="/api/auth/oidc")
    set_session(request, resp, user, remember=cfg["remember_me"])
    return resp

