from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .. import app_settings
from ..db import get_db
from ..deps import admin_user
from ..services import matching, scryfall, vision
from .oidc import OIDCError, discover, redirect_uri

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(admin_user)])


@router.get("/settings")
def get_settings(request: Request, db: Session = Depends(get_db)):
    return {"settings": app_settings.describe(db), "oidc_redirect_uri": redirect_uri(request)}


class SettingsIn(BaseModel):
    values: dict[str, Any]


@router.put("/settings")
def put_settings(body: SettingsIn, request: Request, db: Session = Depends(get_db)):
    try:
        app_settings.update(db, body.values)
    except app_settings.SettingsError as e:
        raise HTTPException(400, str(e)) from e
    return get_settings(request, db)


class VisionTestIn(BaseModel):
    base_url: str | None = None
    api_key: str | None = None


@router.post("/test-vision")
async def test_vision(body: VisionTestIn, db: Session = Depends(get_db)):
    s = app_settings.get_group(db, "vision")
    cfg = vision.VisionConfig.from_settings(s)
    if body.base_url:
        cfg.base_url = body.base_url
    if body.api_key:
        cfg.api_key = body.api_key
    cfg.timeout = 15
    try:
        models = await vision.list_models(cfg)
    except (httpx.HTTPError, ValueError, KeyError) as e:
        return {"ok": False, "error": f"Could not list models at {cfg.base_url}/models: {e}", "models": []}
    return {"ok": True, "models": models}


class OIDCTestIn(BaseModel):
    configuration_url: str | None = None


@router.post("/test-oidc")
async def test_oidc(body: OIDCTestIn, db: Session = Depends(get_db)):
    url = body.configuration_url or app_settings.get(db, "oidc.configuration_url")
    if not url:
        return {"ok": False, "error": "Enter a configuration URL first"}
    try:
        meta = await discover(url, force=True)
    except (httpx.HTTPError, ValueError, OIDCError) as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "issuer": meta["issuer"], "scopes_supported": meta.get("scopes_supported", [])}


@router.get("/cards")
def card_db_status(db: Session = Depends(get_db)):
    return {"count": len(matching.get_index(db)), "last_sync": app_settings.get_state(db, "scryfall_sync")}


@router.post("/cards/sync")
async def card_db_sync(db: Session = Depends(get_db)):
    try:
        return await scryfall.sync_card_names(db)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Could not reach Scryfall: {e}") from e
