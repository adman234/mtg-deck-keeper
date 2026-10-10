from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from .. import app_settings
from ..config import get_env
from ..db import get_db
from ..deps import current_user, current_user_optional
from ..models import Group, User
from ..security import SESSION_COOKIE, create_token, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])

REMEMBER_HOURS = 24 * 30


def user_out(u: User) -> dict:
    return {"id": u.id, "username": u.username, "email": u.email, "full_name": u.full_name,
            "is_admin": u.is_admin, "auth_method": u.auth_method, "has_password": bool(u.password_hash)}


def set_session(request: Request, response: Response, user: User, remember: bool = False) -> None:
    hours = REMEMBER_HOURS if remember else get_env().token_time
    token = create_token({"sub": str(user.id), "typ": "session"}, hours)
    response.set_cookie(
        SESSION_COOKIE, token, max_age=int(hours * 3600), httponly=True, samesite="lax",
        secure=request.url.scheme == "https", path="/",
    )


def default_group(db: Session) -> Group:
    group = db.scalar(select(Group).order_by(Group.id).limit(1))
    if group is None:
        group = Group(name="Home")
        db.add(group)
        db.flush()
    return group


@router.get("/status")
def status(db: Session = Depends(get_db), user: User | None = Depends(current_user_optional)):
    oidc = app_settings.get_group(db, "oidc")
    oidc_ready = bool(oidc["enabled"] and oidc["configuration_url"] and oidc["client_id"])
    return {
        "setup_required": db.scalar(select(func.count(User.id))) == 0,
        "password_login": app_settings.get(db, "auth.password_login") or not oidc_ready,
        "oidc_enabled": oidc_ready,
        "oidc_provider_name": oidc["provider_name"],
        "oidc_auto_redirect": oidc_ready and oidc["auto_redirect"],
        "user": user_out(user) if user else None,
    }


class SetupIn(BaseModel):
    username: str = Field(min_length=2, max_length=100)
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+$", max_length=255)
    password: str = Field(min_length=8)
    full_name: str = ""


@router.post("/setup")
def setup(body: SetupIn, request: Request, response: Response, db: Session = Depends(get_db)):
    if db.scalar(select(func.count(User.id))):
        raise HTTPException(409, "Setup is already done")
    user = User(username=body.username, email=body.email.lower(), full_name=body.full_name,
                password_hash=hash_password(body.password), is_admin=True, group_id=default_group(db).id)
    db.add(user)
    db.commit()
    set_session(request, response, user)
    return user_out(user)


class LoginIn(BaseModel):
    username: str
    password: str
    remember: bool = False


@router.post("/login")
def login(body: LoginIn, request: Request, response: Response, db: Session = Depends(get_db)):
    if not status(db, None)["password_login"]:
        raise HTTPException(403, "Password login is turned off. Use the single sign-on button.")
    ident = body.username.strip().lower()
    user = db.scalar(select(User).where(or_(func.lower(User.username) == ident, User.email == ident)))
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Wrong username or password")
    set_session(request, response, user, body.remember)
    return user_out(user)


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@router.get("/me")
def me(user: User = Depends(current_user)):
    return user_out(user)


class PasswordIn(BaseModel):
    current_password: str = ""
    new_password: str = Field(min_length=8)


@router.put("/me/password")
def change_password(body: PasswordIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if user.password_hash and not verify_password(body.current_password, user.password_hash):
        raise HTTPException(400, "Current password is wrong")
    user.password_hash = hash_password(body.new_password)
    db.commit()
    return {"ok": True}
