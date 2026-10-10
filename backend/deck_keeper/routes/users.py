from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import admin_user
from ..models import User
from ..security import hash_password
from .auth import default_group, user_out

router = APIRouter(prefix="/api/users", tags=["users"], dependencies=[Depends(admin_user)])


@router.get("")
def list_users(db: Session = Depends(get_db)):
    return [user_out(u) for u in db.scalars(select(User).order_by(User.username))]


class UserIn(BaseModel):
    username: str = Field(min_length=2, max_length=100)
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+$", max_length=255)
    full_name: str = ""
    password: str = Field(min_length=8)
    is_admin: bool = False


@router.post("", status_code=201)
def create_user(body: UserIn, db: Session = Depends(get_db)):
    if db.scalar(select(User.id).where(func.lower(User.username) == body.username.lower())):
        raise HTTPException(409, "That username is taken")
    if db.scalar(select(User.id).where(User.email == body.email.lower())):
        raise HTTPException(409, "That email is already used")
    user = User(username=body.username, email=body.email.lower(), full_name=body.full_name,
                password_hash=hash_password(body.password), is_admin=body.is_admin,
                group_id=default_group(db).id)
    db.add(user)
    db.commit()
    return user_out(user)


class UserPatch(BaseModel):
    is_admin: bool | None = None
    full_name: str | None = None
    password: str | None = Field(default=None, min_length=8)


@router.patch("/{user_id}")
def update_user(user_id: int, body: UserPatch, db: Session = Depends(get_db), me: User = Depends(admin_user)):
    user = db.get(User, user_id) or _404()
    if body.is_admin is False and user.id == me.id:
        raise HTTPException(400, "You can't remove your own admin rights")
    if body.is_admin is not None:
        user.is_admin = body.is_admin
    if body.full_name is not None:
        user.full_name = body.full_name
    if body.password:
        user.password_hash = hash_password(body.password)
    db.commit()
    return user_out(user)


@router.delete("/{user_id}", status_code=204)
def delete_user(user_id: int, db: Session = Depends(get_db), me: User = Depends(admin_user)):
    user = db.get(User, user_id) or _404()
    if user.id == me.id:
        raise HTTPException(400, "You can't delete yourself")
    db.delete(user)
    db.commit()


def _404():
    raise HTTPException(404, "User not found")
