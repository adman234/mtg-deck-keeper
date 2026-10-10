import base64
import hashlib
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
from cryptography.fernet import Fernet, InvalidToken

from .config import get_env

SESSION_COOKIE = "dk_session"
ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode()[:72], bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str | None) -> bool:
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(password.encode()[:72], hashed.encode())
    except ValueError:
        return False


def create_token(data: dict, hours: float) -> str:
    payload = {**data, "exp": datetime.now(UTC) + timedelta(hours=hours), "iat": datetime.now(UTC)}
    return jwt.encode(payload, get_env().secret_key, algorithm=ALGORITHM)


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, get_env().secret_key, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None


def _fernet() -> Fernet:
    digest = hashlib.sha256(("settings:" + get_env().secret_key).encode()).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt(value: str) -> str:
    return _fernet().encrypt(value.encode()).decode()


def decrypt(value: str) -> str | None:
    try:
        return _fernet().decrypt(value.encode()).decode()
    except InvalidToken:
        return None
