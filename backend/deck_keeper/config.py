"""Process-level configuration that must be known before the database is up.

Everything a user might want to change at runtime (vision model, OIDC, card sync)
lives in ``app_settings`` instead, where it can come from an env var or the GUI.
"""

import secrets
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Env(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("/app/data")
    database_url: str = ""
    secret_key: str = ""
    base_url: str = ""
    token_time: int = 48  # hours, same name as Mealie
    log_level: str = "info"
    scryfall_api: str = "https://api.scryfall.com"

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{self.data_dir / 'deck-keeper.db'}"

    @property
    def scans_dir(self) -> Path:
        return self.data_dir / "scans"


@lru_cache
def get_env() -> Env:
    env = Env()
    env.data_dir.mkdir(parents=True, exist_ok=True)
    env.scans_dir.mkdir(parents=True, exist_ok=True)
    if not env.secret_key:
        key_file = env.data_dir / ".secret_key"
        if not key_file.exists():
            key_file.write_text(secrets.token_urlsafe(48))
            key_file.chmod(0o600)
        env.secret_key = key_file.read_text().strip()
    return env
