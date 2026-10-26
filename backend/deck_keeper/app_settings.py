"""Runtime settings that can come from an environment variable OR the admin GUI.

Precedence: environment variable > value saved in the GUI > default.
A setting provided by the environment is shown read-only in the GUI so the
two sources never fight.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Setting
from .security import decrypt, encrypt

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Spec:
    key: str
    env: str
    type: type
    default: Any
    label: str
    group: str
    help: str = ""
    secret: bool = False
    choices: tuple[str, ...] | None = None


SPECS: list[Spec] = [
    # Vision model
    Spec("vision.enabled", "VISION_ENABLED", bool, True, "Enable photo scanning", "vision"),
    Spec("vision.base_url", "VISION_BASE_URL", str, "http://llama-swap:8080/v1", "API base URL", "vision",
         "OpenAI-compatible endpoint ending in /v1 (llama-swap, Ollama, LM Studio, vLLM, OpenAI)."),
    Spec("vision.api_key", "VISION_API_KEY", str, "", "API key", "vision",
         "Leave blank for local servers that don't check keys.", secret=True),
    Spec("vision.model", "VISION_MODEL", str, "", "Model", "vision",
         "The model ID the server knows, e.g. the key under models: in llama-swap's config."),
    Spec("vision.timeout", "VISION_TIMEOUT", int, 300, "Request timeout (seconds)", "vision",
         "llama-swap may need to load the model on the first request."),
    Spec("vision.max_concurrency", "VISION_MAX_CONCURRENCY", int, 1, "Parallel requests", "vision",
         "Match llama-server's --parallel value."),
    Spec("vision.confidence_threshold", "VISION_CONFIDENCE_THRESHOLD", float, 0.80, "Review threshold", "vision",
         "Lines below this confidence are flagged for review (0–1)."),
    Spec("vision.prepass_default", "CARD_DETECTION_DEFAULT", bool, True, "Cut out cards when the model gives no boxes",
         "vision", "Fallback: if a photo's whole-table read fails or gives no card positions, find cards with "
         "OpenCV and read them one at a time. Not used when the model gives boxes (close-ups do that job)."),
    Spec("vision.disable_thinking", "VISION_DISABLE_THINKING", bool, True, "Turn off model thinking", "vision",
         "Asks thinking models (Qwen3 and similar) to answer directly. Faster, and stops empty replies "
         "when reasoning uses up the reply budget."),
    Spec("vision.collector_lookup", "MATCH_COLLECTOR_NUMBER", bool, True, "Use set code and collector number",
         "vision", "When the name doesn't match well, look up the exact printing on Scryfall from the code "
         "printed at the bottom-left. Works for any language and frame."),
    Spec("vision.foreign_lookup", "MATCH_FOREIGN_NAMES", bool, True, "Look up foreign-language names", "vision",
         "Search Scryfall for names that aren't English (German, Japanese, …) and for alternate "
         "names such as Godzilla cards. Answers are cached."),
    Spec("vision.candidate_reask", "MATCH_REASK", bool, True, "Ask again with likely candidates", "vision",
         "For hard-to-read cut-out cards, show the model the card again with the closest names and "
         "let it pick. One extra model call per unclear card."),
    Spec("vision.pile_zoom", "PILE_ZOOM", bool, True, "Zoom in on each pile", "vision",
         "Ask the model where each card or pile is, then re-read each one from a close-up to get the "
         "count and name right. One extra model call per pile; needs a model that can give boxes."),
    Spec("vision.image_max_side", "VISION_IMAGE_MAX_SIDE", int, 1600, "Max image size sent (px)", "vision"),
    # Login / OIDC (names match Mealie)
    Spec("auth.password_login", "ALLOW_PASSWORD_LOGIN", bool, True, "Allow username/password login", "auth",
         "Turn off to force OIDC. Set ALLOW_PASSWORD_LOGIN=true in the environment to recover if locked out."),
    Spec("oidc.enabled", "OIDC_AUTH_ENABLED", bool, False, "Enable OIDC login", "oidc"),
    Spec("oidc.provider_name", "OIDC_PROVIDER_NAME", str, "OAuth", "Provider name", "oidc",
         "Shown on the login button."),
    Spec("oidc.configuration_url", "OIDC_CONFIGURATION_URL", str, "", "Configuration URL", "oidc",
         "Ends in /.well-known/openid-configuration"),
    Spec("oidc.client_id", "OIDC_CLIENT_ID", str, "", "Client ID", "oidc"),
    Spec("oidc.client_secret", "OIDC_CLIENT_SECRET", str, "", "Client secret", "oidc", secret=True),
    Spec("oidc.signup_enabled", "OIDC_SIGNUP_ENABLED", bool, True, "Create users on first login", "oidc"),
    Spec("oidc.user_group", "OIDC_USER_GROUP", str, "", "Required user group", "oidc",
         "Leave blank to allow anyone the provider authenticates."),
    Spec("oidc.admin_group", "OIDC_ADMIN_GROUP", str, "", "Admin group", "oidc",
         "Members become admins; checked on every login."),
    Spec("oidc.auto_redirect", "OIDC_AUTO_REDIRECT", bool, False, "Skip the login page", "oidc",
         "Add ?direct=1 to /login to reach the form anyway."),
    Spec("oidc.remember_me", "OIDC_REMEMBER_ME", bool, False, "Remember OIDC logins for 30 days", "oidc"),
    Spec("oidc.user_claim", "OIDC_USER_CLAIM", str, "email", "User claim", "oidc",
         "Claim used to find the user: email, preferred_username or sub."),
    Spec("oidc.groups_claim", "OIDC_GROUPS_CLAIM", str, "groups", "Groups claim", "oidc"),
    Spec("oidc.scopes_override", "OIDC_SCOPES_OVERRIDE", str, "", "Scopes override", "oidc",
         "Default: openid profile email groups"),
    # Card database
    Spec("scryfall.auto_sync", "SCRYFALL_AUTO_SYNC", bool, True, "Check for new cards daily", "cards"),
    Spec("scryfall.sync_hour", "SCRYFALL_SYNC_HOUR", int, 4, "Sync hour (0–23, server time)", "cards"),
]

BY_KEY = {s.key: s for s in SPECS}
_TRUE = {"1", "true", "yes", "on"}


def _coerce(spec: Spec, raw: Any) -> Any:
    if spec.type is bool:
        return raw if isinstance(raw, bool) else str(raw).strip().lower() in _TRUE
    if spec.type is int:
        return int(raw)
    if spec.type is float:
        return float(raw)
    return "" if raw is None else str(raw)


def env_value(spec: Spec) -> str | None:
    raw = os.environ.get(spec.env)
    return raw if raw not in (None, "") else None


def _db_values(db: Session) -> dict[str, str]:
    rows = db.scalars(select(Setting).where(Setting.key.in_(BY_KEY.keys())))
    return {r.key: r.value for r in rows}


def _resolve(spec: Spec, stored: dict[str, str]) -> tuple[Any, str]:
    env = env_value(spec)
    if env is not None:
        try:
            return _coerce(spec, env), "env"
        except ValueError:
            log.warning("Ignoring invalid %s=%r", spec.env, env)
    if spec.key in stored:
        raw = json.loads(stored[spec.key])
        if spec.secret and raw:
            raw = decrypt(raw)
            if raw is None:
                log.warning("Could not decrypt %s; was SECRET_KEY changed? Re-enter it in Settings.", spec.key)
                return spec.default, "default"
        try:
            return _coerce(spec, raw), "db"
        except ValueError:
            pass
    return spec.default, "default"


def get(db: Session, key: str) -> Any:
    spec = BY_KEY[key]
    return _resolve(spec, _db_values(db))[0]


def get_group(db: Session, group: str) -> dict[str, Any]:
    """Return {short_name: value} for a group, e.g. get_group(db, 'oidc')['client_id']."""
    stored = _db_values(db)
    return {s.key.split(".", 1)[1]: _resolve(s, stored)[0] for s in SPECS if s.group == group}


def describe(db: Session) -> list[dict]:
    stored = _db_values(db)
    out = []
    for s in SPECS:
        value, source = _resolve(s, stored)
        out.append({
            "key": s.key,
            "group": s.group,
            "label": s.label,
            "help": s.help,
            "type": s.type.__name__,
            "env": s.env,
            "secret": s.secret,
            "value": None if s.secret else value,
            "is_set": bool(value) if s.secret else None,
            "source": source,
            "locked": source == "env",
        })
    return out


class SettingsError(ValueError):
    pass


def update(db: Session, values: dict[str, Any]) -> None:
    """Save GUI values. For secrets, None means keep and "" means clear."""
    for key, raw in values.items():
        spec = BY_KEY.get(key)
        if spec is None:
            raise SettingsError(f"Unknown setting {key}")
        if env_value(spec) is not None:
            raise SettingsError(f"{spec.label} is set by the {spec.env} environment variable")
        if spec.secret and raw is None:
            continue
        try:
            value = _coerce(spec, raw)
        except (TypeError, ValueError) as e:
            raise SettingsError(f"{spec.label}: {e}") from e
        if spec.secret and value:
            value = encrypt(value)
        row = db.get(Setting, key)
        if row is None:
            db.add(Setting(key=key, value=json.dumps(value)))
        else:
            row.value = json.dumps(value)
    db.commit()


def get_state(db: Session, key: str, default: Any = None) -> Any:
    row = db.get(Setting, f"state.{key}")
    return json.loads(row.value) if row else default


def set_state(db: Session, key: str, value: Any) -> None:
    row = db.get(Setting, f"state.{key}")
    if row is None:
        db.add(Setting(key=f"state.{key}", value=json.dumps(value)))
    else:
        row.value = json.dumps(value)
    db.commit()
