"""
KIara - Authentifizierung
Re-Export-Wrapper ueber kiara_auth (Shared Auth Library).

Alle bisherigen Import-Namen bleiben erhalten:
- Pure Functions (hash_password, verify_password, validate_password)
  als direkte Re-Exports aus kiara_auth.core.
- Session-Funktionen (create_session, validate_session, delete_session)
  als Wrapper mit injiziertem KIara-State (Repositories, Permissions).
- FastAPI-Dependencies und Cookie-Handling als lokale Wrapper
  ueber den CookieAuthAdapter.
"""

import logging
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request, Response

from kiara.repositories.user import SessionRepository
from kiara.shared.constants import SESSION_MAX_AGE_SECONDS
from kiara.shared.crypto import get_secret_key
from kiara.shared.db import db_query

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kategorie 1: Pure Functions — direkter Re-Export aus kiara_auth.core
# ---------------------------------------------------------------------------
from kiara_auth.core import (  # noqa: F401
    hash_password,
    hash_token,
    validate_password,
    verify_password,
)

# ---------------------------------------------------------------------------
# Modul-State (KIara-spezifisch)
# ---------------------------------------------------------------------------

SESSION_MAX_AGE = SESSION_MAX_AGE_SECONDS
_SESSION_SALT = "kiara-session"

session_repo = SessionRepository()

# Cookie-Adapter: dynamischer Cookie-Name, lazy Secret-Key
from kiara_auth.fastapi import CookieAuthAdapter

# Cookie-Secure-Flag aus Config lesen (Default False fuer interne HTTP-Deployments,
# per security.cookie_secure: true in kiara.yaml aktivierbar fuer HTTPS-Deployments).
_cookie_secure = False
try:
    from kiara.config import load_config as _load_cookie_config

    _cookie_secure = bool(
        (_load_cookie_config().get("security") or {}).get("cookie_secure") or False
    )
except Exception as e:
    logger.warning(
        "Cookie-Secure Config-Load fehlgeschlagen (%s: %s) — Default False bleibt aktiv. "
        "Bei HTTPS-Deployment manuell pruefen, dass security.cookie_secure korrekt geladen wird.",
        type(e).__name__,
        e,
    )

_cookie_adapter = CookieAuthAdapter(
    secret_key_provider=get_secret_key,
    cookie_name="kiara_session",
    session_max_age=SESSION_MAX_AGE,
    session_salt=_SESSION_SALT,
    cookie_secure=_cookie_secure,
)

# Kompatibilitaets-Property: COOKIE_NAME als dynamisch lesbarer Name
COOKIE_NAME = _cookie_adapter.cookie_name


def set_cookie_name(name: str) -> None:
    """Setzt den Cookie-Namen fuer diese Prozess-Instanz.

    Erlaubt separate Sessions fuer Chat-UI und Admin-UI,
    da Cookies domaingebunden sind (nicht portgebunden).
    """
    global COOKIE_NAME
    _cookie_adapter.cookie_name = name
    COOKIE_NAME = name


# ---------------------------------------------------------------------------
# Kategorie 2: DI-abhaengige Funktionen — Wrapper mit injiziertem State
# ---------------------------------------------------------------------------


def create_session(user_id: int, *, conn=None) -> tuple[str, str]:
    """Erstellt eine neue Session in der DB und gibt (Session-Token, CSRF-Token) zurueck.

    Beide Tokens sind Klartext. Vor der DB-Speicherung werden beide mit
    SHA-256 gehasht; nur die Hashes liegen in der DB.
    """
    token = secrets.token_hex(32)
    token_hash = hash_token(token)
    csrf_plain = secrets.token_hex(32)
    csrf_hash = hash_token(csrf_plain)
    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=SESSION_MAX_AGE)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    session_repo.create(
        conn=conn,
        user_id=user_id,
        token=token_hash,
        expires_at=expires_at,
        csrf_token_hash=csrf_hash,
    )
    return token, csrf_plain


def validate_session(token: str) -> dict | None:
    """Validiert ein Session-Token und laedt den User inkl. Berechtigungen.

    token ist Klartext (aus unsign_token). Wird intern gehashed fuer DB-Lookup.
    """
    token_hash = hash_token(token)
    with db_query() as db:
        # Spec §1a + §7 (und ImplScan-Finding 2): Sessions ohne csrf_token_hash
        # gelten als invalid. Kein Lazy-Create — Betroffene muessen neu anmelden.
        row = db.execute(
            """SELECT s.*, u.id as uid, u.username, u.vorname, u.nachname,
                      u.email, u.is_admin, u.is_active, u.must_change_password,
                      u.auth_source
               FROM sessions s
               JOIN users u ON s.user_id = u.id
               WHERE s.token = %s AND u.is_active = true
                     AND s.expires_at::timestamp > NOW()
                     AND s.csrf_token_hash IS NOT NULL""",
            (token_hash,),
        ).fetchone()
        if not row:
            return None

        user_id = row["uid"]
        from kiara.web.config_utils import load_user_permissions

        group_ids, datasource_ids = load_user_permissions(db, user_id)

        must_change = row.get("must_change_password", 0)

        return {
            "id": user_id,
            "username": row["username"],
            "vorname": row["vorname"],
            "nachname": row["nachname"],
            "email": row["email"],
            "is_admin": row["is_admin"],
            "must_change_password": must_change,
            "auth_source": row.get("auth_source", "local"),
            "group_ids": group_ids,
            "datasource_ids": datasource_ids,
            "_session_token": token,
        }


def delete_session(token: str, *, conn=None) -> None:
    """Loescht eine Session aus der DB (Logout).

    token ist Klartext (aus unsign_token). Wird intern gehashed fuer DB-Lookup.
    """
    token_hash = hash_token(token)
    session_repo.delete_by_token(token_hash, conn=conn)


# ---------------------------------------------------------------------------
# Kategorie 3: FastAPI-/Cookie-Adapter — Wrapper ueber CookieAuthAdapter
# ---------------------------------------------------------------------------


def sign_token(token: str) -> str:
    """Signiert ein Session-Token mit HMAC fuer Cookie-Sicherheit."""
    return _cookie_adapter.sign_token(token)


def unsign_token(signed: str) -> str | None:
    """Verifiziert und entpackt ein signiertes Token. None bei ungueltigem Token."""
    return _cookie_adapter.unsign_token(signed)


def set_session_cookie(response: Response, token: str) -> None:
    """Setzt das Session-Cookie mit HMAC-signiertem Token."""
    _cookie_adapter.set_session_cookie(response, token)


def delete_session_cookie(response: Response) -> None:
    """Loescht das Session-Cookie (Logout im Browser)."""
    _cookie_adapter.delete_session_cookie(response)


def _extract_user_from_request(request: Request) -> dict | None:
    """Extrahiert und validiert den User aus dem Session-Cookie."""
    token = _cookie_adapter.extract_token_from_request(request)
    if not token:
        return None
    return validate_session(token)


async def get_current_user(request: Request) -> dict:
    """FastAPI-Dependency: Gibt den angemeldeten User zurueck oder wirft 401.

    Tier-neutral — erlaubt sowohl Admin- als auch regulaere Konten.
    Wird fuer Endpoints verwendet, die in beiden UIs funktionieren muessen
    (z.B. /api/auth/me, /api/auth/password, /api/auth/logout).
    """
    user = _extract_user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Nicht angemeldet")
    return user


async def get_chat_user(request: Request) -> dict:
    """FastAPI-Dependency: Gibt den angemeldeten Non-Admin-User zurueck oder wirft 401/403.

    User-Tier — Admin-Konten werden abgewiesen. Wird fuer Chat-,
    Such- und Workflow-Endpoints in der User-UI verwendet.
    """
    user = _extract_user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Nicht angemeldet")
    if user["is_admin"]:
        raise HTTPException(
            status_code=403,
            detail="Admin-Konten haben keinen Chat-Zugriff. "
            "Bitte einen regulaeren Benutzer verwenden.",
        )
    return user


async def get_admin_user(request: Request) -> dict:
    """FastAPI-Dependency: Gibt den angemeldeten Admin-User zurueck oder wirft 401/403."""
    user = _extract_user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Nicht angemeldet")
    if not user["is_admin"]:
        raise HTTPException(status_code=403, detail="Kein Admin-Zugriff")
    return user
