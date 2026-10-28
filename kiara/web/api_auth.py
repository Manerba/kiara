"""
KIara - Auth API-Router
Login, Logout, Me, Passwort-Aenderung.
"""

import asyncio
import logging
import secrets
import threading
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from kiara.repositories.config import ConfigRepository
from kiara.repositories.user import SessionRepository, UserRepository
from kiara.shared.constants import (
    RATE_LIMIT_BASE_DELAY_SECS,
    RATE_LIMIT_INCREMENT_SECS,
    RATE_LIMIT_THRESHOLD,
)
from kiara.shared.db import db_transaction
from kiara.web.auth import (
    create_session,
    delete_session,
    delete_session_cookie,
    get_current_user,
    hash_password,
    hash_token,
    set_session_cookie,
    verify_password,
)
from kiara.web.logging_config import get_security_logger
from kiara.web.middleware import NotAuthenticatedError

logger = logging.getLogger(__name__)
security_log = get_security_logger()

router = APIRouter(prefix="/api/auth", tags=["auth"])

user_repo = UserRepository()
session_repo = SessionRepository()
config_repo = ConfigRepository()

LOGIN_FAILED_MSG = "Anmeldung fehlgeschlagen"
_LDAP_LOGIN_FAILURE_MIN_SECONDS = 0.35

# Rate-Limiting
_rate_limits: dict[str, dict] = {}
_rate_limits_lock = threading.Lock()
_RATE_LIMIT_MAX_ENTRIES = 1000
_RATE_LIMIT_EXPIRY_SECS = 7200  # 2 Stunden
_TRUSTED_PROXIES = {"127.0.0.1", "::1"}


def _get_client_ip(request: Request) -> str:
    """Extrahiert die Client-IP (X-Forwarded-For-aware fuer Reverse-Proxy)."""
    direct_ip = request.client.host if request.client else "unknown"
    if direct_ip in _TRUSTED_PROXIES:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return direct_ip


def _check_rate_limit(key: str) -> int | None:
    """Prueft ob ein Rate-Limit aktiv ist.

    Rate-Limit-Algorithmus (progressive Backoff):
      1. Nach RATE_LIMIT_THRESHOLD Fehlversuchen wird blockiert.
      2. Wartezeit = BASE_DELAY + (Versuche - THRESHOLD) * INCREMENT.
         3 Fehlversuche → 10s, 4 → 15s, 5 → 20s.
      3. Nach erfolgreicher Anmeldung wird der Zaehler zurueckgesetzt.

    Cleanup: Bei > _RATE_LIMIT_MAX_ENTRIES Eintraegen werden alle
    Eintraege entfernt, deren letzter Fehlversuch > 2 Stunden zurueckliegt.

    Args:
        key: Rate-Limit-Schluessel (z.B. IP-Adresse oder "login:<ip>").

    Returns:
        Wartezeit in Sekunden oder None wenn kein Limit aktiv.
    """
    with _rate_limits_lock:
        # Cleanup alter Eintraege bei Ueberschreiten des Limits
        if len(_rate_limits) > _RATE_LIMIT_MAX_ENTRIES:
            now = time.monotonic()
            stale_keys = [
                k for k, v in _rate_limits.items() if now - v["last_fail"] > _RATE_LIMIT_EXPIRY_SECS
            ]
            for k in stale_keys:
                del _rate_limits[k]

        entry = _rate_limits.get(key)
        if not entry or entry["count"] < RATE_LIMIT_THRESHOLD:
            return None
        extra = entry["count"] - RATE_LIMIT_THRESHOLD
        wait = RATE_LIMIT_BASE_DELAY_SECS + extra * RATE_LIMIT_INCREMENT_SECS
        elapsed = time.monotonic() - entry["last_fail"]
        if elapsed < wait:
            return int(wait - elapsed) + 1
        return None


def _record_fail(key: str) -> None:
    """Zaehlt einen fehlgeschlagenen Login-Versuch fuer progressive Verzoegerung."""
    with _rate_limits_lock:
        entry = _rate_limits.get(key)
        if entry:
            entry["count"] += 1
            entry["last_fail"] = time.monotonic()
        else:
            _rate_limits[key] = {"count": 1, "last_fail": time.monotonic()}


def _reset_fails(key: str) -> None:
    """Setzt den Fehlerzaehler nach erfolgreichem Login zurueck."""
    with _rate_limits_lock:
        _rate_limits.pop(key, None)


async def _pad_ldap_failure(started_at: float) -> None:
    """Gleicht schnelle LDAP-Fehlerpfade gegen Username-Enumeration an."""
    remaining = _LDAP_LOGIN_FAILURE_MIN_SECONDS - (time.monotonic() - started_at)
    if remaining > 0:
        await asyncio.sleep(remaining)


async def _get_current_user_for_csrf(request: Request) -> dict:
    """CSRF-Refresh braucht den kanonischen 401-Body mit code-Feld."""
    try:
        return await get_current_user(request)
    except HTTPException as exc:
        if exc.status_code == 401:
            raise NotAuthenticatedError() from exc
        raise


class LoginRequest(BaseModel):
    """Request fuer die Benutzer-Anmeldung."""

    username: str = Field(..., max_length=255)
    password: str = Field(..., max_length=1000)


class PasswordChangeRequest(BaseModel):
    """Request zum Aendern des eigenen Passworts."""

    old_password: str
    new_password: str


@router.post("/login")
async def login(data: LoginRequest, request: Request):
    client_ip = _get_client_ip(request)
    rate_key = f"login:{client_ip}"

    remaining = _check_rate_limit(rate_key)
    if remaining is not None:
        security_log.warning("RATE_LIMITED key=%s", rate_key)
        raise HTTPException(
            status_code=429,
            detail=f"Zu viele Fehlversuche. Bitte {remaining}s warten.",
            headers={"Retry-After": str(remaining)},
        )

    row = user_repo.get_by_username(data.username)

    _DUMMY_HASH = "$2b$12$q3YzUSzWDNUojF2bhmztfeh/PWYexQZ7EqMzbTaqpBbCxTMwzlna."
    if not row:
        # Timing-Angleichung: Dummy-Verify verhindert Username-Enumeration
        verify_password("dummy", _DUMMY_HASH)
        _record_fail(rate_key)
        security_log.warning("LOGIN_FAILED user=%s ip=%s", data.username, client_ip)
        raise HTTPException(status_code=401, detail=LOGIN_FAILED_MSG)

    auth_source = row.get("auth_source", "local")
    is_active = row.get("is_active", 0)

    if not is_active:
        # Timing-Angleichung: Dummy-Verify verhindert Username-Enumeration (aktiv vs. inaktiv)
        verify_password("dummy", _DUMMY_HASH)
        _record_fail(rate_key)
        security_log.warning("LOGIN_FAILED user=%s ip=%s (inactive)", data.username, client_ip)
        raise HTTPException(status_code=401, detail=LOGIN_FAILED_MSG)

    if auth_source == "ldap":
        from kiara.web.config_utils import build_ldap_config

        ldap_started_at = time.monotonic()
        config_row = config_repo.get("ldap_config")

        if not config_row:
            security_log.warning(
                "LOGIN_FAILED user=%s ip=%s reason=ldap_not_configured",
                data.username,
                client_ip,
            )
            await _pad_ldap_failure(ldap_started_at)
            _record_fail(rate_key)
            raise HTTPException(status_code=401, detail=LOGIN_FAILED_MSG)

        try:
            from kiara.web.ldap_utils import authenticate_user

            config = build_ldap_config(config_row)
            success, error = await asyncio.to_thread(
                authenticate_user, config, data.username, data.password
            )
        except ImportError as exc:
            security_log.warning(
                "LOGIN_FAILED user=%s ip=%s reason=ldap_module_missing detail=%s",
                data.username,
                client_ip,
                exc,
            )
            await _pad_ldap_failure(ldap_started_at)
            _record_fail(rate_key)
            raise HTTPException(status_code=401, detail=LOGIN_FAILED_MSG)
        except Exception as exc:
            security_log.warning(
                "LOGIN_FAILED user=%s ip=%s reason=ldap_unreachable detail=%s",
                data.username,
                client_ip,
                exc,
            )
            await _pad_ldap_failure(ldap_started_at)
            _record_fail(rate_key)
            raise HTTPException(status_code=401, detail=LOGIN_FAILED_MSG)

        if not success:
            await _pad_ldap_failure(ldap_started_at)
            _record_fail(rate_key)
            security_log.warning(
                "LOGIN_FAILED user=%s ip=%s reason=ldap_auth_failed detail=%s",
                data.username,
                client_ip,
                error or "unknown",
            )
            raise HTTPException(status_code=401, detail=LOGIN_FAILED_MSG)
    else:
        if not row["password_hash"]:
            # Timing-Angleichung: Dummy-Verify verhindert Timing-Leak bei leerem Hash
            verify_password("dummy", _DUMMY_HASH)
            pw_valid = False
        else:
            pw_valid = verify_password(data.password, row["password_hash"])
        if not pw_valid:
            _record_fail(rate_key)
            security_log.warning("LOGIN_FAILED user=%s ip=%s", data.username, client_ip)
            raise HTTPException(status_code=401, detail=LOGIN_FAILED_MSG)

    _reset_fails(rate_key)
    security_log.info("LOGIN_SUCCESS user=%s ip=%s", data.username, client_ip)

    # Expired sessions bereinigen + Session-Row inkl. CSRF-Hash atomar anlegen
    with db_transaction() as conn:
        session_repo.cleanup_expired(conn=conn)
        token, csrf_plain = create_session(row["id"], conn=conn)

    must_change = row.get("must_change_password", 0)

    user_data = {
        "id": row["id"],
        "username": row["username"],
        "vorname": row["vorname"],
        "nachname": row["nachname"],
        "is_admin": row["is_admin"],
        "must_change_password": must_change,
    }

    response = JSONResponse(
        content={
            "message": "Angemeldet",
            "user": user_data,
            "csrf_token": csrf_plain,
        },
        headers={"Cache-Control": "no-store"},
    )
    set_session_cookie(response, token)
    return response


@router.post("/logout")
async def logout(request: Request, user=Depends(get_current_user)):
    security_log.info("LOGOUT user=%s", user["username"])
    token = user.get("_session_token")
    if token:
        delete_session(token)

    response = JSONResponse(content={"message": "Abgemeldet"})
    delete_session_cookie(response)
    return response


@router.get("/me")
async def me(user=Depends(get_current_user)):
    safe_user = {k: v for k, v in user.items() if not k.startswith("_")}
    return {"user": safe_user}


@router.put("/password")
async def change_password(data: PasswordChangeRequest, user=Depends(get_current_user)):
    rate_key = f"pwchange:{user['id']}"

    remaining = _check_rate_limit(rate_key)
    if remaining is not None:
        security_log.warning("RATE_LIMITED key=%s", rate_key)
        raise HTTPException(
            status_code=429,
            detail=f"Zu viele Fehlversuche. Bitte {remaining}s warten.",
            headers={"Retry-After": str(remaining)},
        )

    user_row = user_repo.get_by_id(user["id"])

    if not user_row:
        _record_fail(rate_key)
        security_log.warning(
            "PASSWORD_CHANGE_FAILED user=%s reason=user_not_found", user["username"]
        )
        raise HTTPException(status_code=400, detail="Altes Passwort ist falsch")

    # LDAP-User haben keinen lokalen Passwort-Hash und koennen ihr Passwort
    # nicht ueber KIara aendern (LDAP-Bind-Account ist read-only). Explizit
    # ablehnen BEVOR bcrypt auf NULL-Hash aufgerufen wird.
    if user_row.get("auth_source") == "ldap" or not user_row.get("password_hash"):
        security_log.warning(
            "PASSWORD_CHANGE_REJECTED user=%s reason=ldap_or_no_local_hash",
            user["username"],
        )
        raise HTTPException(
            status_code=400,
            detail="Passwortaenderung nur fuer lokale Accounts moeglich. "
            "LDAP-Benutzer aendern ihr Passwort im LDAP-System.",
        )

    if not verify_password(data.old_password, user_row["password_hash"]):
        _record_fail(rate_key)
        security_log.warning("PASSWORD_CHANGE_FAILED user=%s", user["username"])
        raise HTTPException(status_code=400, detail="Altes Passwort ist falsch")

    # Dev-Umgebung: Passwort-Komplexitaetspruefung deaktiviert (User-Entscheidung 2026-07-11).
    # from kiara.web.auth import validate_password
    #
    # pw_err = validate_password(data.new_password)
    # if pw_err:
    #     raise HTTPException(status_code=400, detail=pw_err)

    new_hash = hash_password(data.new_password)

    # Passwort-Update + Session-Invalidierung + CSRF-Rotation atomar.
    # Rotation auf DERSELBEN Session-Row — keine neue Session.
    new_csrf_plain: str | None = None
    with db_transaction() as conn:
        user_repo.set_password(user["id"], new_hash, conn=conn)

        # F-69: Alle anderen Sessions invalidieren (aktuelles Geraet behaelt Session)
        current_token = user.get("_session_token", "")
        if current_token:
            current_token_hash = hash_token(current_token)
            session_repo.delete_by_user(user["id"], except_token=current_token_hash, conn=conn)

            # CSRF-Token der verbleibenden Session rotieren.
            new_csrf_plain = secrets.token_hex(32)
            new_csrf_hash = hash_token(new_csrf_plain)
            if not session_repo.set_csrf_hash(current_token_hash, new_csrf_hash, conn=conn):
                # Session zwischen Auth-Dependency und Rotation geloescht/abgelaufen:
                # Transaktion abbrechen, damit Passwort-Update und Session-Invalidierung
                # zurueckrollen und kein unbrauchbarer Klartext-Token ausgeliefert wird.
                # Spec §6, §13: Response muss {"detail":"...","code":"not_authenticated"} sein.
                raise NotAuthenticatedError()
        else:
            session_repo.delete_by_user(user["id"], conn=conn)

    _reset_fails(rate_key)
    security_log.info("PASSWORD_CHANGED user=%s sessions_invalidated=true", user["username"])
    body = {"message": "Passwort geaendert"}
    if new_csrf_plain:
        body["csrf_token"] = new_csrf_plain
    return JSONResponse(content=body, headers={"Cache-Control": "no-store"})


@router.get("/csrf")
async def csrf(user=Depends(_get_current_user_for_csrf)):
    """Rotiert den CSRF-Token der aktuellen Session und liefert Klartext.

    Jeder Aufruf rotiert — der Klartext darf die DB nicht wieder verlassen,
    deshalb kann derselbe Token nach dem Setzen nicht nochmal abgerufen werden.
    """
    current_token = user.get("_session_token") or ""
    if not current_token:
        # Spec §6: 401 mit code=not_authenticated
        raise NotAuthenticatedError()

    current_token_hash = hash_token(current_token)
    new_csrf_plain = secrets.token_hex(32)
    new_csrf_hash = hash_token(new_csrf_plain)
    if not session_repo.set_csrf_hash(current_token_hash, new_csrf_hash):
        # Session zwischen Auth-Dependency und Rotation geloescht (Race).
        # Spec §6: 401 mit code=not_authenticated
        raise NotAuthenticatedError()

    return JSONResponse(
        content={"csrf_token": new_csrf_plain},
        headers={"Cache-Control": "no-store"},
    )
