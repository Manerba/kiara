"""Gemeinsame HTTP-Middleware fuer die KIara-Web-Apps.

Stellt CSRF-Schutz, Security-Headers und Cache-Control bereit.
Wird von app.py (User-UI) und admin_app.py (Admin-UI) genutzt.
"""

from __future__ import annotations

import hmac
import inspect
import logging
import secrets
from base64 import b64encode
from typing import Callable
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

SourceValue = str | Callable[[], str] | Callable[[Request], str]


CSRF_INVALID_BODY = {
    "detail": "CSRF-Token fehlt oder ungueltig",
    "code": "csrf_invalid",
}
NOT_AUTHENTICATED_BODY = {
    "detail": "Nicht angemeldet",
    "code": "not_authenticated",
}


class NotAuthenticatedError(Exception):
    """Wird in Endpoints geworfen, die ``NOT_AUTHENTICATED_BODY`` mit Top-Level
    ``code``-Feld liefern muessen (Spec CSRF §6, §13).

    FastAPIs ``HTTPException`` serialisiert ``detail`` immer in ``{"detail": ...}``
    — daher kann die Spec-Struktur mit ihr nicht ausgedrueckt werden. Der
    zugehoerige Handler (registriert via :func:`add_csrf_middleware`) wandelt
    diese Exception in eine ``JSONResponse`` mit ``NOT_AUTHENTICATED_BODY`` um.
    Gleichzeitig loest der ``raise`` den Rollback in ``db_transaction``-Bloecken
    korrekt aus.
    """

    pass


def _normalize_origin(value: str | None) -> str | None:
    """Normalisiert Origin/Referer auf scheme://host[:port]."""
    if not value:
        return None
    try:
        parts = urlsplit(value)
    except ValueError:
        return None
    if not parts.scheme or not parts.netloc:
        return None
    return f"{parts.scheme}://{parts.netloc}"


def add_csrf_middleware(
    app: FastAPI,
    exempt_prefixes: list[str] | None = None,
    *,
    token_exempt_paths: set[str] | None = None,
    trusted_origins: list[str] | None = None,
) -> None:
    """CSRF-Protection: Synchronizer-Token + Origin-Validation + X-Requested-With.

    Pruefreihenfolge (mutierende /api/*-Calls):
      1. Full-Exempts via `exempt_prefixes` (WOPI/OnlyOffice) werden durchgelassen.
      2. Origin/Referer gegen `trusted_origins` pruefen → 403 csrf_invalid.
      3. `X-Requested-With: XMLHttpRequest` Zusatzschicht → 403 csrf_invalid.
      4. Token-Exempts via `token_exempt_paths` (z.B. /api/auth/login) werden
         nach Origin+XRW durchgelassen, kein Session-/CSRF-Token noetig.
      5. Session-Cookie extrahieren → 401 not_authenticated bei Fehlschlag.
      6. Expiry-aware Session-Lookup → 401 not_authenticated bei fehlender Row.
      7. `sessions.csrf_token_hash` NULL → 401 (Alpha-Re-Login statt Lazy-Create).
      8. `X-CSRF-Token`-Header-Hash gegen DB-Hash → 403 csrf_invalid bei Mismatch.

    Args:
        app: Die FastAPI-App.
        exempt_prefixes: URL-Prefixe fuer Full-Exempts (WOPI, OnlyOffice).
        token_exempt_paths: Exakte Pfade, die nur den Token-Check ueberspringen
                            (Origin/XRW bleiben aktiv). V1: {"/api/auth/login"}.
        trusted_origins: Whitelist vollstaendiger Origins (scheme://host[:port]).
                         Leere/None-Liste = alle Origins abgelehnt (403).
    """

    # Exception-Handler fuer NotAuthenticatedError: Top-Level {detail, code}-Body.
    # Spec CSRF §6, §13: 401-Responses muessen code: "not_authenticated" enthalten.
    @app.exception_handler(NotAuthenticatedError)
    async def _not_authenticated_handler(request: Request, exc: NotAuthenticatedError):
        return JSONResponse(status_code=401, content=NOT_AUTHENTICATED_BODY)

    exempt = tuple(exempt_prefixes or [])
    token_exempt = frozenset(token_exempt_paths or ())
    origin_whitelist = frozenset(trusted_origins or ())

    def _is_exempt(path: str) -> bool:
        for p in exempt:
            if path == p:
                return True
            if p.endswith("/"):
                if path.startswith(p):
                    return True
            elif path.startswith(p + "/"):
                return True
        return False

    def _csrf_403() -> JSONResponse:
        return JSONResponse(status_code=403, content=CSRF_INVALID_BODY)

    def _auth_401() -> JSONResponse:
        return JSONResponse(status_code=401, content=NOT_AUTHENTICATED_BODY)

    # Lazy imports: Middleware wird beim Modul-Load registriert; die Auth-
    # Helper brauchen wir erst pro Request.
    from kiara.web.auth import _cookie_adapter, session_repo
    from kiara_auth.core import hash_token

    @app.middleware("http")
    async def csrf_protection(request: Request, call_next):
        if request.method not in (
            "POST",
            "PUT",
            "DELETE",
            "PATCH",
        ) or not request.url.path.startswith("/api/"):
            return await call_next(request)

        path = request.url.path

        # 1. Full-Exempt (WOPI/OnlyOffice)
        if _is_exempt(path):
            return await call_next(request)

        # 2. Origin/Referer gegen Whitelist
        request_origin = _normalize_origin(request.headers.get("Origin"))
        if not request_origin:
            request_origin = _normalize_origin(request.headers.get("Referer"))
        if not request_origin or request_origin not in origin_whitelist:
            return _csrf_403()

        # 3. X-Requested-With als Zusatzschicht
        if request.headers.get("X-Requested-With") != "XMLHttpRequest":
            return _csrf_403()

        # 4. Token-Exempts (Login)
        if path in token_exempt:
            return await call_next(request)

        # 5. Session-Cookie
        session_token = _cookie_adapter.extract_token_from_request(request)
        if not session_token:
            return _auth_401()

        # 6./7. Expiry-aware Session-Lookup + Hash-Check
        session_row = session_repo.get_valid_by_token(hash_token(session_token))
        if not session_row:
            return _auth_401()
        stored_hash = session_row.get("csrf_token_hash")
        if not stored_hash:
            return _auth_401()

        # 8. X-CSRF-Token Header vergleichen
        submitted = request.headers.get("X-CSRF-Token") or ""
        if not submitted:
            return _csrf_403()
        if not hmac.compare_digest(hash_token(submitted), stored_hash.strip()):
            return _csrf_403()

        return await call_next(request)


def add_security_headers_middleware(
    app: FastAPI,
    extra_img_src: str = "",
    frame_src: str = "'none'",
    extra_connect_src: SourceValue = "",
    extra_script_src: SourceValue = "",
    hsts_max_age: int = 0,
    hsts_include_subdomains: bool = True,
    *,
    extra_style_src: SourceValue = "",
    csp_strict: bool = False,
    csp_report_only: bool = False,
    csp_report_uri: str = "",
) -> None:
    """Security-Headers: CSP, X-Content-Type-Options, Referrer-Policy, X-Frame-Options, HSTS.

    Args:
        app: Die FastAPI-App.
        extra_img_src: Zusaetzliche img-src Werte (z.B. "blob:" fuer User-UI).
        frame_src: frame-src Wert ("'none'" fuer Admin, "*" fuer User/OnlyOffice).
        extra_connect_src: Zusaetzliche connect-src Werte (z.B. Chat-Service-URL).
        extra_script_src: Zusaetzliche script-src Werte. String oder Callable
                          das einen String liefert (fuer dynamische Werte wie OnlyOffice-URL).
        hsts_max_age: Strict-Transport-Security max-age in Sekunden.
                      0 (Default) = HSTS deaktiviert — sinnvoll fuer interne HTTP-Deployments.
                      Bei HTTPS-Deployment z.B. 31536000 (1 Jahr) setzen.
        hsts_include_subdomains: includeSubDomains-Direktive ergaenzen (nur wenn hsts_max_age > 0).
        extra_style_src: Zusaetzliche style-src Werte.
        csp_strict: Erzwingt Strict-CSP ohne unsafe-inline fuer Script/Style.
        csp_report_only: Setzt zusaetzlich einen Strict-CSP-Report-Only-Header.
        csp_report_uri: Optionaler report-uri-Wert fuer Strict-CSP.
    """
    hsts_header_value = None
    if hsts_max_age > 0:
        hsts_header_value = f"max-age={hsts_max_age}"
        if hsts_include_subdomains:
            hsts_header_value += "; includeSubDomains"

    def _generate_csp_nonce() -> str:
        return b64encode(secrets.token_bytes(16)).decode("ascii")

    def _resolve_source(value: SourceValue, request: Request, label: str) -> str:
        if not value:
            return ""
        if not callable(value):
            return str(value).strip()
        try:
            signature = inspect.signature(value)
            resolved = value() if len(signature.parameters) == 0 else value(request)
        except Exception:
            logger.warning(
                "CSP-Zusatzquelle %s konnte nicht aufgeloest werden", label, exc_info=True
            )
            return ""
        return str(resolved or "").strip()

    def _build_csp(
        nonce: str,
        *,
        strict: bool,
        script_extra: str,
        style_extra: str,
        connect_extra: str,
        report_uri: str = "",
    ) -> str:
        img_src = f"'self' data: {extra_img_src}".strip() if extra_img_src else "'self' data:"
        connect_src = f"'self' {connect_extra}".strip() if connect_extra else "'self'"
        script_nonce = f"'nonce-{nonce}'" if nonce else ""
        style_nonce = f"'nonce-{nonce}'" if nonce else ""
        if strict:
            script_src = " ".join(
                part for part in ("'self'", script_nonce, script_extra) if part
            ).strip()
            style_src = " ".join(
                part for part in ("'self'", style_nonce, style_extra) if part
            ).strip()
        else:
            script_src = " ".join(
                part for part in ("'self'", "'unsafe-inline'", script_extra) if part
            ).strip()
            style_src = " ".join(
                part for part in ("'self'", "'unsafe-inline'", style_extra) if part
            ).strip()
        directives = [
            "default-src 'self'",
            f"script-src {script_src}",
            f"style-src {style_src}",
            f"img-src {img_src}",
            f"connect-src {connect_src}",
            f"frame-src {frame_src}",
        ]
        if report_uri:
            directives.append(f"report-uri {report_uri}")
        return "; ".join(directives) + ";"

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        request.state.csp_nonce = _generate_csp_nonce()
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Frame-Options"] = "DENY" if frame_src == "'none'" else "SAMEORIGIN"
        if hsts_header_value is not None:
            response.headers["Strict-Transport-Security"] = hsts_header_value
        # CSP nur auf HTML-Dokumente, nicht auf statische Assets (JS/CSS/Bilder)
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            script_extra = _resolve_source(extra_script_src, request, "extra_script_src")
            style_extra = _resolve_source(extra_style_src, request, "extra_style_src")
            connect_extra = _resolve_source(extra_connect_src, request, "extra_connect_src")
            nonce = getattr(request.state, "csp_nonce", "")
            response.headers["Content-Security-Policy"] = _build_csp(
                nonce,
                strict=csp_strict,
                script_extra=script_extra,
                style_extra=style_extra,
                connect_extra=connect_extra,
                report_uri=csp_report_uri if csp_strict else "",
            )
            if csp_report_only:
                response.headers["Content-Security-Policy-Report-Only"] = _build_csp(
                    nonce,
                    strict=True,
                    script_extra=script_extra,
                    style_extra=style_extra,
                    connect_extra=connect_extra,
                    report_uri=csp_report_uri,
                )
        return response


def add_no_cache_static_middleware(app: FastAPI) -> None:
    """Setzt Cache-Control: no-store fuer /static/ und /api/ Pfade."""

    @app.middleware("http")
    async def no_cache_static(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        elif request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response
