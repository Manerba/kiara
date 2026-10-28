"""OIDC/JWKS-Authentifizierung fuer Retrieval-only REST."""

from __future__ import annotations

import email.utils
import hashlib
import hmac
import json
import logging
import os
import re
import ssl
import threading
import time
from typing import Any
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener
from uuid import uuid4

import jwt

from kiara.repositories.external_identity import (
    OPENDESK_IDENTITY_CLAIM,
    OPENDESK_PROVIDER,
    ExternalIdentityRepository,
)
from kiara.shared.ad_sid import ADSidError, normalize_ad_sid
from kiara.shared.crypto import get_secret_key

logger = logging.getLogger(__name__)
security_logger = logging.getLogger("kiara.security.oidc")
_SECURITY_HANDLER_LOCK = threading.Lock()
_SECURITY_HANDLER_INSTALLED = False

_CACHE_MAX_AGE_RE = re.compile(r"(?:^|,)\s*max-age=(\d+)\s*(?:,|$)", re.IGNORECASE)
_DEFAULT_JWKS_TTL_SECONDS = 300
_ERROR_REFRESH_TTL_SECONDS = 30
_HTTP_TIMEOUT_SECONDS = 5.0
_UNKNOWN_KID_REFRESH_TTL_SECONDS = 30


class OIDCAuthError(Exception):
    """Basisklasse fuer kontrollierte OIDC-Auth-Fehler."""

    reason: str = "oidc_auth_failed"


class OIDCUnauthorizedError(OIDCAuthError):
    """Token oder Principal ist ungueltig."""


class OIDCUnavailableError(OIDCAuthError):
    """OIDC/JWKS/DB-Backend ist temporaer nicht verfuegbar."""


class _HTTPSOnlyRedirectHandler(HTTPRedirectHandler):
    """Redirect-Handler, der OIDC-Redirects auf Nicht-HTTPS blockt."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != "https":
            exc = OIDCUnavailableError("OIDC-Redirect auf Nicht-HTTPS blockiert")
            exc.reason = "oidc_redirect_non_https"
            raise exc
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def ensure_security_log_handler(path: str = "/var/log/kiara/security.log") -> None:
    """Initialisiert verpflichtendes tokenfreies Security-File-Logging."""
    global _SECURITY_HANDLER_INSTALLED
    if _SECURITY_HANDLER_INSTALLED:
        return
    with _SECURITY_HANDLER_LOCK:
        if _SECURITY_HANDLER_INSTALLED:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            handler = logging.FileHandler(path, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
            )
            security_logger.addHandler(handler)
            security_logger.setLevel(logging.INFO)
        except OSError as exc:
            error = OIDCUnavailableError("OIDC Security-Log konnte nicht initialisiert werden")
            error.reason = "security_audit_unavailable"
            raise error from exc
        _SECURITY_HANDLER_INSTALLED = True


def sid_audit_hmac(normalized_sid: str) -> str:
    """HMAC-SHA256 der normalisierten SID mit KIara-Secret."""
    secret = get_secret_key().encode("utf-8")
    return hmac.new(secret, normalized_sid.encode("utf-8"), hashlib.sha256).hexdigest()


def audit_oidc_decision(
    *,
    request_id: str,
    path: str,
    outcome: str,
    reason: str,
    user_id: int | None = None,
    ad_sid: str | None = None,
) -> None:
    """Schreibt tokenfreies Audit fuer OIDC-Auth-Entscheidungen."""
    ensure_security_log_handler()
    sid_hmac = sid_audit_hmac(ad_sid) if ad_sid else ""
    security_logger.info(
        "OIDC_AUTH request_id=%s path=%s outcome=%s reason=%s user_id=%s ad_sid_hmac=%s",
        request_id,
        path,
        outcome,
        reason,
        user_id if user_id is not None else "",
        sid_hmac,
    )


class OIDCRetrievalAuthenticator:
    """Validiert openDesk-Keycloak Access Tokens und baut KIara-Principals."""

    def __init__(
        self,
        config: dict,
        *,
        external_identity_repo: ExternalIdentityRepository | None = None,
        now_func=time.time,
        http_get_json=None,
    ) -> None:
        self._config = config
        self._oidc_config = (config.get("web") or {}).get("oidc_retrieval") or {}
        self._repo = external_identity_repo or ExternalIdentityRepository()
        self._now = now_func
        self._http_get_json = http_get_json or self._http_get_json_default
        self._refresh_lock = threading.Lock()
        self._jwks_by_kid: dict[str, dict[str, Any]] = {}
        self._jwks_expires_at = 0.0
        self._negative_refresh_until = 0.0
        self._unknown_kid_refresh_until = 0.0
        self._discovered_jwks_url: str | None = None

    @property
    def enabled(self) -> bool:
        return self._oidc_config.get("enabled") is True

    def authenticate(self, authorization_header: str, *, path: str) -> dict[str, Any]:
        request_id = str(uuid4())
        normalized_sid = None
        try:
            token = self._extract_bearer_token(authorization_header)
            claims = self._validate_token(token)
            normalized_sid = self._normalize_identity_claim(claims)
            principal = self._resolve_principal(normalized_sid)
        except OIDCUnavailableError as exc:
            audit_oidc_decision(
                request_id=request_id,
                path=path,
                outcome="unavailable",
                reason=getattr(exc, "reason", "oidc_unavailable"),
                ad_sid=normalized_sid,
            )
            raise
        except OIDCAuthError as exc:
            audit_oidc_decision(
                request_id=request_id,
                path=path,
                outcome="deny",
                reason=getattr(exc, "reason", "unauthorized"),
                ad_sid=normalized_sid,
            )
            raise

        audit_oidc_decision(
            request_id=request_id,
            path=path,
            outcome="allow",
            reason="ok",
            user_id=principal.get("id"),
            ad_sid=normalized_sid,
        )
        return principal

    def _extract_bearer_token(self, authorization_header: str) -> str:
        if not isinstance(authorization_header, str) or not authorization_header.startswith(
            "Bearer "
        ):
            exc = OIDCUnauthorizedError("Authorization Bearer Token fehlt")
            exc.reason = "missing_bearer"
            raise exc
        token = authorization_header[7:].strip()
        if not token or token.startswith("orag_"):
            exc = OIDCUnauthorizedError("Ungueltiger Bearer Token")
            exc.reason = "invalid_bearer"
            raise exc
        return token

    def _validate_token(self, token: str) -> dict[str, Any]:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            error = OIDCUnauthorizedError("JWT-Header ungueltig")
            error.reason = "invalid_jwt_header"
            raise error from exc

        algorithms = self._oidc_config.get("algorithms") or ["RS256"]
        alg = header.get("alg")
        if alg not in algorithms:
            exc = OIDCUnauthorizedError("JWT-Algorithmus nicht erlaubt")
            exc.reason = "invalid_algorithm"
            raise exc
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid:
            exc = OIDCUnauthorizedError("JWT kid fehlt")
            exc.reason = "missing_kid"
            raise exc

        key = self._key_for_kid(kid)
        try:
            claims = jwt.decode(
                token,
                key=key,
                algorithms=algorithms,
                audience=self._oidc_config.get("audience"),
                issuer=self._oidc_config.get("issuer"),
                leeway=int(self._oidc_config.get("clock_skew_seconds", 60)),
                options={"require": ["exp", "iss", "aud"]},
            )
        except jwt.PyJWTError as exc:
            error = OIDCUnauthorizedError("JWT ungueltig")
            error.reason = "jwt_rejected"
            raise error from exc

        self._validate_claim_policy(claims)
        return claims

    def _validate_claim_policy(self, claims: dict[str, Any]) -> None:
        if claims.get("typ") != "Bearer":
            exc = OIDCUnauthorizedError("Token-Typ ist kein Access Token")
            exc.reason = "invalid_token_type"
            raise exc

        binding_claim = self._oidc_config.get("client_binding_claim") or "azp"
        binding_value = claims.get(binding_claim)
        if not isinstance(binding_value, str) or binding_value != self._oidc_config.get(
            "client_binding_value"
        ):
            exc = OIDCUnauthorizedError("Client-Bindung ungueltig")
            exc.reason = "invalid_client_binding"
            raise exc

        identity_claim = self._oidc_config.get("identity_claim") or OPENDESK_IDENTITY_CLAIM
        identity_value = claims.get(identity_claim)
        if not isinstance(identity_value, str):
            exc = OIDCUnauthorizedError("AD-SID-Claim fehlt oder hat falschen Typ")
            exc.reason = "invalid_identity_claim"
            raise exc

        self._validate_required_scopes(claims)
        self._validate_required_roles(claims)

    def _validate_required_scopes(self, claims: dict[str, Any]) -> None:
        required = self._oidc_config.get("required_scopes") or []
        if not required:
            return
        scope_value = claims.get("scope")
        if not isinstance(scope_value, str):
            exc = OIDCUnauthorizedError("Scope-Claim hat falschen Typ")
            exc.reason = "invalid_scope_claim"
            raise exc
        scopes = set(scope_value.split())
        if any(scope not in scopes for scope in required):
            exc = OIDCUnauthorizedError("Pflicht-Scope fehlt")
            exc.reason = "missing_required_scope"
            raise exc

    def _validate_required_roles(self, claims: dict[str, Any]) -> None:
        required = self._oidc_config.get("required_roles") or []
        if not required:
            return
        roles = set()
        realm_access = claims.get("realm_access") or {}
        if not isinstance(realm_access, dict):
            exc = OIDCUnauthorizedError("realm_access hat falschen Typ")
            exc.reason = "invalid_role_claim"
            raise exc
        realm_roles = realm_access.get("roles") or []
        if not isinstance(realm_roles, list) or any(not isinstance(r, str) for r in realm_roles):
            exc = OIDCUnauthorizedError("realm_access.roles hat falschen Typ")
            exc.reason = "invalid_role_claim"
            raise exc
        roles.update(realm_roles)

        resource_access = claims.get("resource_access") or {}
        if not isinstance(resource_access, dict):
            exc = OIDCUnauthorizedError("resource_access hat falschen Typ")
            exc.reason = "invalid_role_claim"
            raise exc
        for access in resource_access.values():
            if not isinstance(access, dict):
                exc = OIDCUnauthorizedError("resource_access Eintrag hat falschen Typ")
                exc.reason = "invalid_role_claim"
                raise exc
            client_roles = access.get("roles") or []
            if not isinstance(client_roles, list) or any(
                not isinstance(role, str) for role in client_roles
            ):
                exc = OIDCUnauthorizedError("resource_access.roles hat falschen Typ")
                exc.reason = "invalid_role_claim"
                raise exc
            roles.update(client_roles)

        if any(role not in roles for role in required):
            exc = OIDCUnauthorizedError("Pflicht-Rolle fehlt")
            exc.reason = "missing_required_role"
            raise exc

    def _normalize_identity_claim(self, claims: dict[str, Any]) -> str:
        identity_claim = self._oidc_config.get("identity_claim") or OPENDESK_IDENTITY_CLAIM
        value = claims.get(identity_claim)
        format_name = self._oidc_config.get("identity_claim_format") or "ad_object_sid_or_base64"
        allow_base64 = format_name in {"ad_object_sid_base64", "ad_object_sid_or_base64"}
        try:
            return normalize_ad_sid(value, allow_base64=allow_base64)
        except ADSidError as exc:
            error = OIDCUnauthorizedError("AD-SID ist ungueltig")
            error.reason = "invalid_ad_sid"
            raise error from exc

    def _resolve_principal(self, normalized_sid: str) -> dict[str, Any]:
        try:
            row = self._repo.get_mapping_with_user(
                provider=OPENDESK_PROVIDER,
                issuer=self._oidc_config.get("issuer"),
                identity_claim=self._oidc_config.get("identity_claim") or OPENDESK_IDENTITY_CLAIM,
                external_id=normalized_sid,
            )
        except OIDCAuthError:
            raise
        except Exception as exc:
            error = OIDCUnavailableError("Identity-Mapping nicht verfuegbar")
            error.reason = "identity_mapping_unavailable"
            raise error from exc

        if not row or row.get("mapping_active") is not True or row.get("user_active") is not True:
            exc = OIDCUnauthorizedError("Principal nicht akzeptiert")
            exc.reason = "principal_rejected"
            raise exc

        is_admin = row.get("is_admin") is True
        if is_admin and self._oidc_config.get("allow_admin_full_access") is not True:
            exc = OIDCUnauthorizedError("OIDC-Admin nicht freigegeben")
            exc.reason = "admin_rejected"
            raise exc

        try:
            from kiara.shared.db import db_query
            from kiara.web.config_utils import load_user_permissions

            with db_query() as db:
                group_ids, datasource_ids = load_user_permissions(db, int(row["id"]))
        except Exception as exc:
            error = OIDCUnavailableError("User-Berechtigungen nicht verfuegbar")
            error.reason = "permission_backend_unavailable"
            raise error from exc

        return {
            "id": int(row["id"]),
            "username": row.get("username") or "",
            "email": row.get("email") or "",
            "is_admin": is_admin,
            "group_ids": group_ids,
            "datasource_ids": datasource_ids,
            "auth_source": "oidc",
            "oidc_issuer": self._oidc_config.get("issuer"),
            "oidc_claim": self._oidc_config.get("identity_claim") or OPENDESK_IDENTITY_CLAIM,
            "oidc_external_id": normalized_sid,
        }

    def _key_for_kid(self, kid: str):
        now = self._now()
        if kid in self._jwks_by_kid and now < self._jwks_expires_at:
            return self._jwk_to_key(self._jwks_by_kid[kid])
        if now < self._unknown_kid_refresh_until and kid not in self._jwks_by_kid:
            exc = OIDCUnauthorizedError("JWT kid nicht im JWKS")
            exc.reason = "unknown_kid"
            raise exc
        if now < self._negative_refresh_until and kid not in self._jwks_by_kid:
            exc = OIDCUnavailableError("JWKS-Refresh temporaer gesperrt")
            exc.reason = "jwks_refresh_rate_limited"
            raise exc

        with self._refresh_lock:
            now = self._now()
            if kid in self._jwks_by_kid and now < self._jwks_expires_at:
                return self._jwk_to_key(self._jwks_by_kid[kid])
            if now < self._unknown_kid_refresh_until and kid not in self._jwks_by_kid:
                exc = OIDCUnauthorizedError("JWT kid nicht im JWKS")
                exc.reason = "unknown_kid"
                raise exc
            try:
                self._refresh_jwks_locked()
            except OIDCUnavailableError:
                self._negative_refresh_until = self._now() + _ERROR_REFRESH_TTL_SECONDS
                if kid in self._jwks_by_kid and now < self._jwks_expires_at:
                    return self._jwk_to_key(self._jwks_by_kid[kid])
                raise

            if kid not in self._jwks_by_kid:
                self._unknown_kid_refresh_until = self._now() + _UNKNOWN_KID_REFRESH_TTL_SECONDS
                exc = OIDCUnauthorizedError("JWT kid nicht im JWKS")
                exc.reason = "unknown_kid"
                raise exc
            return self._jwk_to_key(self._jwks_by_kid[kid])

    def _refresh_jwks_locked(self) -> None:
        jwks_url = self._jwks_url()
        data, headers = self._http_get_json(jwks_url)
        keys = data.get("keys") if isinstance(data, dict) else None
        if not isinstance(keys, list) or not keys:
            exc = OIDCUnavailableError("JWKS enthaelt keine Keys")
            exc.reason = "empty_jwks"
            raise exc
        by_kid = {}
        for jwk in keys:
            if self._is_usable_jwk(jwk):
                by_kid[jwk["kid"]] = jwk
        if not by_kid:
            exc = OIDCUnavailableError("JWKS enthaelt keine nutzbaren kid Keys")
            exc.reason = "empty_jwks"
            raise exc
        self._jwks_by_kid = by_kid
        self._jwks_expires_at = self._now() + self._ttl_from_headers(headers)
        self._negative_refresh_until = 0.0

    def _jwks_url(self) -> str:
        discovery_url = self._oidc_config.get("discovery_url") or ""
        if discovery_url:
            if self._discovered_jwks_url:
                return self._discovered_jwks_url
            discovery, _ = self._http_get_json(discovery_url)
            if discovery.get("issuer") != self._oidc_config.get("issuer"):
                exc = OIDCUnavailableError("Discovery-Issuer passt nicht zur Config")
                exc.reason = "discovery_issuer_mismatch"
                raise exc
            jwks_uri = discovery.get("jwks_uri")
            if not isinstance(jwks_uri, str) or not self._is_https_url(jwks_uri):
                exc = OIDCUnavailableError("Discovery-JWKS-URI ist ungueltig")
                exc.reason = "invalid_discovery_jwks_uri"
                raise exc
            self._discovered_jwks_url = jwks_uri
            return jwks_uri

        jwks_url = self._oidc_config.get("jwks_url") or ""
        if not self._is_https_url(jwks_url):
            exc = OIDCUnavailableError("JWKS-URL ist ungueltig")
            exc.reason = "invalid_jwks_url"
            raise exc
        return jwks_url

    def _http_get_json_default(self, url: str) -> tuple[dict[str, Any], dict[str, str]]:
        if not self._is_https_url(url):
            exc = OIDCUnavailableError("OIDC-URL ist nicht HTTPS")
            exc.reason = "non_https_oidc_url"
            raise exc
        ca_bundle = self._oidc_config.get("ca_bundle_path") or None
        context = ssl.create_default_context(cafile=ca_bundle)
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "kiara-oidc"})
        try:
            opener = build_opener(HTTPSHandler(context=context), _HTTPSOnlyRedirectHandler)
            with opener.open(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:
                final_url = response.geturl()
                if not self._is_https_url(final_url):
                    error = OIDCUnavailableError("OIDC-Redirect auf Nicht-HTTPS blockiert")
                    error.reason = "oidc_redirect_non_https"
                    raise error
                body = response.read(1024 * 1024)
                headers = {k.lower(): v for k, v in response.headers.items()}
        except OIDCUnavailableError:
            raise
        except Exception as exc:
            error = OIDCUnavailableError("OIDC-HTTP-Aufruf fehlgeschlagen")
            error.reason = "oidc_http_failed"
            raise error from exc
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            error = OIDCUnavailableError("OIDC-JSON ungueltig")
            error.reason = "oidc_json_invalid"
            raise error from exc
        if not isinstance(parsed, dict):
            error = OIDCUnavailableError("OIDC-JSON ist kein Objekt")
            error.reason = "oidc_json_invalid"
            raise error
        return parsed, headers

    def _ttl_from_headers(self, headers: dict[str, str]) -> int:
        cache_control = headers.get("cache-control", "")
        match = _CACHE_MAX_AGE_RE.search(cache_control)
        if match:
            try:
                return max(1, min(int(match.group(1)), 3600))
            except ValueError:
                pass
        expires = headers.get("expires")
        if expires:
            try:
                expires_dt = email.utils.parsedate_to_datetime(expires)
                ttl = int(expires_dt.timestamp() - self._now())
                if ttl > 0:
                    return max(1, min(ttl, 3600))
            except (TypeError, ValueError, OverflowError):
                pass
        return _DEFAULT_JWKS_TTL_SECONDS

    @staticmethod
    def _is_https_url(value: str) -> bool:
        parsed = urlparse(value)
        return parsed.scheme == "https" and bool(parsed.netloc)

    @staticmethod
    def _is_usable_jwk(jwk: Any) -> bool:
        if not isinstance(jwk, dict) or not isinstance(jwk.get("kid"), str):
            return False
        if jwk.get("kty") != "RSA":
            return False
        use = jwk.get("use")
        if use not in (None, "sig"):
            return False
        alg = jwk.get("alg")
        return alg in (None, "RS256")

    @staticmethod
    def _jwk_to_key(jwk: dict[str, Any]):
        try:
            return jwt.PyJWK.from_dict(jwk).key
        except Exception as exc:
            error = OIDCUnavailableError("JWK konnte nicht geladen werden")
            error.reason = "invalid_jwk"
            raise error from exc
