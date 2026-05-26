"""Auto-Router fuer AdminFunctions — generiert FastAPI-Routen aus der Admin-Registry.

Stellt generate_admin_routes() bereit, das fuer jede AdminFunction mit http_method/http_path
eine vollstaendige FastAPI-Route erzeugt (Auth, Parameter-Extraktion, Typ-Coercion, Changelog).
Wird von kiara.web.admin beim App-Start aufgerufen. Importiert AdminFunction/AdminFunctionRegistry
(TYPE_CHECKING), AdminResult und auth/changelog aus kiara.web.
"""

import base64
import hashlib
import hmac
import json
import logging
import time
import unicodedata
from typing import TYPE_CHECKING, Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from kiara.admin.result import AdminResult
from kiara.shared.crypto import get_secret_key
from kiara.web.auth import get_admin_user
from kiara.web.config_changelog import log_config_change

if TYPE_CHECKING:
    from kiara.admin.function import AdminFunction
    from kiara.admin.registry import AdminFunctionRegistry

logger = logging.getLogger(__name__)

_CONFIRM_TOKEN_TTL_SECONDS = 600
_CONFIRM_TOKEN_HEADER = "x-kiara-confirm-token"
_CONFIRM_STRING_HEADER = "x-kiara-confirm-string"
_CONFIRM_PARAM_KEYS = {
    "_confirm_token",
    "confirm_token",
    "_confirmation_token",
    "confirmation_token",
    "_confirm_string",
    "confirm_string",
    "_confirmation_string",
    "confirmation_string",
}


def generate_admin_routes(app: FastAPI, registry: "AdminFunctionRegistry") -> None:
    """Erzeugt für jede AdminFunction mit http_method/http_path eine FastAPI-Route."""

    for func in registry.list_all():
        if not func.http_method or not func.http_path:
            continue  # Nur Chat-only Funktionen, keine HTTP-Route

        app.add_api_route(
            func.http_path,
            _create_endpoint(func),
            methods=[func.http_method],
        )
        logger.info("Route %s %s → %s", func.http_method, func.http_path, func.name)


def _create_endpoint(func: "AdminFunction"):
    """Erzeugt eine FastAPI-Endpunkt-Funktion für eine AdminFunction.

    Aufgaben:
    1. Auth: Depends(get_admin_user)
    2. Parameter-Extraktion (Path + Query/Body)
    3. Ausführung mit try/except
    4. Changelog-Eintrag bei Erfolg
    5. AdminResult -> JSON Response
    """

    async def endpoint(request: Request, user: dict = Depends(get_admin_user)):
        # Parameter aus Body/Query sammeln; Path-Params werden zuletzt gemerged,
        # damit die URL-adressierte Resource Vorrang hat (Schutz gegen Mass-Assignment
        # via Body/Query bei Update/Delete-Routen wie PUT /datasources/{ds_id}).
        params: dict = {}
        if func.http_method in ("POST", "PUT", "PATCH"):
            try:
                body = await request.json()
                params.update(body)
            except Exception:
                # Kein Body ist OK bei manchen POST-Requests (Content-Length: 0).
                # Bei vorhandenem Body mit ungueltigem JSON: Fehler zurueckgeben.
                cl = request.headers.get("content-length", "0")
                if cl != "0":
                    from fastapi.responses import JSONResponse

                    return JSONResponse(
                        status_code=400,
                        content={
                            "success": False,
                            "message": "Ungueltiges JSON im Request-Body",
                            "data": None,
                            "error": "JSON parse error",
                        },
                    )
        else:
            # multi_items() statt dict() um wiederholte Query-Parameter zu erhalten
            # z.B. ?ids=1&ids=2&ids=3 → {"ids": ["1", "2", "3"]}
            qp = {}
            for key, value in request.query_params.multi_items():
                if key in qp:
                    existing = qp[key]
                    if isinstance(existing, list):
                        existing.append(value)
                    else:
                        qp[key] = [existing, value]
                else:
                    qp[key] = value
            params.update(qp)
        # Path-Params zuletzt: nicht ueberschreibbar durch Body/Query
        params.update(dict(request.path_params))

        confirmation = _extract_http_confirmation(request, params)

        # Typ-Konvertierung für Path-/Query-Parameter (immer str aus URL)
        params = _coerce_params(params, func.parameters or {})

        # Nur Parameter durchlassen die im Schema definiert sind
        # (verhindert TypeError bei unerwarteten Query-Params wie ?_cache=123)
        allowed_keys = set((func.parameters or {}).get("properties", {}).keys())
        params = {k: v for k, v in params.items() if k in allowed_keys}

        confirmation_response = _require_http_confirmation(func, params, user, confirmation)
        if confirmation_response is not None:
            return confirmation_response

        # Ausführung mit Fehlerbehandlung
        try:
            result = await func.fn(user=user, **params)
        except Exception as exc:
            logger.exception("AdminFunction '%s' fehlgeschlagen", func.name)
            result = AdminResult.fail(f"Interner Fehler: {exc}", http_status=500)

        # Typ-Validierung: AdminFunction muss AdminResult zurueckgeben
        if not isinstance(result, AdminResult):
            logger.error(
                "AdminFunction '%s' gab %s statt AdminResult zurueck",
                func.name,
                type(result).__name__,
            )
            result = AdminResult.fail("Interner Fehler", http_status=500)

        # Changelog bei Erfolg (beide Felder muessen gesetzt sein)
        if func.changelog_area and func.changelog_action and result.success:
            _target_id = str(params.get(func.pk_param, "")) if func.pk_param else ""
            if func.pk_param and not _target_id:
                logger.warning(
                    "Changelog: pk_param '%s' nicht in Request-Params fuer '%s'",
                    func.pk_param,
                    func.name,
                )
            try:
                log_config_change(
                    user=user,
                    area=func.changelog_area,
                    action=func.changelog_action,
                    target_id=_target_id,
                    target_label=result.data.get("name", "")
                    if isinstance(result.data, dict)
                    else "",
                    source="gui",  # Auto-Router = GUI-Aufruf
                )
            except Exception:
                logger.exception("Changelog-Eintrag fuer '%s' fehlgeschlagen", func.name)

        # Error-Response: HTTP-Status-Code aus AdminResult
        # Auto-Detection: "nicht gefunden" in Message → 404 (wenn kein expliziter Status)
        if not result.success:
            from fastapi.responses import JSONResponse

            status = result.http_status
            if status == 400 and result.message:
                msg_lower = result.message.lower()
                if "nicht gefunden" in msg_lower or "not found" in msg_lower:
                    status = 404
            if func.raw_response:
                if isinstance(result.data, dict):
                    content = dict(result.data)
                elif result.data is not None:
                    content = {"data": result.data}
                else:
                    content = {}
                content["success"] = False
                if not content.get("message"):
                    content["message"] = result.message
                if not content.get("error"):
                    content["error"] = result.error or result.message
                return JSONResponse(
                    status_code=status,
                    content=content,
                )
            return JSONResponse(
                status_code=status,
                content={
                    "success": False,
                    "message": result.message,
                    "data": result.data,
                    "error": result.error,
                },
            )

        # Success-Response: raw für Frontend-Kompatibilität, wrapped für neue Endpoints
        if func.raw_response:
            return result.data if result.data is not None else {}
        return {
            "success": result.success,
            "message": result.message,
            "data": result.data,
            "error": result.error if not result.success else None,
        }

    endpoint.__name__ = func.name.replace(".", "_")
    return endpoint


def _extract_http_confirmation(request: Request, params: dict) -> dict:
    """Extrahiert Confirmation-Metadaten aus Headern oder internen Body-Feldern."""
    token = _request_header(request, _CONFIRM_TOKEN_HEADER)
    confirm_string = _request_header(request, _CONFIRM_STRING_HEADER)

    for key in list(params.keys()):
        if key not in _CONFIRM_PARAM_KEYS:
            continue
        value = params.pop(key)
        if key.endswith("token") and not token and isinstance(value, str):
            token = value
        elif key.endswith("string") and not confirm_string and isinstance(value, str):
            confirm_string = value

    return {"token": token or "", "confirm_string": confirm_string or ""}


def _request_header(request: Request, name: str) -> str:
    headers = getattr(request, "headers", None)
    if headers is None or not hasattr(headers, "get"):
        return ""
    if isinstance(headers, dict):
        value = {str(k).lower(): v for k, v in headers.items()}.get(name.lower())
        return value if isinstance(value, str) and value else ""
    candidates = (
        name,
        name.lower(),
        name.upper(),
        "-".join(part.capitalize() for part in name.split("-")),
    )
    for candidate in candidates:
        try:
            value = headers.get(candidate)
        except Exception:
            value = None
        if isinstance(value, str) and value:
            return value
    return ""


def _require_http_confirmation(
    func: "AdminFunction",
    params: dict,
    user: dict,
    confirmation: dict,
) -> JSONResponse | None:
    if not func.requires_confirmation:
        return None

    expected_confirm_string = _render_confirm_string(func, params)
    token = confirmation.get("token") or ""
    provided_confirm_string = confirmation.get("confirm_string") or ""

    if token and provided_confirm_string:
        if _normalize_confirm_string(provided_confirm_string) == _normalize_confirm_string(
            expected_confirm_string
        ):
            valid, reason = _verify_http_confirmation_token(func, params, user, token)
            if valid:
                return None
            logger.warning("HTTP-Confirmation fuer '%s' abgelehnt: %s", func.name, reason)
        else:
            logger.warning("HTTP-Confirmation fuer '%s' mit falschem Confirm-String", func.name)

    confirm_token = _create_http_confirmation_token(func, params, user)
    return JSONResponse(
        status_code=428,
        content={
            "success": False,
            "message": "Bestaetigung erforderlich",
            "data": None,
            "error": "confirmation_required",
            "requires_confirmation": True,
            "confirmation": {
                "function_name": func.name,
                "confirm_string": expected_confirm_string,
                "confirm_token": confirm_token,
                "expires_in": _CONFIRM_TOKEN_TTL_SECONDS,
                "params": _json_safe(params),
            },
        },
    )


def _render_confirm_string(func: "AdminFunction", params: dict) -> str:
    try:
        fmt_params = {
            key: str(value).lower() if isinstance(value, bool) else value
            for key, value in params.items()
        }
        confirm_string = (
            func.confirm_template.format(**fmt_params) if func.confirm_template else func.name
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("confirm_template.format() fehlgeschlagen fuer %s: %s", func.name, exc)
        confirm_string = func.name
    if not confirm_string or not str(confirm_string).strip():
        confirm_string = func.name
    return str(confirm_string)


def _normalize_confirm_string(value: str) -> str:
    # HTTP-Header sind per RFC 7230 ASCII; Confirm-Strings koennen aber Umlaute
    # enthalten (z.B. "loeschen" vs. "löschen"). NFKD-Folding strippt Diakritika
    # und macht den Vergleich transport-stabil — Server und Client koennen den
    # String mit oder ohne Umlaute senden.
    text = " ".join(str(value).split()).lower()
    folded = unicodedata.normalize("NFKD", text)
    return "".join(c for c in folded if ord(c) < 128 and not unicodedata.combining(c))


def _create_http_confirmation_token(func: "AdminFunction", params: dict, user: dict) -> str:
    payload = {
        "v": 1,
        "exp": int(time.time()) + _CONFIRM_TOKEN_TTL_SECONDS,
        "fn": func.name,
        "uid": str((user or {}).get("id", "")),
        "params_hash": _params_hash(params),
    }
    raw_payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = hmac.new(get_secret_key().encode("utf-8"), raw_payload, hashlib.sha256).digest()
    return f"{_b64encode(raw_payload)}.{_b64encode(signature)}"


def _verify_http_confirmation_token(
    func: "AdminFunction",
    params: dict,
    user: dict,
    token: str,
) -> tuple[bool, str]:
    try:
        payload_part, signature_part = token.split(".", 1)
        raw_payload = _b64decode(payload_part)
        provided_signature = _b64decode(signature_part)
    except Exception:
        return False, "token_format"

    expected_signature = hmac.new(
        get_secret_key().encode("utf-8"),
        raw_payload,
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(provided_signature, expected_signature):
        return False, "signature"

    try:
        payload = json.loads(raw_payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return False, "payload"

    if payload.get("v") != 1:
        return False, "version"
    if int(payload.get("exp") or 0) < int(time.time()):
        return False, "expired"
    if payload.get("fn") != func.name:
        return False, "function"
    if payload.get("uid") != str((user or {}).get("id", "")):
        return False, "user"
    if payload.get("params_hash") != _params_hash(params):
        return False, "params"
    return True, "ok"


def _params_hash(params: dict) -> str:
    canonical = json.dumps(
        _json_safe(params), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _json_safe(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def _coerce_params(params: dict, schema: dict) -> dict:
    """Konvertiert URL-Parameter (immer str) in die im JSON-Schema definierten Typen.

    Unterstützt: integer, number, boolean, array (mit items-Coercion).
    Strings bleiben unverändert. Bei fehlgeschlagener Konvertierung bleibt
    der Originalwert erhalten (Validierung erfolgt in der Funktion selbst).
    """
    properties = schema.get("properties", {})
    coerced = {}
    for key, value in params.items():
        prop = properties.get(key, {})
        prop_type = prop.get("type")

        # Array-Parameter: Single-Query-Werte (z.B. ?ids=123) in Liste wrappen,
        # da multi_items() bei nur einem Vorkommen den String belaesst.
        if prop_type == "array":
            if isinstance(value, str):
                value = [value]
            if isinstance(value, list):
                items_type = prop.get("items", {}).get("type")
                if items_type:
                    value = [_coerce_single(v, items_type) for v in value]
            coerced[key] = value
            continue

        if not isinstance(value, str):
            coerced[key] = value
            continue

        if prop_type == "integer":
            try:
                value = int(value)
            except (ValueError, TypeError):
                pass
        elif prop_type == "number":
            try:
                value = float(value)
            except (ValueError, TypeError):
                pass
        elif prop_type == "boolean":
            lower = value.lower()
            if lower in ("true", "1", "yes"):
                value = True
            elif lower in ("false", "0", "no"):
                value = False
        coerced[key] = value
    return coerced


def _coerce_single(value, type_name: str):
    """Konvertiert einen einzelnen Wert in den angegebenen Typ."""
    if not isinstance(value, str):
        return value
    if type_name == "integer":
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    if type_name == "number":
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    if type_name == "boolean":
        lower = value.lower()
        if lower in ("true", "1", "yes"):
            return True
        if lower in ("false", "0", "no"):
            return False
    return value
