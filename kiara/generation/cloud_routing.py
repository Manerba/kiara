"""Cloud-Routing und Backend-Aufloesung fuer Chat-API, Admin-Chat und MCP.

Stellt wiederverwendbare Routing- und Resolver-Funktionen bereit, die von
api_chat.py, api_admin_chat.py und mcp_server.py gemeinsam genutzt werden.

Restriction-Level-Logik:
    - _load_restriction_levels(): Laedt Restriction-Levels fuer DS-IDs
    - _is_cloud_blocked(): Prueft ob Cloud-Nutzung gesperrt ist (fail-closed)
    - _create_chain_backend(): Erstellt Backend mit trusted/restriction-Logik
    - _resolve_backend_dict(): Loest Backend-Dict fuer Agent/Admin auf
"""

import logging
import threading
import time

from kiara.backend_selector import get_all_active, get_backend_by_id, get_next_backend
from kiara.generation._validators import (
    is_cloud_blocked_by_restrictions,
    needs_anonymization,
)
from kiara.generation.backends import LLMBackend, create_backend, wrap_with_anonymization
from kiara.shared.constants import CLOUD_KEYWORDS
from kiara.shared.enums import RoutingMode

logger = logging.getLogger(__name__)

_RESTRICTION_ERROR_TTL_SECS = 10.0
_restriction_error_cache: dict[tuple[int, ...], tuple[float, int, str]] = {}
_restriction_error_lock = threading.Lock()


def _restriction_cache_key(filter_datasource_ids: list[int] | None) -> tuple[int, ...]:
    return tuple(sorted({int(ds_id) for ds_id in (filter_datasource_ids or [])}))


def _clear_restriction_error_cache(key: tuple[int, ...]) -> None:
    with _restriction_error_lock:
        _restriction_error_cache.pop(key, None)


def _restriction_failure_in_cooldown(key: tuple[int, ...]) -> bool:
    now = time.monotonic()
    with _restriction_error_lock:
        cached = _restriction_error_cache.get(key)
        if not cached:
            return False
        fail_until, attempts, last_error = cached
        if attempts >= 2 and now < fail_until:
            logger.warning(
                "Restriction-Level-Abfrage im Cooldown fuer DS %s: %s",
                list(key),
                last_error,
            )
            return True
        if now >= fail_until:
            _restriction_error_cache.pop(key, None)
    return False


def _record_restriction_failure(key: tuple[int, ...], exc: RuntimeError) -> None:
    now = time.monotonic()
    with _restriction_error_lock:
        _old_until, old_attempts, _old_error = _restriction_error_cache.get(key, (0.0, 0, ""))
        attempts = old_attempts + 1
        ttl = min(_RESTRICTION_ERROR_TTL_SECS * attempts, 60.0)
        _restriction_error_cache[key] = (now + ttl, attempts, str(exc))


def _parse_routing_mode(raw_mode: object) -> RoutingMode:
    """Parse config value to RoutingMode; invalid values fail closed to local."""
    if isinstance(raw_mode, RoutingMode):
        return raw_mode
    try:
        return RoutingMode(str(raw_mode))
    except (TypeError, ValueError):
        logger.warning(
            "Ungueltiger Cloud-Routing-Modus %r; route lokal (fail-closed)",
            raw_mode,
        )
        return RoutingMode.LOCAL


def _degradation_feature_available(feature: str, default: bool = True) -> bool:
    """Read DegradationManager feature state when available.

    CLI/worker contexts may not have the web health manager initialised. In that
    case the router keeps its config/backend based behaviour.
    """
    try:
        from kiara.web.api_health import _get_degradation_manager

        dm = _get_degradation_manager()
        return bool(dm.is_feature_available(feature))
    except Exception:
        return default


def _load_restriction_levels(filter_datasource_ids: list[int] | None) -> list[str]:
    """Laedt Restriction-Levels fuer eine Liste von Datasource-IDs.

    Args:
        filter_datasource_ids: DS-IDs fuer die Levels abzufragen sind.

    Returns:
        Liste der Restriction-Levels (eines pro DS-ID).

    Raises:
        RuntimeError: Bei DB-Fehler oder fehlenden Rows.
    """
    if not filter_datasource_ids:
        return []

    from kiara.repositories.datasource import DatasourceRepository

    try:
        ds_repo = DatasourceRepository()
        rows = ds_repo.get_restriction_levels(list(filter_datasource_ids))
    except Exception as exc:
        raise RuntimeError(f"Restriction-Level-Abfrage fehlgeschlagen: {exc}") from exc

    if len(rows) != len(set(filter_datasource_ids)):
        raise RuntimeError(
            f"Restriction-Level: {len(rows)} Rows fuer "
            f"{len(set(filter_datasource_ids))} IDs — fehlende/stale Datasources"
        )

    return [r["restriction_level"] for r in rows]


def _route_backend(config: dict, question: str, filter_datasource_ids: list[int] | None) -> str:
    """Entscheidet ob lokal oder Cloud genutzt wird.

    Args:
        config: Globale Config.
        question: Die Benutzerfrage.
        filter_datasource_ids: Datasource-IDs des Users (fuer Restriction-Check).

    Returns:
        RoutingMode.LOCAL oder RoutingMode.CLOUD.
    """
    cloud_cfg = config.get("cloud", {})
    if not cloud_cfg.get("enabled"):
        return RoutingMode.LOCAL

    # DegradationManager: Bei NO_CLOUD-Modus lokal routen. Bei NO_LLM darf
    # eine gesunde Cloud explizit als Fallback genutzt werden.
    if not _degradation_feature_available("cloud", default=True):
        return RoutingMode.LOCAL
    llm_available = _degradation_feature_available("llm", default=True)

    # Verfuegbarkeit pruefen OHNE den Round-Robin-Selector zu konsumieren
    if not get_all_active("cloud"):
        return RoutingMode.LOCAL

    mode = _parse_routing_mode(cloud_cfg.get("mode", RoutingMode.AUTO))

    if mode == RoutingMode.LOCAL:
        return RoutingMode.LOCAL

    if _is_cloud_blocked(config, filter_datasource_ids):
        return RoutingMode.LOCAL

    if mode == RoutingMode.CLOUD:
        return RoutingMode.CLOUD

    if not llm_available:
        return RoutingMode.CLOUD

    # Auto-Modus: strategische Keywords pruefen (Konstante + Admin-Config)
    q_lower = question.lower()
    configured_keywords = cloud_cfg.get("routing", {}).get("cloud_keywords") or []
    extra_keywords = {str(kw).strip().lower() for kw in configured_keywords if str(kw).strip()}
    for kw in CLOUD_KEYWORDS | extra_keywords:
        if kw in q_lower:
            return RoutingMode.CLOUD

    return RoutingMode.LOCAL


def _is_cloud_blocked(config: dict, filter_datasource_ids: list[int] | None) -> bool:
    """Prueft ob Restriction-Levels der User-Datenquellen Cloud-Nutzung sperren.

    Fail-closed: Bei DB-Fehlern wird Cloud blockiert.
    """
    if not filter_datasource_ids:
        return False

    cache_key = _restriction_cache_key(filter_datasource_ids)
    if _restriction_failure_in_cooldown(cache_key):
        return True

    try:
        levels = _load_restriction_levels(filter_datasource_ids)
    except RuntimeError as exc:
        logger.warning("Restriction-Level-Abfrage fehlgeschlagen: %s", exc)
        _record_restriction_failure(cache_key, exc)
        return True  # Fail-closed

    _clear_restriction_error_cache(cache_key)
    return is_cloud_blocked_by_restrictions(levels)


def _load_cloud_restriction_levels(
    filter_datasource_ids: list[int] | None,
    *,
    explicit: bool,
) -> list[str]:
    """Zentraler Restriction-Loader fuer explizite und automatische Cloud-Pfade."""
    try:
        return _load_restriction_levels(filter_datasource_ids)
    except RuntimeError as exc:
        if explicit:
            raise ValueError("Restriction-Level-Abfrage fehlgeschlagen") from exc
        raise


def _resolve_backend_dict(
    config: dict,
    llm_backend_id: int | None,
    filter_datasource_ids: list[int] | None,
) -> dict:
    """Loest das Backend-Dict fuer Agent- und Admin-Chat auf.

    Regeln:
    1. Explizite Backend-ID -> `get_backend_by_id()`, bei unbekannter ID -> ValueError
    2. Keine ID -> zuerst lokales LLM, dann Cloud
    3. Provider nicht Cloud -> Return
    4. trusted=True -> Return (keine Restriction-Pruefung)
    5. Explizit gewaehltes Cloud-Backend + blockiert -> ValueError
    6. Auto-gewaehltes Cloud-Backend + blockiert -> Fallback auf lokales LLM
    """
    explicit = llm_backend_id is not None
    backend = None

    if explicit:
        backend = get_backend_by_id(llm_backend_id)
        if backend is None:
            raise ValueError(f"Backend mit ID {llm_backend_id} nicht gefunden oder inaktiv")
        if backend.get("category") not in ("llm", "cloud"):
            raise ValueError(
                f"Backend mit ID {llm_backend_id} ist kein LLM-/Cloud-Backend "
                f"(Kategorie: {backend.get('category')})"
            )
    else:
        backend = get_next_backend("llm") or get_next_backend("cloud")
        if backend is None:
            raise RuntimeError("Kein aktives LLM- oder Cloud-Backend verfuegbar")

    provider = backend.get("provider", "ollama")
    if provider not in ("anthropic", "openai"):
        return backend

    # trusted-Check
    if bool(backend.get("trusted", 0)):
        return backend

    if not _is_cloud_blocked(config, filter_datasource_ids):
        return backend

    # Cloud blockiert — bei expliziter Wahl Fehler, bei Auto-Wahl Fallback
    if explicit:
        raise ValueError(
            f"Cloud-Backend '{backend.get('name', '')}' ist durch Restriction-Level-Regeln "
            f"gesperrt. Bitte ein lokales Backend waehlen oder die Datenquellen-Filter anpassen."
        )

    logger.warning(
        "Auto-Cloud-Backend gesperrt (Restriction-Level), Fallback auf lokales LLM-Backend"
    )
    local_backend = get_next_backend("llm")
    if local_backend is not None:
        return local_backend

    raise RuntimeError(
        "Cloud-Backend durch Restriction-Level gesperrt und kein lokales LLM-Backend verfuegbar"
    )


def _wrap_cloud_backend(
    config: dict, backend_obj: LLMBackend, restriction_levels: list[str]
) -> LLMBackend:
    """Wrappt ein Cloud-Backend mit Anonymisierung falls noetig."""
    if needs_anonymization(restriction_levels):
        return wrap_with_anonymization(backend_obj, config, restriction_levels=restriction_levels)
    return backend_obj


def _create_chain_backend(
    config: dict, question: str, llm_backend_id: int | None, filter_datasource_ids: list[int] | None
) -> LLMBackend:
    """Erstellt das LLMBackend fuer Standard-/Rewriter-RAG.

    Ablauf:
    1. Optionales explizites Backend via get_backend_by_id()
    2. Explizites Cloud-Backend: trusted-Check + restriction-Level-Logik
    3. Explizites lokales Backend: User-Wahl respektieren, kein Auto-Routing
    4. Sonst Auto-Routing via _route_backend()
    5. Bei Auto-Cloud: restriction-basiertes Wrapping
    """
    explicit_backend = None
    if llm_backend_id is not None:
        explicit_backend = get_backend_by_id(llm_backend_id)
        if explicit_backend is None:
            raise ValueError(f"Backend mit ID {llm_backend_id} nicht gefunden oder inaktiv")

    # Explizit gewaehltes Cloud-Backend
    if explicit_backend and explicit_backend.get("provider") in ("anthropic", "openai"):
        trusted = bool(explicit_backend.get("trusted", 0))
        if trusted:
            return create_backend(config, llm_backend_id)

        restriction_levels = _load_cloud_restriction_levels(
            filter_datasource_ids,
            explicit=True,
        )

        if is_cloud_blocked_by_restrictions(restriction_levels):
            raise ValueError(
                f"Cloud-Backend '{explicit_backend.get('name', '')}' ist durch Restriction-Level-Regeln "
                f"gesperrt. Bitte ein lokales Backend waehlen oder die Datenquellen-Filter anpassen."
            )
        return _wrap_cloud_backend(
            config, create_backend(config, llm_backend_id), restriction_levels
        )

    # Explizit gewaehltes nicht-Cloud-Backend: User-/Policy-Entscheidung respektieren,
    # niemals durch Auto-Routing auf Cloud umlenken.
    if explicit_backend is not None:
        return create_backend(config, llm_backend_id)

    # Auto-Routing (nur ohne explizite Backend-ID)
    route = _route_backend(config, question, filter_datasource_ids)
    if route == RoutingMode.CLOUD:
        cloud_backend = get_next_backend("cloud")
        if not cloud_backend:
            logger.warning("Cloud-Route gewaehlt, aber kein aktives Cloud-Backend verfuegbar")
            return create_backend(config)

        if bool(cloud_backend.get("trusted", 0)):
            return create_backend(config, cloud_backend["id"])

        try:
            restriction_levels = _load_cloud_restriction_levels(
                filter_datasource_ids,
                explicit=False,
            )
        except RuntimeError as exc:
            logger.warning("Restriction-Level-Abfrage fehlgeschlagen — Fallback auf lokal: %s", exc)
            return create_backend(config, None)

        if is_cloud_blocked_by_restrictions(restriction_levels):
            logger.info("Cloud-Route blockiert durch Restriction-Level — Fallback auf lokal")
            return create_backend(config, None)

        return _wrap_cloud_backend(
            config, create_backend(config, cloud_backend["id"]), restriction_levels
        )

    return create_backend(config)
