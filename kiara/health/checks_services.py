"""Service Health-Checks (Ollama, LDAP, Cloud-KI, ERP)."""

import asyncio
import logging
import time

import httpx

from kiara.config import DEFAULT_TIMEOUTS
from kiara.health.models import ComponentHealth
from kiara.repositories.config import ConfigRepository
from kiara.shared.enums import HealthStatus
from kiara.shared.ldap import test_connection as ldap_test_connection
from kiara.shared.ollama_utils import check_ollama_models
from kiara.web.config_utils import build_ldap_config

logger = logging.getLogger(__name__)


def _unhealthy_result(
    name: str, start: float, message: str, details: dict | None = None
) -> ComponentHealth:
    """Erzeugt ein UNHEALTHY-ComponentHealth mit Latenz-Berechnung.

    Zentraler Helper fuer alle Health-Checks, eliminiert die wiederholte
    Latenz-Berechnung und ComponentHealth-Konstruktion bei Fehlern.
    """
    latency = (time.time() - start) * 1000
    return ComponentHealth(
        name=name,
        status=HealthStatus.UNHEALTHY,
        latency_ms=round(latency, 1),
        message=message,
        details=details or {},
    )


async def check_ollama(config: dict) -> ComponentHealth:
    """GET {ollama_url}/api/tags.

    Healthy: erreichbar + konfiguriertes LLM-Modell verfuegbar.
    Degraded: erreichbar aber Modell nicht in der Liste.
    Unhealthy: nicht erreichbar.
    """
    from kiara.config import get_llm_backend

    backend = get_llm_backend(config)
    ollama_url = backend["url"]
    llm_model = backend["model"]
    start = time.time()

    ok, models, _raw, error = await check_ollama_models(
        ollama_url,
        timeout=DEFAULT_TIMEOUTS["http_health"],
    )
    latency = (time.time() - start) * 1000

    if not ok:
        logger.error("Ollama Health-Check Fehler: %s", error)
        return _unhealthy_result(
            "ollama",
            start,
            f"Ollama nicht erreichbar: {str(error)[:200]}",
            {"url": ollama_url},
        )

    # Exact match; if model has no explicit tag, also accept ":latest"
    if llm_model in models or (":" not in llm_model and f"{llm_model}:latest" in models):
        return ComponentHealth(
            name="ollama",
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency, 1),
            message=f"Ollama erreichbar, Modell '{llm_model}' verfuegbar",
            details={"models": models, "url": ollama_url},
        )
    else:
        return ComponentHealth(
            name="ollama",
            status=HealthStatus.DEGRADED,
            latency_ms=round(latency, 1),
            message=f"Ollama erreichbar, aber Modell '{llm_model}' nicht gefunden",
            details={"models": models, "url": ollama_url},
        )


async def check_ldap(config: dict) -> ComponentHealth:
    """LDAP pruefen wenn konfiguriert.

    Unconfigured wenn keine LDAP-Config in der DB vorhanden.
    Healthy wenn Server erreichbar.
    Unhealthy wenn konfiguriert aber nicht erreichbar.
    """
    start = time.time()

    # Check if LDAP is configured via database
    try:
        config_repo = ConfigRepository()
        row = config_repo.get("ldap_config")
    except Exception:
        row = None

    try:
        if not row:
            return ComponentHealth(
                name="ldap",
                status=HealthStatus.UNCONFIGURED,
                message="LDAP nicht konfiguriert",
            )

        config = build_ldap_config(row)
        host = row["server_url"].replace("ldap://", "").replace("ldaps://", "").strip("/")
        port = config["port"]

        # Full LDAP check: bind + search (not just TCP)
        try:
            success, error = await asyncio.to_thread(ldap_test_connection, config)
        except Exception as exc:
            return _unhealthy_result(
                "ldap",
                start,
                f"LDAP-Verbindungstest Fehler: {str(exc)[:100]}",
                {"server": host, "port": port},
            )

        latency = (time.time() - start) * 1000
        if success:
            return ComponentHealth(
                name="ldap",
                status=HealthStatus.HEALTHY,
                latency_ms=round(latency, 1),
                message=f"LDAP-Server erreichbar ({host}:{port})",
                details={"server": host, "port": port},
            )
        return _unhealthy_result(
            "ldap",
            start,
            f"LDAP-Verbindungstest fehlgeschlagen: {(error or 'Unbekannt')[:100]}",
            {"server": host, "port": port},
        )

    except Exception as exc:
        logger.error("LDAP Health-Check Fehler: %s", exc)
        return _unhealthy_result("ldap", start, f"LDAP Health-Check Fehler: {str(exc)[:100]}")


def _probe_erp_sql_sync(dsn: str, driver: str) -> str:
    """Synchroner ERP-SQL-Probe (wird via asyncio.to_thread aufgerufen).

    Liefert den effektiv verwendeten Driver-Namen zurueck (Auto-Detection).
    """
    from kiara.connectors.sql_connector import SQLConnector

    connector = SQLConnector(dsn, driver)
    with connector.connection() as conn:
        conn.cursor().execute("SELECT 1")
    return connector._driver


async def check_erp(config: dict) -> ComponentHealth:
    """ERP pruefen.

    Unconfigured wenn tools.enabled = False.
    Mock-Backend ist immer healthy.
    REST/SQL wird auf Erreichbarkeit geprueft.
    """
    tools_cfg = config.get("tools", {})
    enabled = tools_cfg.get("enabled", False)

    if not enabled:
        return ComponentHealth(
            name="erp",
            status=HealthStatus.UNCONFIGURED,
            message="ERP-Tools nicht aktiviert",
        )

    erp_cfg = tools_cfg.get("erp", {})
    backend = erp_cfg.get("backend", "mock")

    if backend == "mock":
        return ComponentHealth(
            name="erp",
            status=HealthStatus.HEALTHY,
            message="ERP-Backend: Mock (immer verfuegbar)",
            details={"backend": "mock"},
        )

    if backend == "rest":
        rest_url = erp_cfg.get("rest_url", "")
        if not rest_url:
            return ComponentHealth(
                name="erp",
                status=HealthStatus.DEGRADED,
                message="ERP REST-Backend konfiguriert, aber keine URL gesetzt",
                details={"backend": "rest"},
            )

        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUTS["http_health"]) as client:
                resp = await client.get(rest_url)
                resp.raise_for_status()
                latency = (time.time() - start) * 1000
            return ComponentHealth(
                name="erp",
                status=HealthStatus.HEALTHY,
                latency_ms=round(latency, 1),
                message=f"ERP REST-Backend erreichbar (HTTP {resp.status_code})",
                details={"backend": "rest", "url": rest_url},
            )
        except Exception as exc:
            return _unhealthy_result(
                "erp",
                start,
                f"ERP REST-Backend nicht erreichbar: {exc}",
                {"backend": "rest", "url": rest_url},
            )

    if backend == "sql":
        sql_dsn = erp_cfg.get("sql_dsn", "")
        if not sql_dsn:
            return ComponentHealth(
                name="erp",
                status=HealthStatus.DEGRADED,
                message="ERP SQL-Backend konfiguriert, aber kein DSN gesetzt",
                details={"backend": "sql"},
            )
        start = time.time()
        try:
            from kiara.shared.crypto import decrypt, is_encrypted

            dsn = decrypt(sql_dsn) if is_encrypted(sql_dsn) else sql_dsn
            sql_driver = erp_cfg.get("sql_driver", "auto")
            driver_used = await asyncio.to_thread(_probe_erp_sql_sync, dsn, sql_driver)
            latency = (time.time() - start) * 1000
            return ComponentHealth(
                name="erp",
                status=HealthStatus.HEALTHY,
                latency_ms=round(latency, 1),
                message="ERP SQL-Backend erreichbar",
                details={"backend": "sql", "driver": driver_used, "dsn_set": True},
            )
        except Exception as exc:
            return _unhealthy_result(
                "erp", start, f"ERP SQL-Backend nicht erreichbar: {exc}", {"backend": "sql"}
            )

    return ComponentHealth(
        name="erp",
        status=HealthStatus.DEGRADED,
        message=f"Unbekannter ERP-Backend-Typ: {backend}",
        details={"backend": backend},
    )


async def check_cloud_ki(config: dict) -> ComponentHealth:
    """Cloud-KI-Backend (Anthropic/OpenAI) pruefen.

    Unconfigured wenn cloud.enabled = False oder kein API-Key gesetzt.
    Healthy wenn API-Endpoint erreichbar (leichtgewichtiger GET/HEAD).
    Unhealthy wenn nicht erreichbar.
    """
    start = time.time()

    from kiara.config import get_cloud_backend

    backend = get_cloud_backend(config)

    if not backend or not backend.get("api_key"):
        return ComponentHealth(
            name="cloud_ki",
            status=HealthStatus.UNCONFIGURED,
            message="Cloud-KI nicht konfiguriert oder kein API-Key gesetzt",
        )

    provider = backend.get("provider", "openai")
    api_key = backend["api_key"]
    base_url = backend.get("base_url", "")

    # Provider-spezifische Probe-URLs
    if provider == "anthropic":
        probe_url = (base_url or "https://api.anthropic.com").rstrip("/") + "/v1/models"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
    else:
        normalized = (base_url or "https://api.openai.com").rstrip("/")
        if normalized.endswith("/v1"):
            normalized = normalized[:-3]
        probe_url = normalized + "/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUTS["http_health"]) as client:
            resp = await client.get(probe_url, headers=headers)
            resp.raise_for_status()
            latency = (time.time() - start) * 1000
        return ComponentHealth(
            name="cloud_ki",
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency, 1),
            message=f"Cloud-KI erreichbar ({provider}, HTTP {resp.status_code})",
            details={"provider": provider, "model": backend.get("model", "")},
        )
    except httpx.HTTPStatusError as exc:
        latency = (time.time() - start) * 1000
        # 401/403 = API-Key ungueltig, aber Endpoint erreichbar
        if exc.response.status_code in (401, 403):
            return ComponentHealth(
                name="cloud_ki",
                status=HealthStatus.DEGRADED,
                latency_ms=round(latency, 1),
                message=f"Cloud-KI erreichbar, aber API-Key ungueltig ({provider}, HTTP {exc.response.status_code})",
                details={"provider": provider},
            )
        return _unhealthy_result(
            "cloud_ki",
            start,
            f"Cloud-KI HTTP-Fehler: {provider} {exc.response.status_code}",
            {"provider": provider},
        )
    except Exception as exc:
        return _unhealthy_result(
            "cloud_ki",
            start,
            f"Cloud-KI nicht erreichbar: {str(exc)[:200]}",
            {"provider": provider},
        )


async def check_colbert(config: dict) -> ComponentHealth:
    """POST /api/embed_colbert Health-Probe.

    Unconfigured: retrieval.colbert.enabled ist False.
    Healthy: Endpoint antwortet innerhalb 2s und liefert ein gueltiges Payload.
    Degraded: Endpoint antwortet langsam (>2s) ODER URL/HTTP/Payload schlagen fehl.
    """
    start = time.time()
    colbert_cfg = config.get("retrieval", {}).get("colbert", {})
    if not colbert_cfg.get("enabled"):
        return ComponentHealth(
            name="colbert",
            status=HealthStatus.UNCONFIGURED,
            latency_ms=0.0,
            message="ColBERT nicht aktiviert",
        )

    def _degraded(message: str, details: dict | None = None) -> ComponentHealth:
        latency = (time.time() - start) * 1000
        return ComponentHealth(
            name="colbert",
            status=HealthStatus.DEGRADED,
            latency_ms=round(latency, 1),
            message=message,
            details=details or {},
        )

    # URL-Aufloesung via bestehende Logik
    try:
        from kiara.retrieval.colbert import _resolve_colbert_url

        url = _resolve_colbert_url(config)
    except Exception as exc:
        return _degraded(f"URL-Aufloesung fehlgeschlagen: {exc}")

    from kiara.backend_selector import normalize_model_name
    from kiara.shared.constants import COLBERT_DEFAULT_MODEL

    model = normalize_model_name(colbert_cfg.get("model") or COLBERT_DEFAULT_MODEL)

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUTS["http_health"]) as client:
            resp = await client.post(
                f"{url}/api/embed_colbert",
                json={"model": model, "input": ["health"]},
            )
            resp.raise_for_status()
            payload = resp.json()
            embeddings = payload.get("embeddings")
            if (
                not isinstance(embeddings, list)
                or len(embeddings) != 1
                or not isinstance(embeddings[0], list)
            ):
                raise TypeError(
                    "embeddings hat nicht die erwartete Form "
                    "[token_vectors] fuer genau einen Probe-Input"
                )
    except Exception as exc:
        return _degraded(
            f"ColBERT-Backend nicht erreichbar oder Payload ungueltig: {exc}",
            {"model": model, "url": url},
        )

    latency = (time.time() - start) * 1000
    status = HealthStatus.HEALTHY if latency < 2000 else HealthStatus.DEGRADED
    message = (
        "ColBERT-Backend erreichbar"
        if status == HealthStatus.HEALTHY
        else f"Langsam ({latency:.0f}ms)"
    )

    return ComponentHealth(
        name="colbert",
        status=status,
        latency_ms=round(latency, 1),
        message=message,
        details={"model": model, "url": url},
    )


async def check_reranker(config: dict) -> ComponentHealth:
    """Cross-Encoder Reranker pruefen.

    Unconfigured wenn reranking deaktiviert.
    Healthy wenn aktiviert + sentence-transformers installiert + Modell im Cache.
    Degraded wenn aktiviert aber Dependency fehlt ODER Modell nicht im Cache.

    Das Modell selbst wird NICHT geladen (wuerde beim ersten Aufruf
    einen HuggingFace-Download triggern). Der tatsaechliche Modell-Download
    passiert lazy beim ersten Reranking-Aufruf.
    """
    retrieval_cfg = config.get("retrieval", {})
    enabled = retrieval_cfg.get("reranking", False)

    if not enabled:
        return ComponentHealth(
            name="reranker",
            status=HealthStatus.UNCONFIGURED,
            message="Cross-Encoder Reranking nicht aktiviert",
        )

    model_name = retrieval_cfg.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    top_k = retrieval_cfg.get("rerank_top_k", 5)
    multiplier = retrieval_cfg.get("rerank_fetch_multiplier", 3)
    start = time.time()

    try:
        from sentence_transformers import CrossEncoder  # noqa: F401
    except ImportError:
        return ComponentHealth(
            name="reranker",
            status=HealthStatus.DEGRADED,
            latency_ms=round((time.time() - start) * 1000, 1),
            message="Reranking aktiviert, aber sentence-transformers nicht installiert",
            details={"model": model_name, "package_installed": False},
        )

    # Pruefen ob Modell bereits im HuggingFace-Cache liegt
    model_cached = False
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_name, "config.json")
        model_cached = isinstance(cached, str)
    except Exception as exc:
        logger.debug("Reranker Cache-Check fehlgeschlagen: %s", exc)

    latency = (time.time() - start) * 1000
    if model_cached:
        status = HealthStatus.HEALTHY
        message = f"Cross-Encoder '{model_name}' bereit (top_k={top_k}, fetch×{multiplier})"
    else:
        status = HealthStatus.DEGRADED
        message = (
            f"Cross-Encoder '{model_name}' nicht im Cache "
            f"— Modellverfuegbarkeit nicht validiert (top_k={top_k}, fetch×{multiplier})"
        )

    return ComponentHealth(
        name="reranker",
        status=status,
        latency_ms=round(latency, 1),
        message=message,
        details={
            "model": model_name,
            "package_installed": True,
            "model_cached": model_cached,
            "top_k": top_k,
            "fetch_multiplier": multiplier,
        },
    )
