"""KIara Admin - Datasource Feature Checks.

DDU (Deep Document Understanding), VLM (Vision Language Model)
Verfuegbarkeitspruefung und Embedding-Modell-Auflistung.
"""

import logging
import threading
import time
from typing import Sequence

logger = logging.getLogger(__name__)

import httpx
from fastapi import APIRouter, Depends, Query

from kiara.config import DEFAULT_TIMEOUTS, load_config
from kiara.shared.http_pinning import (
    PinningConfigError,
    PinningError,
    pinned_request_async,
    validate_allowed_private_cidrs,
)
from kiara.web.auth import get_admin_user

router = APIRouter(tags=["datasource-features"])

# ---------------------------------------------------------------------------
# DDU-Availability Cache (30s TTL — konsistent mit backend_selector._CACHE_TTL)
# ---------------------------------------------------------------------------
_DDU_CACHE_TTL = 30  # Sekunden
_ddu_lock = threading.Lock()
_ddu_cache_result: dict | None = None
_ddu_cache_ts: float = 0.0


def invalidate_ddu_cache() -> None:
    """Invalidiert den DDU-Availability-Cache (nach Backend-CRUD aufrufen)."""
    global _ddu_cache_result, _ddu_cache_ts
    with _ddu_lock:
        _ddu_cache_result = None
        _ddu_cache_ts = 0.0


async def _check_ddu_remote(
    ddu_backend: dict,
    *,
    allowed_private_cidrs: Sequence[str] | None = None,
) -> dict | None:
    """Prueft, ob docling-serve unter ``ddu_backend['url']`` erreichbar ist.

    Nutzt den zentralen Pinning-Helper, damit DNS-Rebinding/SSRF-TOCTOU
    konsistent mit dem Webcrawler-Pfad blockiert werden. ``PinningConfigError``
    bleibt fuer den Aufrufer (Backend-Iteration) sichtbar; ``PinningError`` und
    sonstige ``httpx.RequestError`` enden als DEBUG-Log + ``None``.
    """
    url = f"{ddu_backend['url'].rstrip('/')}/health"
    try:
        resp = await pinned_request_async(
            "GET",
            url,
            allowed_private_cidrs=allowed_private_cidrs,
            timeout=DEFAULT_TIMEOUTS["http_health"],
        )
        if resp.status_code == 200:
            return {
                "available": True,
                "mode": "remote",
                "url": ddu_backend["url"],
            }
    except PinningConfigError:
        raise
    except (PinningError, httpx.RequestError) as exc:
        logger.debug("DDU-Remote-Check fehlgeschlagen: %s", exc)
    return None


@router.get("/api/admin/ddu-available")
async def ddu_available(
    user: dict = Depends(get_admin_user),
    url: str | None = Query(None, description="Optionale URL zum direkten Testen"),
) -> dict:
    """Prueft ob DDU verfuegbar ist (mindestens ein aktives Backend erreichbar).

    Iteriert ueber alle konfigurierten DDU-Backends (sortiert nach Prioritaet).
    Ergebnis wird 30s gecached. Bei uebergebenem ``url``-Parameter wird
    stattdessen direkt diese URL geprueft (kein Cache, kein Allowlist-Bonus).
    """
    if url:
        # Probe-Pfad: keine Allowlist - nur globale Ziele duerfen geprobt werden.
        result = await _check_ddu_remote({"url": url.strip()}, allowed_private_cidrs=None)
        return result or {
            "available": False,
            "message": "DDU unter dieser URL nicht erreichbar",
        }

    global _ddu_cache_result, _ddu_cache_ts

    now = time.time()
    with _ddu_lock:
        if _ddu_cache_result is not None and (now - _ddu_cache_ts) < _DDU_CACHE_TTL:
            return _ddu_cache_result

    from kiara.config import get_ddu_backends, get_ddu_internal_cidrs

    config = load_config()
    backends = get_ddu_backends(config)

    if not backends:
        result = {
            "available": False,
            "message": "Kein aktives DDU-Backend konfiguriert",
        }
        with _ddu_lock:
            _ddu_cache_result = result
            _ddu_cache_ts = now
        return result

    cache_skip = False
    try:
        # CIDR-Validierung VOR der Backend-Schleife: ungueltige Config wird
        # fail-closed bevor ein Remote-Healthcheck ausgefuehrt wird.
        cidrs = get_ddu_internal_cidrs(config)
        validate_allowed_private_cidrs(cidrs)

        result = None
        has_failover = len(backends) > 1

        selected_backend_index = None
        for idx, backend in enumerate(backends):
            if backend.get("is_local") or not backend.get("url"):
                logger.info("DDU-Backend ohne URL ignoriert: KIara nutzt Docling remote-only")
                continue
            remote_result = await _check_ddu_remote(backend, allowed_private_cidrs=cidrs)
            if remote_result:
                result = remote_result
                selected_backend_index = idx
                break

        if result is None:
            result = {
                "available": False,
                "message": "Alle konfigurierten DDU-Backends nicht erreichbar",
            }
        elif has_failover and selected_backend_index not in (None, 0):
            result["failover"] = True
    except PinningConfigError as exc:
        logger.warning("ungueltige DDU-Pinning-Konfiguration: %s", exc)
        result = {
            "available": False,
            "message": "ungueltige ddu.internal_cidrs-Konfiguration",
        }
        cache_skip = True

    if not cache_skip:
        with _ddu_lock:
            _ddu_cache_result = result
            _ddu_cache_ts = now

    return result


# ---------------------------------------------------------------------------
# VLM-Availability Cache (30s TTL)
# ---------------------------------------------------------------------------
_VLM_CACHE_TTL = 30
_vlm_lock = threading.Lock()
_vlm_cache_result: dict | None = None
_vlm_cache_ts: float = 0.0


@router.get("/api/admin/vlm-available")
async def vlm_available(user: dict = Depends(get_admin_user)) -> dict:
    """Prueft ob ein VLM-Modell konfiguriert und auf Ollama verfuegbar ist."""
    global _vlm_cache_result, _vlm_cache_ts

    now = time.time()
    with _vlm_lock:
        if _vlm_cache_result is not None and (now - _vlm_cache_ts) < _VLM_CACHE_TTL:
            return _vlm_cache_result

    from kiara.config import get_vlm_backend

    config = load_config()
    vlm_backend = get_vlm_backend(config)
    if not vlm_backend:
        result = {
            "available": False,
            "message": "Kein VLM-Modell konfiguriert (KI-Backends in Admin-UI)",
        }
    else:
        vlm_model = vlm_backend["model"]
        vlm_url = vlm_backend["url"]
        try:
            import ollama as ollama_client

            client = ollama_client.Client(host=vlm_url)
            resp = client.list()
            try:
                raw_models = getattr(resp, "models", None) or resp.get("models", [])
            except (AttributeError, TypeError):
                raw_models = []
            model_names = [getattr(m, "model", None) or m.get("name", "") for m in raw_models]
            available = any(
                name == vlm_model or (":" not in vlm_model and name.split(":")[0] == vlm_model)
                for name in model_names
            )
            if available:
                result = {"available": True, "model": vlm_model}
            else:
                result = {
                    "available": False,
                    "message": f"Modell '{vlm_model}' nicht auf Ollama gefunden. Installation: ollama pull {vlm_model}",
                }
        except Exception as exc:
            logger.warning("VLM-Probe fehlgeschlagen (model=%s): %s", vlm_model, exc, exc_info=True)
            result = {"available": False, "message": "Ollama nicht erreichbar"}

    with _vlm_lock:
        _vlm_cache_result = result
        _vlm_cache_ts = now

    return result


# ---------------------------------------------------------------------------
# Embedding-Models Cache (30s TTL)
# ---------------------------------------------------------------------------
_EMBED_CACHE_TTL = 30
_embed_lock = threading.Lock()
_embed_cache_result: dict | None = None
_embed_cache_ts: float = 0.0


@router.get("/api/admin/embedding-models")
async def list_embedding_models(user: dict = Depends(get_admin_user)) -> dict:
    """List embedding models from configured backends (not all server models)."""
    global _embed_cache_result, _embed_cache_ts

    now = time.time()
    with _embed_lock:
        if _embed_cache_result is not None and (now - _embed_cache_ts) < _EMBED_CACHE_TTL:
            return _embed_cache_result

    from kiara.backend_selector import get_all_active, normalize_model_name
    from kiara.retrieval.vectorstore import VectorStore
    from kiara.shared.constants import LATE_CHUNKING_SUPPORTED_MODELS

    config = load_config()

    backends = get_all_active("embedding")
    if not backends:
        result = {"models": [], "default_model": "", "error": "Kein Embedding-Backend konfiguriert"}
    else:
        collections = {c["model"]: c["chunks"] for c in VectorStore.list_collections(config)}
        default_model = normalize_model_name(backends[0]["model"])

        seen: set[str] = set()
        models = []
        for b in backends:
            m = normalize_model_name(b["model"])
            if m in seen:
                continue
            seen.add(m)
            models.append(
                {
                    "name": m,
                    "has_collection": m in collections,
                    "chunk_count": collections.get(m, 0),
                    "is_default": m == default_model,
                    "supports_late_chunking": m in LATE_CHUNKING_SUPPORTED_MODELS
                    and b.get("provider") == "ollama",
                }
            )

        result = {"models": sorted(models, key=lambda x: x["name"]), "default_model": default_model}

    with _embed_lock:
        _embed_cache_result = result
        _embed_cache_ts = now

    return result
