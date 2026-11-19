"""KIara Admin - Datasource Feature Checks.

DDU (Deep Document Understanding), VLM (Vision Language Model)
Verfuegbarkeitspruefung und Embedding-Modell-Auflistung.
"""

import asyncio
import logging
import threading
import time

logger = logging.getLogger(__name__)

import httpx
from fastapi import APIRouter, Depends, Query

from kiara.config import DEFAULT_TIMEOUTS, load_config
from kiara.web.auth import get_admin_user

router = APIRouter(tags=["datasource-features"])

# ---------------------------------------------------------------------------
# DDU-Verfuegbarkeit
# ---------------------------------------------------------------------------
# Hausaenderung (30.10.2026): Keine Netzwerkprobe beim Oeffnen der Maske und
# kein IP-Pinning/Allowlist (``ddu.internal_cidrs``). Grund: Ein ``/health`` am
# Kiron-Proxy startet Docling und sperrt die GPU; das Pinning wurde nur hier
# verwendet, der Ingest (``check_ddu_health``) prueft ohne. Ob ein Backend
# erreichbar ist, entscheidet der Ingest-Preflight bzw. der explizite
# Test-Button der Backend-Verwaltung.


def invalidate_ddu_cache() -> None:
    """Kompatibilitaets-Stub: Es gibt keinen DDU-Cache mehr (keine Netzwerkprobe)."""


@router.get("/api/admin/ddu-available")
async def ddu_available(
    user: dict = Depends(get_admin_user),
    url: str | None = Query(None, description="Optionale URL zum expliziten Health-Test"),
) -> dict:
    """DDU-Status fuer die UI.

    Ohne ``url``: reine Konfigurationspruefung (mindestens ein aktives Remote-Backend
    mit URL), **keine** Netzwerkanfrage. Mit ``url``: expliziter Health-Test dieser
    URL (``GET <url>/health``) ueber denselben Check wie der Ingest-Preflight; das
    kann ein ruhendes Docling starten.
    """
    if url:
        probe = url.strip()
        if not probe.lower().startswith(("http://", "https://")):
            return {
                "available": False,
                "message": "DDU-URL muss mit http:// oder https:// beginnen",
            }
        from kiara.ingestion.ddu_client import check_ddu_health

        ok, err = await asyncio.to_thread(check_ddu_health, [{"url": probe}])
        if ok:
            return {"available": True, "mode": "remote", "url": probe}
        return {"available": False, "message": f"DDU unter dieser URL nicht erreichbar: {err}"}

    from kiara.config import get_ddu_backends

    backends = [b for b in get_ddu_backends(load_config()) if b.get("url") and not b.get("is_local")]
    if not backends:
        return {
            "available": False,
            "message": "Kein aktives DDU-Backend mit URL konfiguriert (KI-Modelle, Kategorie DDU)",
        }
    return {"available": True, "mode": "remote", "url": backends[0]["url"]}


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


@router.get("/api/admin/embedding-contracts")
async def list_embedding_contracts(user: dict = Depends(get_admin_user)) -> dict:
    """Read-only KIron discovery plus persisted per-index S4 diagnostics."""

    from kiara.backend_selector import get_all_active
    from kiara.retrieval.embedding_discovery import (
        discover_configured_embedding_backends,
    )

    backends = get_all_active("embedding")
    inventory = await asyncio.to_thread(
        discover_configured_embedding_backends,
        backends,
        timeout=DEFAULT_TIMEOUTS["http_api"],
    )
    result = inventory.to_dict()
    try:
        result["indexes"] = await asyncio.to_thread(
            _persisted_index_contract_diagnostics,
            load_config(),
        )
        result["index_diagnostics_error"] = None
    except Exception as exc:
        logger.warning(
            "Persistente Index-Contract-Diagnose fehlgeschlagen: %s",
            exc,
            exc_info=True,
        )
        result["indexes"] = []
        result["index_diagnostics_error"] = exc.__class__.__name__
    return result


def _persisted_index_contract_diagnostics(config: dict) -> list[dict]:
    """Return detached diagnostics; never mutate repository/discovery rows."""
    from kiara.repositories.index_contract import IndexContractRepository
    from kiara.retrieval.index_compatibility import (
        IndexCompatibilityError,
        IndexVectorKind,
        _validate_stored_contract,
        make_index_target,
        resolve_current_index_contract,
    )
    from kiara.shared.db import db_query

    stored_rows = IndexContractRepository().list_all()
    diagnostics: list[dict] = []
    with db_query() as conn:
        for original in stored_rows:
            row = dict(original)
            target = make_index_target(
                config,
                collection_name=str(row["collection_name"]),
                vector_kind=IndexVectorKind(str(row["vector_kind"])),
                requested_model=str(row["requested_model"]),
            )
            current = resolve_current_index_contract(conn, config, target)
            persisted_state = str(row.get("verification_state") or "unverified")
            reason_code = str(row.get("reason_code") or "")
            snapshot_drift = None
            if target.logical_index_key != row.get("logical_index_key"):
                state = "mismatch"
                reason_code = "index_logical_key_mismatch"
            elif persisted_state == "verified":
                try:
                    snapshot_drift = _validate_stored_contract(
                        "diagnostic",
                        target,
                        row,
                        current,
                    )
                    state = "compatible"
                    reason_code = "index_contract_compatible"
                except IndexCompatibilityError as exc:
                    state = "mismatch"
                    reason_code = exc.reason_code
            else:
                state = persisted_state

            diagnostics.append(
                {
                    **target.to_dict(),
                    "compatibility_state": state,
                    "reason_code": reason_code,
                    "stored": {
                        "canonical_model_id": row.get("canonical_model_id"),
                        "profile_id": row.get("profile_id"),
                        "profile_kind": row.get("profile_kind"),
                        "profile_endpoint": row.get("profile_endpoint"),
                        "index_compatibility_id": row.get("index_compatibility_id"),
                        "query_compatibility_id": row.get("query_compatibility_id"),
                        "contract_snapshot_sha256": row.get("contract_snapshot_sha256"),
                        "fence_version": row.get("fence_version"),
                    },
                    "current": current.diagnostic_dict(),
                    "snapshot_drift": (
                        snapshot_drift.to_dict() if snapshot_drift is not None else None
                    ),
                    "evidence": {
                        "type": row.get("evidence_type"),
                        "created_at": str(
                            row.get("evidence_created_at") or row.get("evidence_at") or ""
                        ),
                        "sample_count": row.get("evidence_sample_count"),
                        "max_abs_error": row.get("max_abs_error"),
                        "max_cosine_distance": row.get("max_cosine_distance"),
                        "abs_tolerance": row.get("abs_tolerance"),
                        "cosine_tolerance": row.get("cosine_tolerance"),
                        "collection_config_sha256": row.get("collection_config_sha256"),
                        "sample_set_sha256": row.get("sample_set_sha256"),
                    },
                    "blocked_operations": (
                        []
                        if state == "compatible"
                        else [
                            "read",
                            "write",
                            "re_ingestion",
                            "model_switch",
                            "cache_read",
                            "collection_admin",
                        ]
                    ),
                    "operator_action": (
                        "none"
                        if state == "compatible"
                        else "expliziten Index-Contract-Audit/Backfill nach Evidenzpruefung ausfuehren"
                    ),
                }
            )
    return diagnostics


@router.get("/api/admin/embedding-models")
async def list_embedding_models(user: dict = Depends(get_admin_user)) -> dict:
    """List embedding models from configured backends (not all server models)."""
    global _embed_cache_result, _embed_cache_ts

    now = time.time()
    with _embed_lock:
        if _embed_cache_result is not None and (now - _embed_cache_ts) < _EMBED_CACHE_TTL:
            return _embed_cache_result

    from kiara.backend_selector import get_all_active, normalize_model_name
    from kiara.retrieval.embedding_contract import ProfileKind
    from kiara.retrieval.embedding_roles import (
        SEARCH_DOCUMENT,
        backend_supports_embedding_operation,
    )
    from kiara.retrieval.vectorstore import VectorStore

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
            supports_late, late_reason = backend_supports_embedding_operation(
                b,
                role=SEARCH_DOCUMENT,
                kind=ProfileKind.LATE_CHUNKING,
                endpoint="/api/embed_late",
            )
            models.append(
                {
                    "name": m,
                    "has_collection": m in collections,
                    "chunk_count": collections.get(m, 0),
                    "is_default": m == default_model,
                    "supports_late_chunking": supports_late,
                    "late_chunking_contract_reason": late_reason,
                }
            )

        result = {"models": sorted(models, key=lambda x: x["name"]), "default_model": default_model}

    with _embed_lock:
        _embed_cache_result = result
        _embed_cache_ts = now

    return result
