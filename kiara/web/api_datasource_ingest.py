"""KIara Admin - Datasource Ingestion.

Queue-basierte Ingestion via IngestionWorker.
Steuerung (Pause/Cancel) und Status-Abfrage.
"""

import copy
import logging
import os
from typing import TYPE_CHECKING, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from kiara.config import load_config
from kiara.repositories.datasource import DatasourceRepository
from kiara.repositories.job import IngestionJobRepository
from kiara.shared.db import db_query
from kiara.shared.enums import IngestionStatus
from kiara.web.api_utils import ERR_DATASOURCE_NOT_FOUND, count_files_limited
from kiara.web.auth import get_admin_user

if TYPE_CHECKING:
    from kiara.connectors.registry import ensure_loaded
    from kiara.connectors.registry import get as get_connector
    from kiara.web.api_datasource import (
        _decrypt_config,
        _effective_path,
        _get_connector_config,
    )

logger = logging.getLogger(__name__)

router = APIRouter(tags=["datasource-ingest"])


class IngestRequest(BaseModel):
    """Request-Body fuer manuelle Ingestion-Ausloesung."""

    embedding_model: str | None = None


class HtmlReingestRequest(IngestRequest):
    """Bestehende HTML-Inhalte gezielt mit der aktuellen Aufbereitung indexieren."""

    mode: Literal["outdated", "all"] = "outdated"


ds_repo = DatasourceRepository()
job_repo = IngestionJobRepository()


def _enqueue_validated_job(
    ds_id: int,
    data: IngestRequest | None,
    job_type: str,
    message: str,
    user: dict,
    *,
    html_reingest: str | None = None,
) -> dict:
    """Validiert die Datenquelle und stellt einen Ingestion-Job in die Queue.

    Gemeinsame Logik fuer ingest_datasource() und reingest_datasource().

    Args:
        ds_id: Datenquellen-ID.
        data: Optionaler Request mit Embedding-Modell.
        job_type: 'incremental' oder 'full'.
        message: Erfolgs-Meldung fuer die API-Response.
        user: Authentifizierter Admin-User.

    Returns:
        API-Response-Dict mit message, status und job_id.

    Raises:
        HTTPException(404): Datenquelle nicht gefunden.
        HTTPException(400): Datenquelle deaktiviert.
        HTTPException(409): Job bereits in Queue.
    """
    from kiara.web.ingestion_worker import enqueue_job

    row = ds_repo.get_by_id(ds_id)
    if not row:
        raise HTTPException(status_code=404, detail=ERR_DATASOURCE_NOT_FOUND)
    if not row["is_active"]:
        raise HTTPException(status_code=400, detail="Datenquelle ist deaktiviert")

    em = data.embedding_model if data else ""
    if not em:
        from kiara.config import get_embedding_model

        em = get_embedding_model(load_config())
    try:
        html_options = {"html_reingest": html_reingest} if html_reingest is not None else {}
        job_id = enqueue_job(
            ds_id,
            job_type=job_type,
            triggered_by="manual",
            embedding_model=em,
            **html_options,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    logger.info(
        "%s enqueued for DS %s (model=%s) by %s → Job #%s",
        job_type.title(),
        ds_id,
        em,
        user["username"],
        job_id,
    )
    return {"message": message, "status": IngestionStatus.PENDING, "job_id": job_id}


@router.post("/api/admin/datasources/{ds_id}/ingest")
async def ingest_datasource(
    ds_id: int, data: IngestRequest | None = None, user=Depends(get_admin_user)
):
    """Start incremental ingestion (stellt Job in Queue)."""
    return _enqueue_validated_job(ds_id, data, "incremental", "Ingestion eingestellt", user)


@router.post("/api/admin/datasources/{ds_id}/reingest")
async def reingest_datasource(
    ds_id: int, data: IngestRequest | None = None, user=Depends(get_admin_user)
):
    """Start full re-ingestion (stellt Job in Queue)."""
    return _enqueue_validated_job(ds_id, data, "full", "Voll-Ingestion eingestellt", user)


@router.post("/api/admin/datasources/{ds_id}/reingest-html")
async def reingest_datasource_html(
    ds_id: int, data: HtmlReingestRequest | None = None, user=Depends(get_admin_user)
):
    """HTML-Auswahl und Neuaufbereitung als normalen Ingestion-Job einstellen."""
    request = data or HtmlReingestRequest()
    return _enqueue_validated_job(
        ds_id,
        request,
        "incremental",
        "HTML-Neuaufbereitung eingestellt",
        user,
        html_reingest=request.mode,
    )


@router.post("/api/admin/datasources/{ds_id}/cancel-ingestion")
async def cancel_ingestion(ds_id: int, user=Depends(get_admin_user)):
    """Cancel: laufenden oder pending Job fuer diese DS abbrechen."""
    from kiara.web.ingestion_worker import get_running_job_id, request_cancel

    # Laufenden Job fuer diese DS finden
    running_id = get_running_job_id()
    if running_id:
        job_row = job_repo.get_by_id(running_id)
        if job_row and job_row["datasource_id"] == ds_id:
            if not request_cancel(running_id):
                raise HTTPException(
                    status_code=409, detail="Job-Status hat sich geändert, Abbruch nicht möglich"
                )
            logger.info(
                "Ingestion cancel for DS %s (Job #%s) by %s", ds_id, running_id, user["username"]
            )
            return {"message": "Abbruch angefordert"}

    # Pending Job fuer diese DS finden
    with db_query() as db:
        pending = db.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s AND status = %s",
            (ds_id, IngestionStatus.PENDING.value),
        ).fetchone()
    if pending:
        if not request_cancel(pending["id"]):
            raise HTTPException(
                status_code=409, detail="Job-Status hat sich geändert, Abbruch nicht möglich"
            )
        logger.info(
            "Pending Job #%s fuer DS %s gecancelt by %s", pending["id"], ds_id, user["username"]
        )
        return {"message": "Ausstehender Job abgebrochen"}

    raise HTTPException(status_code=409, detail="Keine laufende oder ausstehende Ingestion")


@router.post("/api/admin/datasources/{ds_id}/pause-ingestion")
async def pause_ingestion(ds_id: int, user=Depends(get_admin_user)):
    """Toggle pause/resume fuer laufende Ingestion."""
    from kiara.web.ingestion_worker import get_running_job_id, request_pause, request_resume

    current_id = get_running_job_id()
    if not current_id:
        raise HTTPException(status_code=409, detail="Keine laufende Ingestion")

    # Pruefen ob der laufende Job zu dieser DS gehoert
    job_row = job_repo.get_by_id(current_id)
    if not job_row or job_row["datasource_id"] != ds_id:
        raise HTTPException(
            status_code=409, detail="Keine laufende Ingestion fuer diese Datenquelle"
        )

    # Atomarer Toggle: resume ist spezifischer (prueft status=running UND requested_action='pause'),
    # daher zuerst versuchen. Beide Funktionen nutzen WHERE-Klauseln als atomaren Guard.
    if request_resume(current_id):
        return {"message": "Ingestion fortgesetzt", "paused": False}
    if request_pause(current_id):
        return {"message": "Ingestion pausiert", "paused": True}

    raise HTTPException(status_code=409, detail="Ingestion laeuft nicht mehr")


# --- Migriert zu Admin-Registry: datasource.ingestion_status (Sprint 04) ---
# @router.get("/api/admin/datasources/{ds_id}/ingestion-status")
# async def get_ingestion_status(ds_id: int, user=Depends(get_admin_user)):
#     """Aktueller Ingestion-Status (fuer Polling) — liest aus Cache-Tabelle."""
#     with db_query() as db:
#         row = db.execute(
#             "SELECT payload_json FROM cache_datasources WHERE datasource_id = %s",
#             (ds_id,),
#         ).fetchone()
#     if not row:
#         from kiara.web.cache_refresh import refresh_datasource_cache
#         refresh_datasource_cache(ds_id)
#         with db_query() as db:
#             row = db.execute(
#                 "SELECT payload_json FROM cache_datasources WHERE datasource_id = %s",
#                 (ds_id,),
#             ).fetchone()
#     if not row:
#         raise HTTPException(status_code=404, detail=ERR_DATASOURCE_NOT_FOUND)
#     data = json.loads(row["payload_json"])
#     return {
#         "ingestion_status": data.get("ingestion_status", "idle"),
#         "last_ingested_at": data.get("last_ingested_at"),
#         "last_error": data.get("last_error", ""),
#         "chunk_count": data.get("chunk_count", 0),
#         "file_count": data.get("file_count", 0),
#         "processed_files": data.get("processed_files", 0),
#         "progress_total": data.get("progress_total", 0),
#         "progress_processed": data.get("progress_processed", 0),
#         "started_at": data.get("started_at"),
#         "phase": data.get("phase"),
#         "job_id": data.get("job_id"),
#     }


# Keep old /index endpoint as alias for backwards compatibility
@router.post("/api/admin/datasources/{ds_id}/index")
async def index_datasource(ds_id: int, user=Depends(get_admin_user)):
    """Legacy: Trigger indexing (redirects to async ingest)."""
    return await ingest_datasource(ds_id, user=user)


# ---------------------------------------------------------------------------
# Legacy compatibility: keep old status endpoint working
# ---------------------------------------------------------------------------


@router.get("/api/datasource/status")
async def get_datasource_status_legacy(user=Depends(get_admin_user)):
    """Legacy: Get first datasource status."""
    config = load_config()
    path = config["datasource"]["path"]

    with db_query() as db:
        row = db.execute(
            "SELECT * FROM datasources WHERE is_active = true ORDER BY id LIMIT 1"
        ).fetchone()
    if row:
        _load_lazy_attrs()
        path = _effective_path(row)

    accessible = os.path.isdir(path)
    file_count = count_files_limited(path) if accessible else 0

    return {"path": path, "accessible": accessible, "file_count": file_count}


# ---------------------------------------------------------------------------
# _prepare_ds_config — wiederhergestellt fuer Rueckwaertskompatibilitaet
# (wird von api_datasource.py re-exportiert + Tests)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Lazy-Modul-Attribute fuer _prepare_ds_config (vermeidet Zirkular-Import).
# @patch("kiara.web.api_datasource_ingest.get_connector") etc. loest
# __getattr__ aus, das die echten Referenzen einmalig laedt.
# ---------------------------------------------------------------------------

_LAZY_ATTRS = {
    "get_connector": ("kiara.connectors.registry", "get"),
    "ensure_loaded": ("kiara.connectors.registry", "ensure_loaded"),
    "_get_connector_config": ("kiara.web.api_datasource", "_get_connector_config"),
    "_decrypt_config": ("kiara.web.api_datasource", "_decrypt_config"),
    "_effective_path": ("kiara.web.api_datasource", "_effective_path"),
}


def _load_lazy_attrs() -> None:
    # Beim ersten Zugriff auf ein Lazy-Attribut alle auf einmal laden,
    # damit auch Funktionen innerhalb des Moduls (bare name lookup)
    # die Referenzen in globals() finden.
    import importlib

    for attr_name_key, (module_path, real_attr) in _LAZY_ATTRS.items():
        if attr_name_key not in globals():
            mod = importlib.import_module(module_path)
            globals()[attr_name_key] = getattr(mod, real_attr)


def __getattr__(name):
    if name in _LAZY_ATTRS:
        _load_lazy_attrs()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _prepare_ds_config(row):
    """Prepare config for single-datasource ingestion via Registry.

    Returns:
        (config_dict, None) bei Erfolg
        (None, error_message) bei Fehler
    """
    _load_lazy_attrs()
    ensure_loaded()
    config = copy.deepcopy(load_config())

    connector_cls = get_connector(row["type"])
    if not connector_cls:
        return None, f"Unbekannter Datenquellen-Typ: {row['type']}"

    conn_config = _get_connector_config(row)
    conn_config = _decrypt_config(conn_config, connector_cls)

    try:
        path = connector_cls.prepare_and_scan(conn_config, dict(row), config)
    except Exception as exc:
        return None, f"Vorbereitung fehlgeschlagen: {exc}"

    if path is None:
        return None, f"Vorbereitung fehlgeschlagen fuer Typ {row['type']}"

    config["datasource"]["path"] = path
    return config, None
