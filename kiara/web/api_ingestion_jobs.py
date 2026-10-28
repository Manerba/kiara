"""KIara Admin - Ingestion Jobs & Schedule API.

REST-Endpoints fuer Job-Queue (Liste, Details, Cancel, Pause, History-Cleanup)
und Cron-Schedule-Management.
"""

import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, model_validator

from kiara.repositories.cache import CacheRepository
from kiara.repositories.datasource import DatasourceRepository
from kiara.repositories.job import FileLogRepository, IngestionJobRepository
from kiara.shared.db import db_query, db_transaction
from kiara.shared.enums import IngestionStatus
from kiara.web.api_utils import ERR_DATASOURCE_NOT_FOUND, safe_refresh
from kiara.web.auth import get_admin_user
from kiara.web.config_utils import safe_json_loads

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingestion-jobs"])

# Module-level repo instances
_job_repo = IngestionJobRepository()
_file_log_repo = FileLogRepository()
_cache_repo = CacheRepository()
_ds_repo = DatasourceRepository()

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class EnqueueRequest(BaseModel):
    """Request zum Einstellen eines Ingestion-Jobs in die Queue."""

    datasource_id: int
    job_type: Literal["incremental", "full"] = "incremental"
    triggered_by: Literal["manual", "cron", "api", "auto-resume", "chat"] = "manual"
    embedding_model: str = ""
    html_reingest: Literal["outdated", "all"] | None = None

    @model_validator(mode="after")
    def validate_html_mode(self):
        if self.html_reingest is not None and self.job_type != "incremental":
            raise ValueError("HTML-Neuaufbereitung erfordert einen inkrementellen Job")
        return self


class ScheduleRequest(BaseModel):
    """Request zum Setzen eines Cron-Schedules fuer eine Datenquelle."""

    cron_expr: str
    preset: str | None = None


# ---------------------------------------------------------------------------
# Job-Endpoints
# ---------------------------------------------------------------------------

# --- Migriert zu Admin-Registry: jobs.list (Sprint 04) ---
# @router.get("/api/admin/ingestion-jobs")
# async def list_ingestion_jobs(
#     ds_id: int | None = Query(None),
#     status: str | None = Query(None),
#     limit: int = Query(50, ge=1, le=500),
#     offset: int = Query(0, ge=0),
#     user: dict = Depends(get_admin_user),
# ) -> dict:
#     """Jobs-Liste — liest aus vorberechneter Cache-Tabelle."""
#     jobs, total = _cache_repo.get_cached_jobs(
#         ds_id=ds_id, status=status, limit=limit, offset=offset,
#     )
#     if not jobs and total == 0 and not status and ds_id is None:
#         from kiara.web.cache_refresh import refresh_jobs_cache
#         refresh_jobs_cache()
#         jobs, total = _cache_repo.get_cached_jobs(limit=limit, offset=offset)
#     return {"data": jobs, "total": total, "limit": limit, "offset": offset}

# --- Migriert zu Admin-Registry: jobs.status (Sprint 04) ---
# @router.get("/api/admin/ingestion-jobs/{job_id}")
# async def get_ingestion_job(job_id: int, user: dict = Depends(get_admin_user)) -> dict:
#     """Einzelner Job mit vollem stats_json + Resume-Ketten-Metadaten."""
#     from kiara.shared.db import db_query
#     with db_query() as db:
#         row = db.execute(
#             "SELECT j.*, d.name AS datasource_name "
#             "FROM ingestion_jobs j LEFT JOIN datasources d ON j.datasource_id = d.id "
#             "WHERE j.id = %s", (job_id,),
#         ).fetchone()
#         if not row:
#             raise HTTPException(status_code=404, detail="Job nicht gefunden")
#         job = dict(row)
#         try:
#             job["stats"] = json.loads(job.get("stats_json") or "{}")
#         except (json.JSONDecodeError, TypeError):
#             job["stats"] = {}
#         job.pop("stats_json", None)
#         _enrich_resume_chains([job], db)
#     from kiara.web.ingestion_worker import get_running_job_id, is_job_paused
#     if job["status"] == IngestionStatus.RUNNING:
#         if get_running_job_id() == job_id:
#             job["is_paused"] = is_job_paused(job_id)
#         else:
#             job["is_ghost"] = True
#     return job


@router.post("/api/admin/ingestion-jobs")
async def enqueue_ingestion_job(data: EnqueueRequest, user: dict = Depends(get_admin_user)) -> dict:
    """Job in Queue stellen."""
    from kiara.web.ingestion_worker import enqueue_job

    # DS pruefen
    ds = _ds_repo.get_by_id(data.datasource_id)
    if not ds:
        raise HTTPException(status_code=404, detail=ERR_DATASOURCE_NOT_FOUND)
    if not ds["is_active"]:
        raise HTTPException(status_code=400, detail="Datenquelle ist deaktiviert")

    try:
        html_options = {"html_reingest": data.html_reingest} if data.html_reingest else {}
        job_id = enqueue_job(
            data.datasource_id,
            job_type=data.job_type,
            triggered_by=data.triggered_by,
            embedding_model=data.embedding_model,
            **html_options,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    logger.info("Job #%s enqueued by %s", job_id, user["username"])
    return {"job_id": job_id, "message": "Job eingestellt"}


@router.post("/api/admin/ingestion-jobs/{job_id}/cancel")
async def cancel_ingestion_job(job_id: int, user: dict = Depends(get_admin_user)) -> dict:
    """Job abbrechen."""
    from kiara.web.ingestion_worker import request_cancel

    if not request_cancel(job_id):
        raise HTTPException(status_code=409, detail="Job kann nicht abgebrochen werden")

    logger.info("Job #%s cancel requested by %s", job_id, user["username"])
    from kiara.web.cache_refresh import refresh_jobs_cache

    safe_refresh(refresh_jobs_cache, job_id)
    return {"message": "Abbruch angefordert"}


@router.post("/api/admin/ingestion-jobs/{job_id}/retry-html")
async def retry_html_ingestion_job(job_id: int, user: dict = Depends(get_admin_user)) -> dict:
    """HTML-Auftrag mit seiner gespeicherten Auswahl und Vertragsbindung wiederholen."""
    from kiara.web.ingestion_worker import retry_html_job

    try:
        new_job_id = retry_html_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    logger.info("HTML job #%s retried as #%s by %s", job_id, new_job_id, user["username"])
    return {"job_id": new_job_id, "message": "HTML-Job erneut eingestellt"}


@router.post("/api/admin/ingestion-jobs/{job_id}/pause")
async def pause_ingestion_job(job_id: int, user: dict = Depends(get_admin_user)) -> dict:
    """Pause/Resume Toggle."""
    from kiara.web.ingestion_worker import get_running_job_id

    current = get_running_job_id()
    if current != job_id:
        with db_query() as db:
            job_row = db.execute(
                "SELECT status FROM ingestion_jobs WHERE id = %s", (job_id,)
            ).fetchone()
        if not job_row:
            raise HTTPException(status_code=404, detail=f"Job {job_id} nicht gefunden")
        raise HTTPException(
            status_code=409, detail=f"Job {job_id} laeuft nicht (Status: {job_row['status']})"
        )

    # Atomarer Pause/Resume Toggle: SELECT FOR UPDATE + UPDATE in einer Transaktion
    with db_transaction() as db:
        row = db.execute(
            "SELECT requested_action FROM ingestion_jobs WHERE id = %s AND status = %s FOR UPDATE",
            (job_id, IngestionStatus.RUNNING.value),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=409, detail=f"Job {job_id} laeuft nicht mehr")
        if row["requested_action"] == "pause":
            db.execute(
                "UPDATE ingestion_jobs SET requested_action = NULL WHERE id = %s",
                (job_id,),
            )
            msg = {"message": "Ingestion fortgesetzt", "paused": False}
        elif row["requested_action"] == "cancel":
            raise HTTPException(status_code=409, detail="Job wird bereits abgebrochen")
        else:
            db.execute(
                "UPDATE ingestion_jobs SET requested_action = 'pause' WHERE id = %s",
                (job_id,),
            )
            msg = {"message": "Ingestion pausiert", "paused": True}

    logger.info(
        "Job #%s %s by %s", job_id, "resumed" if not msg["paused"] else "paused", user["username"]
    )
    from kiara.web.cache_refresh import refresh_jobs_cache

    safe_refresh(refresh_jobs_cache, job_id)
    return msg


# --- Migriert zu Admin-Registry: jobs.logs (Sprint 04) ---
# @router.get("/api/admin/ingestion-jobs/{job_id}/file-logs")
# async def get_file_logs(
#     job_id: int,
#     status: str | None = Query(None),
#     search: str | None = Query(None),
#     sort: str = Query("file_name"),
#     direction: str = Query("asc"),
#     limit: int = Query(100, ge=1, le=1000),
#     offset: int = Query(0, ge=0),
#     user: dict = Depends(get_admin_user),
# ) -> dict:
#     """Per-File-Protokoll fuer einen Ingestion-Job."""
#     if not _job_repo.exists(job_id):
#         raise HTTPException(status_code=404, detail="Job nicht gefunden")
#     valid_status = status if status and status in ("success", "error", "skipped") else None
#     rows, total = _file_log_repo.list_by_job(
#         job_id, status=valid_status, search=search,
#         sort=sort, direction=direction, limit=limit, offset=offset,
#     )
#     summary_data = _file_log_repo.aggregate_by_status(job_id)
#     summary = {"success": 0, "error": 0, "skipped": 0, "total_chunks": 0, "avg_duration_ms": 0}
#     total_files_for_avg = 0
#     total_dur = 0
#     for s, sr in summary_data.items():
#         cnt = sr["count"]
#         summary[s] = cnt
#         summary["total_chunks"] += sr["chunks"] or 0
#         total_dur += (sr["avg_ms"] or 0) * cnt
#         total_files_for_avg += cnt
#     if total_files_for_avg:
#         summary["avg_duration_ms"] = int(total_dur / total_files_for_avg)
#     return {"data": rows, "total": total, "summary": summary}


# --- Migriert zu Admin-Registry: jobs.cleanup_history (Sprint 05) ---
# @router.delete("/api/admin/ingestion-jobs/history")
# async def delete_job_history(
#     days: int = Query(30, ge=1),
#     user: dict = Depends(get_admin_user),
# ) -> dict:
#     """Alte abgeschlossene Jobs loeschen (aelter als X Tage)."""
#     count = _job_repo.cleanup_old(days=days)
#     logger.info("Job-Historie bereinigt: %s Jobs geloescht (aelter als %s Tage) by %s",
#                 count, days, user["username"])
#     return {"deleted": count, "message": f"{count} Jobs geloescht"}

# --- Migriert zu Admin-Registry: jobs.delete (Sprint 05) ---
# @router.delete("/api/admin/ingestion-jobs/{job_id}")
# async def delete_single_job(
#     job_id: int,
#     user: dict = Depends(get_admin_user),
# ) -> dict:
#     """Einzelnen abgeschlossenen Job loeschen."""
#     job = _job_repo.get_by_id(job_id)
#     if not job:
#         raise HTTPException(status_code=404, detail="Job nicht gefunden")
#     if job["status"] in ("running", "pending", "paused"):
#         raise HTTPException(status_code=409, detail="Laufende/ausstehende Jobs koennen nicht geloescht werden")
#     from kiara.shared.db import db_transaction
#     with db_transaction() as db:
#         db.execute("DELETE FROM ingestion_file_logs WHERE job_id = %s", (job_id,))
#         db.execute("DELETE FROM ingestion_jobs WHERE id = %s", (job_id,))
#     logger.info("Job #%d geloescht by %s", job_id, user["username"])
#     from kiara.web.cache_refresh import refresh_jobs_cache
#     refresh_jobs_cache()
#     return {"message": f"Job #{job_id} geloescht"}


# ---------------------------------------------------------------------------
# Resume-Chain-Hilfsfunktion
# ---------------------------------------------------------------------------


def _enrich_resume_chains(jobs, db):
    """Reichert Jobs mit Resume-Ketten-Metadaten an.

    Fuer jeden Job mit resumed_from_job: Kette rueckwaerts traversieren und
    Felder hinzufuegen (chain_length, original_job_id, ancestor_duration, chain).
    """
    for job in jobs:
        if not job.get("resumed_from_job"):
            continue
        chain = []
        ancestor_duration = 0
        current_id = job["resumed_from_job"]
        seen = set()
        while current_id and current_id not in seen:
            seen.add(current_id)
            ancestor = db.execute(
                "SELECT id, status, started_at, finished_at, error_message, "
                "stats_json, resumed_from_job "
                "FROM ingestion_jobs WHERE id = %s",
                (current_id,),
            ).fetchone()
            if not ancestor:
                break
            # Dauer des Vorfahren berechnen
            a_duration = 0
            if ancestor["started_at"] and ancestor["finished_at"]:
                a_stats = safe_json_loads(ancestor["stats_json"], {})
                if isinstance(a_stats, dict):
                    a_duration = a_stats.get("duration_seconds", 0)
            ancestor_duration += a_duration
            chain.append(
                {
                    "id": ancestor["id"],
                    "status": ancestor["status"],
                    "started_at": ancestor["started_at"],
                    "finished_at": ancestor["finished_at"],
                    "error_message": ancestor["error_message"] or "",
                    "duration_seconds": a_duration,
                }
            )
            current_id = ancestor["resumed_from_job"]
        if chain:
            job["resume_chain_length"] = len(chain) + 1  # inkl. aktueller Job
            job["resume_chain_original_job_id"] = chain[-1]["id"]
            job["resume_chain_ancestor_duration"] = ancestor_duration
            job["resume_chain"] = chain


# ---------------------------------------------------------------------------
# Schedule-Endpoints
# ---------------------------------------------------------------------------

# --- Migriert zu Admin-Registry: jobs.schedule (Sprint 04) ---
# @router.get("/api/admin/datasources/{ds_id}/schedule")
# async def get_schedule(ds_id: int, user: dict = Depends(get_admin_user)) -> dict:
#     """Schedule fuer eine Datenquelle lesen."""
#     from kiara.web.cron_manager import read_schedules
#     schedules = read_schedules()
#     schedule = schedules.get(ds_id)
#     if schedule:
#         return {"ds_id": ds_id, "active": True, **schedule}
#     return {"ds_id": ds_id, "active": False, "cron_expr": "", "name": ""}

# --- Migriert zu Admin-Registry: jobs.schedule.set (Sprint 05) ---
# @router.put("/api/admin/datasources/{ds_id}/schedule")
# async def set_schedule(ds_id: int, data: ScheduleRequest, user: dict = Depends(get_admin_user)) -> dict:
#     """Schedule setzen oder aktualisieren."""
#     from kiara.web.cron_manager import write_schedule, validate_cron_expr, INTERVAL_PRESETS
#     ds = _ds_repo.get_by_id(ds_id)
#     if not ds:
#         raise HTTPException(status_code=404, detail=ERR_DATASOURCE_NOT_FOUND)
#     cron_expr = data.cron_expr
#     if data.preset and data.preset in INTERVAL_PRESETS:
#         cron_expr = INTERVAL_PRESETS[data.preset]
#     if not validate_cron_expr(cron_expr):
#         raise HTTPException(status_code=400, detail=f"Ungueltiger Cron-Ausdruck: {cron_expr}")
#     try:
#         write_schedule(ds_id, ds["name"], cron_expr)
#     except Exception as exc:
#         raise HTTPException(status_code=500, detail=str(exc))
#     logger.info("Schedule gesetzt fuer DS %s by %s: %s", ds_id, user["username"], cron_expr)
#     return {"message": "Schedule gesetzt", "cron_expr": cron_expr}

# --- Migriert zu Admin-Registry: jobs.schedule.delete (Sprint 05) ---
# @router.delete("/api/admin/datasources/{ds_id}/schedule")
# async def delete_schedule(ds_id: int, user: dict = Depends(get_admin_user)) -> dict:
#     """Schedule entfernen."""
#     from kiara.web.cron_manager import remove_schedule
#     try:
#         remove_schedule(ds_id)
#     except Exception as exc:
#         raise HTTPException(status_code=500, detail=str(exc))
#     logger.info("Schedule entfernt fuer DS %s by %s", ds_id, user["username"])
#     return {"message": "Schedule entfernt"}
