import json

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult
from kiara.repositories.cache import CacheRepository
from kiara.repositories.job import FileLogRepository, IngestionJobRepository
from kiara.shared.enums import IngestionStatus

_cache_repo = CacheRepository()
_job_repo = IngestionJobRepository()
_file_log_repo = FileLogRepository()


@admin_function(
    name="jobs.list",
    category="jobs",
    description="Ingestion-Jobs auflisten (paginiert, filterbar)",
    help="Listet Ingestion-Jobs auf, optional gefiltert nach Datenquelle oder "
    "Status. Unterstützt Paginierung mit limit/offset.",
    examples=[
        "Zeig mir alle Jobs",
        "Jobs für Datenquelle 3",
        "Laufende Jobs anzeigen",
    ],
    parameters={
        "type": "object",
        "properties": {
            "ds_id": {"type": "integer", "description": "Filter nach Datenquellen-ID"},
            "status": {
                "type": "string",
                "description": "Filter nach Status (pending, running, completed, error)",
            },
            "limit": {"type": "integer", "description": "Maximale Anzahl Ergebnisse (Default: 50)"},
            "offset": {"type": "integer", "description": "Offset für Paginierung (Default: 0)"},
        },
        "required": [],
    },
    returns="Paginierte Job-Liste mit Gesamtanzahl",
    raw_response=True,
    http=("GET", "/api/admin/ingestion-jobs"),
)
async def jobs_list(
    user: dict, ds_id: int = None, status: str = None, limit: int = 50, offset: int = 0
) -> AdminResult:
    jobs, total = _cache_repo.get_cached_jobs(
        ds_id=ds_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    if not jobs and total == 0 and not status and ds_id is None:
        from kiara.web.cache_refresh import refresh_jobs_cache

        refresh_jobs_cache()
        jobs, total = _cache_repo.get_cached_jobs(limit=limit, offset=offset)
    return AdminResult.ok(
        "Jobs", data={"data": jobs, "total": total, "limit": limit, "offset": offset}
    )


@admin_function(
    name="jobs.status",
    category="jobs",
    description="Status eines einzelnen Jobs",
    help="Zeigt detaillierte Informationen zu einem Ingestion-Job: Status, "
    "Fortschritt, Statistiken und Resume-Ketten.",
    examples=[
        "Status von Job 42",
        "Was macht Job 7?",
    ],
    parameters={
        "type": "object",
        "properties": {"job_id": {"type": "integer", "description": "Job-ID"}},
        "required": ["job_id"],
    },
    returns="Job-Details mit Status, Fortschritt und Statistiken",
    raw_response=True,
    http=("GET", "/api/admin/ingestion-jobs/{job_id}"),
)
async def jobs_status(user: dict, job_id: int) -> AdminResult:
    from kiara.shared.db import db_query
    from kiara.web.api_ingestion_jobs import _enrich_resume_chains

    with db_query() as db:
        row = db.execute(
            "SELECT j.*, d.name AS datasource_name "
            "FROM ingestion_jobs j LEFT JOIN datasources d ON j.datasource_id = d.id "
            "WHERE j.id = %s",
            (job_id,),
        ).fetchone()
        if not row:
            return AdminResult.fail("Job nicht gefunden")

        job = dict(row)
        try:
            job["stats"] = json.loads(job.get("stats_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            job["stats"] = {}
        job.pop("stats_json", None)

        _enrich_resume_chains([job], db)

    from kiara.web.ingestion_worker import get_running_job_id, is_job_paused

    if job["status"] == IngestionStatus.RUNNING:
        if get_running_job_id() == job_id:
            job["is_paused"] = is_job_paused(job_id)
        else:
            job["is_ghost"] = True

    return AdminResult.ok("Job-Status", data=job)


@admin_function(
    name="jobs.logs",
    category="jobs",
    description="File-Logs eines Ingestion-Jobs",
    help="Zeigt das Per-File-Protokoll eines Jobs: Welche Dateien wurden "
    "verarbeitet, übersprungen oder hatten Fehler. Filterbar und paginiert.",
    examples=[
        "File-Logs von Job 42",
        "Fehlerhafte Dateien in Job 7",
    ],
    parameters={
        "type": "object",
        "properties": {
            "job_id": {"type": "integer", "description": "Job-ID"},
            "status": {"type": "string", "description": "Filter nach File-Status"},
            "search": {"type": "string", "description": "Dateinamen-Suche"},
            "sort": {"type": "string", "description": "Sortierfeld (Default: file_name)"},
            "direction": {
                "type": "string",
                "description": "Sortierrichtung: asc/desc (Default: asc)",
            },
            "limit": {"type": "integer", "description": "Max Ergebnisse (Default: 100, max: 1000)"},
            "offset": {"type": "integer", "description": "Offset (Default: 0)"},
        },
        "required": ["job_id"],
    },
    returns="File-Logs mit Summary (success/error/skipped Counts)",
    raw_response=True,
    http=("GET", "/api/admin/ingestion-jobs/{job_id}/file-logs"),
)
async def jobs_logs(
    user: dict,
    job_id: int,
    status: str = None,
    search: str = None,
    sort: str = "file_name",
    direction: str = "asc",
    limit: int = 100,
    offset: int = 0,
) -> AdminResult:
    if not _job_repo.exists(job_id):
        return AdminResult.fail("Job nicht gefunden")

    valid_status = status if status and status in ("success", "error", "skipped") else None

    rows, total = _file_log_repo.list_by_job(
        job_id,
        status=valid_status,
        search=search,
        sort=sort,
        direction=direction,
        limit=limit,
        offset=offset,
    )

    summary_data = _file_log_repo.aggregate_by_status(job_id)
    summary = {"success": 0, "error": 0, "skipped": 0, "total_chunks": 0, "avg_duration_ms": 0}
    total_files_for_avg = 0
    total_dur = 0
    for s, sr in summary_data.items():
        cnt = sr["count"]
        summary[s] = cnt
        summary["total_chunks"] += sr["chunks"] or 0
        total_dur += (sr["avg_ms"] or 0) * cnt
        total_files_for_avg += cnt
    if total_files_for_avg:
        summary["avg_duration_ms"] = int(total_dur / total_files_for_avg)

    return AdminResult.ok("File-Logs", data={"data": rows, "total": total, "summary": summary})


@admin_function(
    name="jobs.schedule",
    category="jobs",
    description="Cron-Schedule einer Datenquelle",
    help="Zeigt den konfigurierten Cron-Schedule für automatische Ingestion einer Datenquelle.",
    examples=[
        "Schedule von DS 1",
        "Hat Datenquelle 3 einen Schedule?",
    ],
    parameters={
        "type": "object",
        "properties": {"ds_id": {"type": "integer", "description": "ID der Datenquelle"}},
        "required": ["ds_id"],
    },
    returns="Schedule mit Cron-Ausdruck und Aktivitätsstatus",
    raw_response=True,
    http=("GET", "/api/admin/datasources/{ds_id}/schedule"),
)
async def jobs_schedule(user: dict, ds_id: int) -> AdminResult:
    from kiara.web.cron_manager import read_schedules

    schedules = read_schedules()
    schedule = schedules.get(ds_id)
    if schedule:
        data = {"ds_id": ds_id, "active": True, **schedule}
    else:
        data = {"ds_id": ds_id, "active": False, "cron_expr": "", "name": ""}
    return AdminResult.ok("Schedule", data=data)


# ---------------------------------------------------------------------------
# Write-Funktionen (Sprint 05)
# ---------------------------------------------------------------------------


@admin_function(
    name="jobs.start",
    category="jobs",
    description="Inkrementelle Indexierung starten",
    help="Startet eine inkrementelle Indexierung für eine Datenquelle. "
    "Nur neue/geänderte Dateien werden verarbeitet (File-Hash-Vergleich). "
    "Die Datenquelle muss aktiv sein (is_active=1).",
    examples=[
        "Starte die Indexierung für Datenquelle 5",
        "Indexiere DS 12",
    ],
    parameters={
        "type": "object",
        "properties": {
            "ds_id": {"type": "integer", "description": "ID der Datenquelle"},
        },
        "required": ["ds_id"],
    },
    returns="Job-ID und Status",
)
async def jobs_start(user: dict, ds_id: int) -> AdminResult:
    from kiara.config import load_config
    from kiara.repositories.datasource import DatasourceRepository

    ds_repo = DatasourceRepository()
    ds = ds_repo.get_by_id(ds_id)
    if not ds:
        return AdminResult.fail(f"Datenquelle {ds_id} nicht gefunden")
    if not ds.get("is_active"):
        return AdminResult.fail(f"Datenquelle {ds_id} ist deaktiviert")
    from kiara.web.ingestion_worker import enqueue_job

    config = load_config()
    # enqueue_job Signatur: (ds_id, job_type, triggered_by, *, embedding_model)
    # ValueError bei bereits aktivem Job fuer diese DS
    try:
        job_id = enqueue_job(
            ds_id,
            job_type="incremental",
            triggered_by="chat",
            embedding_model=config["embedding_model"],
        )
    except ValueError as exc:
        return AdminResult.fail(str(exc))
    return AdminResult.ok(
        f"Ingestion gestartet (Job {job_id})",
        data={"job_id": job_id, "ds_id": ds_id, "mode": "incremental"},
    )


@admin_function(
    name="jobs.start_full",
    category="jobs",
    description="Voll-Neuindexierung starten",
    help="Startet eine vollständige Neuindexierung: ALLE bestehenden Embeddings der "
    "Datenquelle werden gelöscht und alle Dateien neu verarbeitet. "
    "Kann bei großen Datenquellen Stunden dauern. IRREVERSIBEL.",
    examples=[
        "Starte Voll-Reindexierung für DS 5",
        "Alle Dateien von Datenquelle 12 neu indexieren",
    ],
    parameters={
        "type": "object",
        "properties": {
            "ds_id": {"type": "integer", "description": "ID der Datenquelle"},
        },
        "required": ["ds_id"],
    },
    returns="Job-ID und Status",
    requires_confirmation=True,
    confirm_template="ds {ds_id} voll-reindexieren",
    changelog_area="jobs",
    changelog_action="start_full",
    pk_param="ds_id",
)
async def jobs_start_full(user: dict, ds_id: int) -> AdminResult:
    from kiara.config import load_config
    from kiara.repositories.datasource import DatasourceRepository

    ds_repo = DatasourceRepository()
    ds = ds_repo.get_by_id(ds_id)
    if not ds:
        return AdminResult.fail(f"Datenquelle {ds_id} nicht gefunden")
    if not ds.get("is_active"):
        return AdminResult.fail(f"Datenquelle {ds_id} ist deaktiviert")
    from kiara.web.ingestion_worker import enqueue_job

    config = load_config()
    # enqueue_job Signatur: (ds_id, job_type, triggered_by, *, embedding_model)
    # ValueError bei bereits aktivem Job fuer diese DS
    try:
        job_id = enqueue_job(
            ds_id, job_type="full", triggered_by="chat", embedding_model=config["embedding_model"]
        )
    except ValueError as exc:
        return AdminResult.fail(str(exc))
    return AdminResult.ok(
        f"Voll-Ingestion gestartet (Job {job_id})",
        data={"job_id": job_id, "ds_id": ds_id, "mode": "full"},
    )


@admin_function(
    name="jobs.cancel",
    category="jobs",
    description="Laufenden Ingestion-Job abbrechen",
    help="Bricht einen laufenden oder wartenden Job ab. Der Abbruch erfolgt "
    "über die DB-Spalte requested_action (prozess-unabhängig).",
    examples=[
        "Breche Job 42 ab",
        "Stoppe die laufende Indexierung",
    ],
    parameters={
        "type": "object",
        "properties": {
            "job_id": {"type": "integer", "description": "Job-ID"},
        },
        "required": ["job_id"],
    },
    returns="Abbruch-Bestätigung",
    requires_confirmation=True,
    confirm_template="job {job_id} abbrechen",
    changelog_area="jobs",
    changelog_action="cancel",
    pk_param="job_id",
)
async def jobs_cancel(user: dict, job_id: int) -> AdminResult:
    from kiara.web.ingestion_worker import request_cancel

    success = request_cancel(job_id)
    if not success:
        return AdminResult.fail(
            f"Job {job_id} kann nicht abgebrochen werden (nicht laufend/wartend)"
        )
    from kiara.web.cache_refresh import refresh_jobs_cache

    refresh_jobs_cache()
    return AdminResult.ok(
        f"Job {job_id} wird abgebrochen", data={"job_id": job_id, "status": "cancelling"}
    )


@admin_function(
    name="jobs.pause",
    category="jobs",
    description="Laufenden Job pausieren",
    help="Pausiert einen aktuell laufenden Job. Der Job muss der aktuell "
    "ausgeführte sein (get_running_job_id) und darf nicht bereits pausiert sein.",
    examples=[
        "Pausiere Job 42",
        "Halte die Indexierung an",
    ],
    parameters={
        "type": "object",
        "properties": {
            "job_id": {"type": "integer", "description": "Job-ID"},
        },
        "required": ["job_id"],
    },
    returns="Pause-Bestätigung",
)
async def jobs_pause(user: dict, job_id: int) -> AdminResult:
    from kiara.web.ingestion_worker import get_running_job_id, is_job_paused, request_pause

    running_id = get_running_job_id()
    if running_id != job_id:
        return AdminResult.fail(f"Job {job_id} läuft nicht (aktuell: {running_id or 'keiner'})")
    if is_job_paused(job_id):
        return AdminResult.fail(f"Job {job_id} ist bereits pausiert")
    if not request_pause(job_id):
        return AdminResult.fail(f"Job {job_id} konnte nicht pausiert werden")
    from kiara.web.cache_refresh import refresh_jobs_cache

    refresh_jobs_cache()
    return AdminResult.ok(f"Job {job_id} pausiert", data={"job_id": job_id, "paused": True})


@admin_function(
    name="jobs.resume",
    category="jobs",
    description="Pausierten Job fortsetzen",
    help="Setzt einen pausierten Job fort. Der Job muss pausiert sein.",
    examples=[
        "Setze Job 42 fort",
        "Indexierung weiterlaufen lassen",
    ],
    parameters={
        "type": "object",
        "properties": {
            "job_id": {"type": "integer", "description": "Job-ID"},
        },
        "required": ["job_id"],
    },
    returns="Fortsetzen-Bestätigung",
)
async def jobs_resume(user: dict, job_id: int) -> AdminResult:
    from kiara.web.ingestion_worker import is_job_paused, request_resume

    if not is_job_paused(job_id):
        return AdminResult.fail(f"Job {job_id} ist nicht pausiert")
    if not request_resume(job_id):
        return AdminResult.fail(f"Job {job_id} konnte nicht fortgesetzt werden")
    from kiara.web.cache_refresh import refresh_jobs_cache

    refresh_jobs_cache()
    return AdminResult.ok(f"Job {job_id} fortgesetzt", data={"job_id": job_id, "paused": False})


@admin_function(
    name="jobs.cleanup_history",
    category="jobs",
    description="Alte abgeschlossene Jobs löschen",
    help="Löscht alle abgeschlossenen Jobs die älter als die angegebene Anzahl "
    "Tage sind. Laufende/wartende/pausierte Jobs werden nicht gelöscht.",
    examples=[
        "Lösche Jobs älter als 30 Tage",
        "Job-History bereinigen",
    ],
    parameters={
        "type": "object",
        "properties": {
            "days": {
                "type": "integer",
                "description": "Jobs älter als X Tage löschen (Default: 30)",
            },
        },
        "required": [],
    },
    returns="Anzahl gelöschter Jobs",
    requires_confirmation=True,
    confirm_template="jobs älter als {days} tage löschen",
    changelog_area="jobs",
    changelog_action="delete",
    raw_response=True,
    http=("DELETE", "/api/admin/ingestion-jobs/history"),
)
async def jobs_cleanup_history(user: dict, days: int = 30) -> AdminResult:
    days = max(1, days)
    job_repo = IngestionJobRepository()
    count = job_repo.cleanup_old(days=days)
    # raw_response=True → Format muss bestehendem Endpunkt entsprechen
    return AdminResult.ok(
        f"{count} alte Jobs gelöscht", data={"deleted": count, "message": f"{count} Jobs geloescht"}
    )


@admin_function(
    name="jobs.delete",
    category="jobs",
    description="Einzelnen Job löschen",
    help="Löscht einen einzelnen abgeschlossenen/fehlerhaften Job inkl. File-Logs. "
    "Laufende, wartende oder pausierte Jobs können nicht gelöscht werden.",
    examples=[
        "Lösche Job 42",
    ],
    parameters={
        "type": "object",
        "properties": {
            "job_id": {"type": "integer", "description": "Job-ID"},
        },
        "required": ["job_id"],
    },
    returns="Löschbestätigung",
    requires_confirmation=True,
    confirm_template="job {job_id} löschen",
    pk_param="job_id",
    changelog_area="jobs",
    changelog_action="delete",
    raw_response=True,
    http=("DELETE", "/api/admin/ingestion-jobs/{job_id}"),
)
async def jobs_delete(user: dict, job_id: int) -> AdminResult:
    job_repo = IngestionJobRepository()
    job = job_repo.get_by_id(job_id)
    if not job:
        return AdminResult.fail(f"Job {job_id} nicht gefunden")
    if job["status"] in ("running", "pending", "paused"):
        return AdminResult.fail(f"Job {job_id} ist noch aktiv (Status: {job['status']})")
    from kiara.shared.db import db_transaction

    with db_transaction() as conn:
        conn.execute("DELETE FROM ingestion_file_logs WHERE job_id = %s", (job_id,))
        conn.execute("DELETE FROM ingestion_jobs WHERE id = %s", (job_id,))
    from kiara.web.cache_refresh import refresh_jobs_cache

    refresh_jobs_cache()
    # raw_response=True → Format muss bestehendem Endpunkt entsprechen
    return AdminResult.ok(f"Job {job_id} gelöscht", data={"message": f"Job #{job_id} geloescht"})


# ---------------------------------------------------------------------------
# Schedule-Verwaltung (Sprint 05)
# ---------------------------------------------------------------------------


@admin_function(
    name="jobs.schedule.set",
    category="jobs",
    description="Automatischen Sync-Schedule setzen",
    help="Setzt einen Cron-Schedule für automatische Indexierung einer Datenquelle. "
    "Cron-Ausdruck mit 5 Feldern (Minute Stunde Tag Monat Wochentag). "
    "Beispiel: '0 2 * * *' = täglich um 02:00 Uhr.",
    examples=[
        "Setze tägliche Synchronisation um 3 Uhr für DS 5",
        "Schedule für DS 12: jeden Montag um 6:00",
    ],
    parameters={
        "type": "object",
        "properties": {
            "ds_id": {"type": "integer", "description": "ID der Datenquelle"},
            "cron_expr": {"type": "string", "description": "Cron-Ausdruck (5 Felder)"},
            "preset": {
                "type": "string",
                "description": "Preset-Name (optional, z.B. 'daily', 'weekly')",
            },
        },
        "required": ["ds_id", "cron_expr"],
    },
    returns="Bestätigung des gesetzten Schedules",
    pk_param="ds_id",
    changelog_area="schedules",
    changelog_action="update",
    raw_response=True,
    http=("PUT", "/api/admin/datasources/{ds_id}/schedule"),
)
async def jobs_schedule_set(
    user: dict, ds_id: int, cron_expr: str, preset: str | None = None
) -> AdminResult:
    from kiara.repositories.datasource import DatasourceRepository

    ds_repo = DatasourceRepository()
    ds = ds_repo.get_by_id(ds_id)
    if not ds:
        return AdminResult.fail(f"Datenquelle {ds_id} nicht gefunden")
    # write_schedule Signatur: (ds_id: int, name: str, cron_expr: str)
    from kiara.web.cron_manager import write_schedule

    # write_schedule wirft ValueError bei ungueltigem Cron-Ausdruck
    try:
        write_schedule(ds_id, ds["name"], cron_expr)
    except ValueError as exc:
        return AdminResult.fail(str(exc))
    # raw_response=True → Format muss bestehendem Endpunkt entsprechen
    return AdminResult.ok(
        f"Schedule für DS {ds_id} gesetzt: {cron_expr}",
        data={"message": "Schedule gesetzt", "cron_expr": cron_expr},
    )


@admin_function(
    name="jobs.schedule.delete",
    category="jobs",
    description="Automatischen Sync-Schedule entfernen",
    help="Entfernt den Cron-Schedule einer Datenquelle. Die Datenquelle wird "
    "nicht mehr automatisch synchronisiert.",
    examples=[
        "Entferne den Schedule von DS 5",
        "Automatische Synchronisation für DS 12 stoppen",
    ],
    parameters={
        "type": "object",
        "properties": {
            "ds_id": {"type": "integer", "description": "ID der Datenquelle"},
        },
        "required": ["ds_id"],
    },
    returns="Bestätigung der Schedule-Entfernung",
    requires_confirmation=True,
    confirm_template="schedule für ds {ds_id} löschen",
    pk_param="ds_id",
    changelog_area="schedules",
    changelog_action="delete",
    raw_response=True,
    http=("DELETE", "/api/admin/datasources/{ds_id}/schedule"),
)
async def jobs_schedule_delete(user: dict, ds_id: int) -> AdminResult:
    from kiara.repositories.datasource import DatasourceRepository

    ds_repo = DatasourceRepository()
    if not ds_repo.exists(ds_id):
        return AdminResult.fail(f"Datenquelle {ds_id} nicht gefunden")
    from kiara.web.cron_manager import remove_schedule

    remove_schedule(ds_id)
    # raw_response=True → Format muss bestehendem Endpunkt entsprechen
    return AdminResult.ok(
        f"Schedule für DS {ds_id} entfernt", data={"message": "Schedule entfernt"}
    )
