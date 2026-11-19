"""Queue lifecycle for PKG-only jobs that do not write a vector index."""

import json
import time

from kiara.cad.ingestion import job_is_sql_only, sql_only_source, targets
from kiara.shared.db import db_transaction


def enqueue(ds_id, job_type, triggered_by, config, *, resumed_from=None, selection=None):
    from kiara.connectors.registry import source_unavailable_reason
    from kiara.web.ingestion_worker import _ACTIVE_JOB_STATUSES, INGESTION_DATASOURCE_LOCK_NAMESPACE

    with db_transaction() as conn:
        conn.execute(
            "SELECT pg_advisory_xact_lock(%s,%s)", (INGESTION_DATASOURCE_LOCK_NAMESPACE, ds_id)
        )
        ds = conn.execute("SELECT * FROM datasources WHERE id=%s FOR UPDATE", (ds_id,)).fetchone()
        if not ds or not ds["is_active"]:
            raise ValueError("Datenquelle fehlt oder ist deaktiviert")
        ds = dict(ds)
        if selection:
            if isinstance(selection, str):
                selection = json.loads(selection)
            for key, value in selection.get("cad_targets", {}).items():
                ds["ds_" + key] = value
            ds["ds_file_types"] = json.dumps(selection.get("cad_file_types", [".pkg"]))
            ds["ds_accept_all_text"] = False
        if not sql_only_source(ds, config):
            raise ValueError("PKG-Ingest-Einstellungen wurden geändert; Job erneut starten")
        if reason := source_unavailable_reason(dict(ds)):
            raise ValueError(reason)
        if conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id=%s AND status=ANY(%s)",
            (ds_id, list(_ACTIVE_JOB_STATUSES)),
        ).fetchone():
            raise ValueError("Datenquelle hat bereits einen aktiven Job")
        stats = {"cad_sql_only": True, "cad_targets": targets(dict(ds)), "cad_file_types": [".pkg"]}
        return conn.execute(
            "INSERT INTO ingestion_jobs(datasource_id,job_type,triggered_by,embedding_model,stats_json,queued_at,resumed_from_job) "
            "VALUES(%s,%s,%s,'',%s::jsonb,now(),%s) RETURNING id",
            (ds_id, job_type, triggered_by, json.dumps(stats), resumed_from),
        ).fetchone()["id"]


def claim(job_id):
    with db_transaction() as conn:
        row = conn.execute(
            "SELECT j.* FROM ingestion_jobs j JOIN datasources d ON d.id=j.datasource_id "
            "WHERE j.id=%s AND j.status='pending' AND d.is_active=true FOR UPDATE OF j SKIP LOCKED",
            (job_id,),
        ).fetchone()
        if not row or not job_is_sql_only(dict(row)):
            return None
        conn.execute(
            "UPDATE ingestion_jobs SET status='initializing',started_at=now() WHERE id=%s",
            (job_id,),
        )
        return {**dict(row), "status": "initializing"}


def start(row):
    with db_transaction() as conn:
        changed = conn.execute(
            "UPDATE ingestion_jobs SET status='running' WHERE id=%s AND status='initializing' "
            "AND requested_action IS NULL RETURNING id",
            (row["id"],),
        ).fetchone()
        if changed:
            conn.execute(
                "UPDATE datasources SET ingestion_status='running' WHERE id=%s",
                (row["datasource_id"],),
            )
        return bool(changed)


def execute(worker, row, check_control):
    from kiara.ingestion.datasource_runtime import build_datasource_overrides
    from kiara.ingestion.pipeline import run_ingestion
    from kiara.web.api_utils import safe_refresh
    from kiara.web.cache_refresh import refresh_datasource_cache, refresh_jobs_cache

    started = time.monotonic()
    job_id, ds_id = row["id"], row["datasource_id"]
    ctx = worker._load_job_context(row, ds_id, job_id)
    config, ds = ctx["config"], ctx["ds_row"]
    tracker = getattr(worker, "_activity_tracker", None)
    if tracker is not None:
        tracker.begin_job(job_id, ds_id, ds["type"])
    if not sql_only_source(dict(ds), config):
        raise ValueError("Ungültige PKG-Jobauswahl")
    if not start(row):
        return
    callbacks = worker._build_callbacks(job_id, ds_id)
    worker._set_job_phase(job_id, "CAD-Datenquelle vorbereiten…")
    path = ctx["connector_cls"].prepare_and_scan(
        ctx["conn_config"], dict(ds), config, control_callback=check_control
    )
    if path is None:
        if check_control() in {"cancel", "shutdown"}:
            if check_control() == "cancel":
                worker._finalize_cancelled(job_id, ds_id, {})
            return
        raise ValueError("CAD-Datenquelle konnte nicht vorbereitet werden")
    config.setdefault("datasource", {})["path"] = path
    try:
        stats = run_ingestion(
            config,
            datasource_id=ds_id,
            incremental=row["job_type"] != "full",
            ds_overrides=build_datasource_overrides(ds),
            control_callback=check_control,
            progress_callback=callbacks["on_progress"],
            file_log_callback=callbacks["on_file_log"],
            file_start_callback=callbacks["on_file_start"],
            status_callback=callbacks["on_status"],
        )
    finally:
        callbacks["flush_file_logs"]()
    stats.update(
        cad_sql_only=True,
        cad_targets=targets(dict(ds)),
        cad_file_types=[".pkg"],
        duration_seconds=int(time.monotonic() - started),
    )
    signal = check_control()
    if signal == "shutdown":
        return
    if signal == "cancel" or stats.get("cancelled"):
        worker._finalize_cancelled(job_id, ds_id, stats)
    elif stats.get("abort_reason"):
        worker._finalize_error(job_id, ds_id, stats["abort_reason"], stats=stats)
    else:
        with db_transaction() as conn:
            conn.execute(
                "UPDATE ingestion_jobs SET status='success',finished_at=now(),requested_action=NULL,stats_json=%s::jsonb WHERE id=%s",
                (json.dumps(stats), job_id),
            )
            conn.execute(
                "UPDATE datasources SET ingestion_status='success',last_error='',processed_files=%s WHERE id=%s",
                (stats["processed_files"], ds_id),
            )
        # Existing vector counts and embeddings remain untouched.
        safe_refresh(refresh_datasource_cache, ds_id)
        safe_refresh(refresh_jobs_cache, job_id)


def preserve_targets(conn, job_id, stats):
    row = conn.execute("SELECT stats_json FROM ingestion_jobs WHERE id=%s", (job_id,)).fetchone()
    old = row["stats_json"] or {} if row else {}
    if isinstance(old, str):
        old = json.loads(old)
    for key in ("cad_targets", "cad_sql_only", "cad_file_types"):
        if key in old:
            stats.setdefault(key, old[key])
