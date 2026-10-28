"""KIara - Ingestion Worker (persistente Job-Queue).

Singleton-Worker-Thread (kiara-admin oder kiara-worker.service).
Pollt ingestion_jobs nach pending Jobs (FIFO) und verarbeitet sie sequentiell.

Steuerung (Cancel/Pause/Resume) laeuft ueber die DB-Spalte
``requested_action`` — prozess-unabhaengig.
"""

import copy
import json
import logging
import os
import threading
import time
from datetime import datetime
from functools import wraps

from kiara.ingestion.datasource_runtime import build_datasource_overrides, load_connector_config
from kiara.ingestion.html_migration import (
    prepare_files as prepare_migration_files,
)
from kiara.ingestion.html_migration import (
    preserve_selection,
    selection_from_stats,
    selection_spec,
    validate_run_selection,
    validate_selection,
)
from kiara.ingestion.html_reingest import request_fields
from kiara.repositories.datasource import DatasourceEmbeddingRepository, DatasourceRepository
from kiara.repositories.job import IngestionJobRepository
from kiara.shared.constants import (
    INGESTION_DATASOURCE_LOCK_NAMESPACE,
    INGESTION_HEARTBEAT_TIMEOUT,
    INGESTION_STARTUP_GRACE,
    INGESTION_WORKER_POLL_INTERVAL,
)
from kiara.shared.db import db_query, db_transaction
from kiara.shared.enums import IngestionStatus
from kiara.web.api_utils import insert_returning, safe_refresh
from kiara.web.ingestion_forensics import request_privileged_ingestion_forensics
from kiara.web.ingestion_stall_detection import (
    CIFSDStateDetector,
    DStateEvent,
    IngestionActivityTracker,
    read_cifs_proc_diagnostics,
    read_python_thread_snapshot,
    read_thread_proc_snapshot,
)

logger = logging.getLogger(__name__)
stall_security_logger = logging.getLogger("kiara.security.ingestion")

POLL_INTERVAL = INGESTION_WORKER_POLL_INTERVAL

# Module-level repo instances
_job_repo = IngestionJobRepository()
_ds_repo = DatasourceRepository()
_emb_repo = DatasourceEmbeddingRepository()


def _guarded_worker_job(operation: str):
    """Fence a worker phase with the immutable job contract snapshot."""

    def decorate(function):
        @wraps(function)
        def wrapped(self, job_id, ds_id, config, embedding_model, *args, **kwargs):
            from kiara.backend_selector import normalize_model_name
            from kiara.retrieval.index_compatibility import (
                IndexCompatibilityError,
                guard_job_contract,
                validate_job_contract_snapshot,
            )

            with db_query() as read_db:
                row = read_db.execute(
                    "SELECT * FROM ingestion_jobs WHERE id = %s",
                    (job_id,),
                ).fetchone()
            if not row:
                raise ValueError(f"Job #{job_id} nicht gefunden")
            with guard_job_contract(
                config,
                dict(row),
                operation=operation,
                write=True,
                allow_initialize=True,
                validation_scope="stored_binding",
            ) as token:
                locked = token.connection.execute(
                    "SELECT * FROM ingestion_jobs WHERE id = %s FOR UPDATE",
                    (job_id,),
                ).fetchone()
                if not locked:
                    raise ValueError(f"Job #{job_id} verschwand waehrend {operation}")
                validate_job_contract_snapshot(
                    dict(locked),
                    token,
                    operation=operation,
                )
                if int(locked["datasource_id"]) != int(
                    ds_id
                ) or token.target.requested_model != normalize_model_name(str(embedding_model)):
                    raise IndexCompatibilityError(
                        "index_job_invocation_mismatch",
                        operation,
                        token.target,
                        (
                            f"Job #{job_id}: Aufruf DS/Modell={ds_id}/{embedding_model!r}, "
                            f"Snapshot={locked['datasource_id']}/{token.target.requested_model!r}"
                        ),
                    )
                return function(
                    self,
                    job_id,
                    ds_id,
                    config,
                    embedding_model,
                    *args,
                    **kwargs,
                )

        return wrapped

    return decorate


def _update_job_stats(db, job_id, **updates):
    """Update stats_json fields in ingestion_jobs (atomares Merge).

    Nutzt PostgreSQL ``||``-Operator fuer atomares JSON-Merge ohne
    Read-Modify-Write Race-Condition (Heartbeat-Thread + Haupt-Thread).
    """
    db.execute(
        "UPDATE ingestion_jobs "
        "SET stats_json = COALESCE(stats_json::jsonb, '{}'::jsonb) || %s::jsonb "
        "WHERE id = %s",
        (json.dumps(updates, ensure_ascii=False), job_id),
    )


# ---------------------------------------------------------------------------
# DB-basierte Steuerung (prozess-unabhaengig)
# ---------------------------------------------------------------------------

_HEARTBEAT_TIMEOUT = INGESTION_HEARTBEAT_TIMEOUT
_STARTUP_GRACE = INGESTION_STARTUP_GRACE
_ACTIVE_JOB_STATUSES = (
    IngestionStatus.PENDING.value,
    IngestionStatus.INITIALIZING.value,
    IngestionStatus.RUNNING.value,
    IngestionStatus.PAUSED.value,
)


def _is_heartbeat_alive(stats_json, started_at: str | None) -> bool:
    """Prueft ob ein Running-Job einen aktiven Heartbeat hat.

    ``stats_json`` ist nach JSONB-Umstellung bereits ein dict (psycopg2
    dekodiert JSONB automatisch); als Fallback fuer Alt-Pfade bleibt
    str-Deserialisierung.

    Returns True wenn:
    - Heartbeat existiert und juenger als _HEARTBEAT_TIMEOUT ist, ODER
    - Kein Heartbeat existiert, aber started_at < _STARTUP_GRACE (frisch gestartet).
    """

    def _age_seconds(dt: datetime) -> float:
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        return (now - dt).total_seconds()

    if isinstance(stats_json, dict):
        stats = stats_json
    else:
        try:
            stats = json.loads(stats_json or "{}")
        except (ValueError, TypeError):
            stats = {}
    hb = stats.get("heartbeat")
    if hb:
        try:
            hb_dt = hb if isinstance(hb, datetime) else datetime.fromisoformat(hb)
            age = _age_seconds(hb_dt)
            return age <= _HEARTBEAT_TIMEOUT
        except (ValueError, TypeError):
            pass  # Fallthrough auf started_at Check
    # Kein Heartbeat: Frisch gestartete Jobs bekommen Grace-Period.
    # PostgreSQL TIMESTAMPTZ kommt via psycopg2 bereits als datetime zurueck.
    if started_at:
        try:
            started_dt = (
                started_at
                if isinstance(started_at, datetime)
                else datetime.fromisoformat(started_at)
            )
            age = _age_seconds(started_dt)
            return age <= _STARTUP_GRACE
        except (ValueError, TypeError):
            pass
    return started_at is None  # Kein started_at (alte Rows) → alive; abgelaufen → Ghost


def request_cancel(job_id: int) -> bool:
    """Setzt requested_action='cancel' in DB fuer laufenden oder pending Job."""
    with db_transaction() as db:
        # Pending: direkt auf cancelled setzen (sofort wirksam, kein Worker noetig)
        result = db.execute(
            "UPDATE ingestion_jobs SET status = %s, "
            "finished_at = NOW() "
            "WHERE id = %s AND status = %s",
            (IngestionStatus.CANCELLED.value, job_id, IngestionStatus.PENDING.value),
        )
        if result.rowcount > 0:
            return True

        row = db.execute(
            "SELECT id, datasource_id, status, stats_json, started_at "
            "FROM ingestion_jobs WHERE id = %s AND status IN (%s, %s, %s) FOR UPDATE",
            (
                job_id,
                IngestionStatus.INITIALIZING.value,
                IngestionStatus.RUNNING.value,
                IngestionStatus.PAUSED.value,
            ),
        ).fetchone()
        if not row:
            return False

        if row["status"] == IngestionStatus.INITIALIZING.value:
            result = db.execute(
                "UPDATE ingestion_jobs SET status = %s, requested_action = NULL, "
                "finished_at = NOW() WHERE id = %s AND status = %s",
                (
                    IngestionStatus.CANCELLED.value,
                    job_id,
                    IngestionStatus.INITIALIZING.value,
                ),
            )
            return result.rowcount > 0

        if not _is_heartbeat_alive(row["stats_json"], row["started_at"]):
            abort_stats = {
                "abort_reason": "Cancel fuer stale running Job ohne aktiven Worker",
            }
            db.execute(
                "UPDATE ingestion_jobs SET status = %s, requested_action = NULL, "
                "finished_at = NOW(), "
                "stats_json = COALESCE(stats_json::jsonb, '{}'::jsonb) || %s::jsonb "
                "WHERE id = %s AND status IN (%s, %s, %s)",
                (
                    IngestionStatus.CANCELLED.value,
                    json.dumps(abort_stats, ensure_ascii=False),
                    job_id,
                    IngestionStatus.INITIALIZING.value,
                    IngestionStatus.RUNNING.value,
                    IngestionStatus.PAUSED.value,
                ),
            )
            db.execute(
                "UPDATE datasources SET ingestion_status = %s WHERE id = %s",
                (IngestionStatus.IDLE.value, row["datasource_id"]),
            )
            return True

        # Running + lebendiger Worker: requested_action setzen, Worker liest es.
        result = db.execute(
            "UPDATE ingestion_jobs SET requested_action = 'cancel' "
            "WHERE id = %s AND status IN (%s, %s, %s)",
            (
                job_id,
                IngestionStatus.INITIALIZING.value,
                IngestionStatus.RUNNING.value,
                IngestionStatus.PAUSED.value,
            ),
        )
        return result.rowcount > 0


def request_pause(job_id: int) -> bool:
    """Setzt requested_action='pause' in DB."""
    with db_transaction() as db:
        result = db.execute(
            "UPDATE ingestion_jobs SET status = %s, requested_action = 'pause' "
            "WHERE id = %s AND status = %s AND requested_action IS NULL",
            (IngestionStatus.PAUSED.value, job_id, IngestionStatus.RUNNING.value),
        )
        return result.rowcount > 0


def request_resume(job_id: int) -> bool:
    """Setzt requested_action=NULL in DB (hebt Pause auf)."""
    with db_transaction() as db:
        result = db.execute(
            "UPDATE ingestion_jobs SET status = %s, requested_action = NULL "
            "WHERE id = %s AND status = %s AND requested_action = 'pause'",
            (IngestionStatus.RUNNING.value, job_id, IngestionStatus.PAUSED.value),
        )
        return result.rowcount > 0


def get_running_job_id() -> int | None:
    """Liest aktuell laufenden Job aus DB (status='running').
    Filtert Ghosts anhand Heartbeat-Alter.
    Fallback: Jobs ohne Heartbeat gelten als real wenn started_at < 60s."""
    with db_query() as db:
        rows = db.execute(
            "SELECT id, stats_json, started_at FROM ingestion_jobs "
            "WHERE status=%s ORDER BY started_at DESC",
            (IngestionStatus.RUNNING.value,),
        ).fetchall()
    for row in rows:
        if _is_heartbeat_alive(row["stats_json"], row["started_at"]):
            return row["id"]
    return None


def is_job_paused(job_id: int) -> bool:
    """Prueft ob requested_action='pause' fuer diesen Job gesetzt ist."""
    with db_query() as db:
        row = db.execute(
            "SELECT status, requested_action FROM ingestion_jobs WHERE id = %s",
            (job_id,),
        ).fetchone()
    return bool(
        row and row["status"] == IngestionStatus.PAUSED.value and row["requested_action"] == "pause"
    )


class IngestionWorker:
    """Sequentieller Ingestion-Worker mit persistenter Queue.

    Verarbeitet immer nur einen Job gleichzeitig (State-Machine).
    Jobs werden in der ingestion_jobs-Tabelle persistiert.

    Lebenszyklus:
        start() → _poll_loop() (Poll-Thread) → _execute_job() pro Job
        + _heartbeat_loop() (separater Liveness-Thread)
        + _stall_watchdog_loop() (nativer D-State-/CIFS-Diagnosethread)

    Steuerung ueber DB-Spalte ``requested_action`` (cancel/pause/resume),
    geprueft via is_cancel_requested() / is_pause_requested().
    """

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._stall_watchdog_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._current_job_id: int | None = None
        self._processing_tid: int | None = None
        self._lock = threading.Lock()
        self._activity_tracker = IngestionActivityTracker()
        self._stall_detector: CIFSDStateDetector | None = None
        self._stall_poll_interval = 5.0
        self._progress_cache = None

    def start(self) -> None:
        """Startet den Worker-Thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._recover_stale_jobs()
        self._configure_stall_detection()
        if self._stall_detector is not None:
            self._stall_watchdog_thread = threading.Thread(
                target=self._stall_watchdog_loop,
                daemon=True,
                name="ingestion-stall-watchdog",
            )
            self._stall_watchdog_thread.start()
        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="ingestion-worker",
        )
        self._thread.start()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="heartbeat",
        )
        self._heartbeat_thread.start()
        logger.info("IngestionWorker gestartet")

    def stop(self) -> None:
        """Graceful Shutdown."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        if self._stall_watchdog_thread:
            self._stall_watchdog_thread.join(timeout=5)
        logger.info("IngestionWorker gestoppt")

    def get_current_job_id(self) -> int | None:
        """Gibt die ID des aktuell laufenden Jobs zurueck."""
        with self._lock:
            return self._current_job_id

    def _set_job_phase(self, job_id, phase):
        """Setzt die aktuelle Verarbeitungsphase im Job (fuer Admin-Anzeige)."""
        tracker = getattr(self, "_activity_tracker", None)
        activity = None
        if tracker is not None:
            activity = tracker.record(job_id, "phase_changed", phase=str(phase))
        try:
            with db_transaction() as db:
                updates = {"phase": phase}
                if activity is not None:
                    updates["ingestion_activity"] = activity.for_persistence()
                _update_job_stats(db, job_id, **updates)
        except Exception as exc:
            logger.debug("Job-Phase-Update fehlgeschlagen: %s", exc)

    def _heartbeat_loop(self):
        """Schreibt alle 30s einen Heartbeat in stats_json.

        Verwendet _update_job_stats() (Read-Modify-Write, backend-agnostic).
        """
        while not self._stop.is_set():
            job_id = self.get_current_job_id()
            if job_id:
                try:
                    updates = {"heartbeat": datetime.now().isoformat()}
                    tracker = getattr(self, "_activity_tracker", None)
                    if tracker is not None:
                        activity = tracker.snapshot()
                        if activity.job_id == job_id:
                            updates["ingestion_activity"] = activity.for_persistence()
                    with db_transaction() as db:
                        _update_job_stats(db, job_id, **updates)
                except Exception as exc:
                    logger.warning("Heartbeat-Update fehlgeschlagen: %s", exc, exc_info=True)
            self._stop.wait(30)

    def _configure_stall_detection(self) -> None:
        """Laedt und initialisiert die rein diagnostische D-State-Erkennung."""
        from kiara.config import get_ingestion_stall_detection_config, load_config

        detection = get_ingestion_stall_detection_config(load_config())
        self._stall_detector = None
        self._stall_poll_interval = float(detection["poll_interval_seconds"])
        if not detection["enabled"]:
            logger.info("Ingestion-D-State-Erkennung ist deaktiviert")
            return
        self._stall_detector = CIFSDStateDetector(
            warning_after_seconds=float(detection["warning_after_seconds"]),
            timeout_seconds=float(detection["timeout_seconds"]),
        )
        logger.info(
            "Ingestion-D-State-Erkennung aktiv (poll=%.1fs, warning=%.1fs, timeout=%.1fs)",
            self._stall_poll_interval,
            float(detection["warning_after_seconds"]),
            float(detection["timeout_seconds"]),
        )

    def _stall_watchdog_loop(self) -> None:
        """Beobachtet den nativen Verarbeitungsthread unabhaengig vom Heartbeat."""
        while not self._stop.is_set():
            try:
                with self._lock:
                    processing_tid = self._processing_tid
                activity = self._activity_tracker.snapshot()
                thread_snapshot = None
                if processing_tid is not None and activity.job_id is not None:
                    thread_snapshot = read_thread_proc_snapshot(os.getpid(), processing_tid)

                detector = self._stall_detector
                if detector is not None:
                    for event in detector.observe(activity, thread_snapshot):
                        self._handle_stall_event(event)
            except Exception as exc:
                # Diagnosecode darf weder Worker noch Watchdog dauerhaft beenden.
                logger.error(
                    "Ingestion-D-State-Watchdog Poll fehlgeschlagen: %s",
                    exc,
                    exc_info=True,
                )
            self._stop.wait(self._stall_poll_interval)

    def _handle_stall_event(self, event: DStateEvent) -> None:
        """Schreibt einen Zustandsuebergang in System-Log und Job-Stats."""
        details = dict(event.details)
        if event.state in ("detected", "confirmed"):
            details["cifs_proc"] = read_cifs_proc_diagnostics()
            thread_details = details.get("thread") or {}
            pid = thread_details.get("pid")
            tid = thread_details.get("tid")
            if isinstance(tid, int):
                details["python_thread"] = read_python_thread_snapshot(tid)
            if isinstance(pid, int) and isinstance(tid, int):
                details["privileged_forensics"] = request_privileged_ingestion_forensics(pid, tid)

        activity = details.get("activity") or {}
        thread_details = details.get("thread") or {}
        message = (
            f"{event.event_type} "
            f"job_id={activity.get('job_id')} "
            f"datasource_id={activity.get('datasource_id')} "
            f"pid={thread_details.get('pid')} "
            f"tid={thread_details.get('tid')} "
            f"duration_seconds={event.duration_seconds:.3f} "
            f"reason={event.reason} mode=diagnostic_only"
        )
        level = getattr(logging, event.level, logging.INFO)
        stall_security_logger.log(level, message, extra={"kiara_details": details})
        logger.log(level, message)
        self._persist_stall_event(event, details)

    @staticmethod
    def _persist_stall_event(event: DStateEvent, details: dict) -> None:
        """Persistiert einen kompakten aktuellen Zustand am Job-Datensatz."""
        activity = details.get("activity") or {}
        job_id = activity.get("job_id")
        if job_id is None:
            return
        thread_details = details.get("thread") or {}
        compact = {
            "state": event.state,
            "event_type": event.event_type,
            "reason": event.reason,
            "duration_seconds": event.duration_seconds,
            "observed_at": details.get("observed_at"),
            "d_state_first_seen_at": details.get("d_state_first_seen_at"),
            "classification": details.get("classification"),
            "pid": thread_details.get("pid"),
            "tid": thread_details.get("tid"),
            "thread_state": thread_details.get("state"),
            "wchan": thread_details.get("wchan"),
            "current_file": activity.get("file_name"),
            "current_file_path": activity.get("file_path"),
            "phase": activity.get("phase"),
            "activity_sequence": activity.get("activity_sequence"),
            "chunks_created_current_file": activity.get("chunks_created_current_file"),
            "chunks_written_current_file": activity.get("chunks_written_current_file"),
            "automatic_recovery": False,
        }
        try:
            with db_transaction() as db:
                _update_job_stats(db, int(job_id), stall_detection=compact)
        except Exception as exc:
            logger.warning(
                "Job %s: D-State-Diagnosestatus konnte nicht persistiert werden: %s",
                job_id,
                exc,
                exc_info=True,
            )

    def _recover_stale_jobs(self):
        """Beim Start: 'running' Jobs als 'error' markieren und Auto-Resume einreihen.

        Wichtig: Error-Markierung und Auto-Resume laufen in getrennten
        Transaktionen, damit ein Fehler beim Resume nicht die Bereinigung
        der verwaisten Jobs zurueckrollt.
        """
        # Phase 1: Verwaiste Jobs einsammeln und als error markieren
        stale_rows = self._identify_stale_jobs()
        if not stale_rows:
            return

        # Phase 2: Auto-Resume (separate Transaktion)
        try:
            resumed = 0
            for srow in stale_rows:
                if self._recover_single_job(None, srow):
                    resumed += 1

            if resumed:
                logger.info(
                    "%s unterbrochene Jobs werden automatisch fortgesetzt",
                    resumed,
                )
        except Exception as exc:
            logger.error(
                "Stale-Job-Recovery (Auto-Resume) fehlgeschlagen: %s",
                exc,
                exc_info=True,
            )

    def _identify_stale_jobs(self):
        """Findet verwaiste 'running' Jobs und markiert sie als 'error'.

        Laeuft in einer eigenen Transaktion. Setzt alle laufenden Jobs auf
        error-Status, die zugehoerigen Datasources auf idle und zaehlt die
        waehrend des abgebrochenen Jobs bereits persistierten Chunk-/Datei-
        Metadaten aus den Source-of-Truth-Tabellen neu. So zeigt
        ``datasource_embeddings`` auch vor dem Auto-Resume den realen Stand.

        Returns:
            list: Die gefundenen stale-Job-Rows (leer wenn keine vorhanden).
        """
        stale_rows = []
        try:
            with db_transaction() as db:
                stale_rows = db.execute(
                    "SELECT * FROM ingestion_jobs WHERE status IN (%s, %s) FOR UPDATE SKIP LOCKED",
                    (
                        IngestionStatus.INITIALIZING.value,
                        IngestionStatus.RUNNING.value,
                    ),
                ).fetchall()

                if not stale_rows:
                    return []

                db.execute(
                    "UPDATE ingestion_jobs SET status = CASE "
                    "WHEN requested_action = 'cancel' THEN %s "
                    "ELSE %s END, "
                    "error_message = CASE "
                    "WHEN requested_action = 'cancel' THEN NULL "
                    "ELSE 'Abgebrochen durch Service-Restart' END, "
                    "requested_action = NULL, "
                    "finished_at = NOW() "
                    "WHERE status IN (%s, %s)",
                    (
                        IngestionStatus.CANCELLED.value,
                        IngestionStatus.ERROR.value,
                        IngestionStatus.INITIALIZING.value,
                        IngestionStatus.RUNNING.value,
                    ),
                )
                db.execute(
                    "UPDATE datasources SET ingestion_status = %s WHERE ingestion_status = %s",
                    (IngestionStatus.IDLE.value, IngestionStatus.RUNNING.value),
                )
                recounted = self._recount_stale_job_stats(db, stale_rows)
            cancelled = sum(1 for r in stale_rows if r["requested_action"] == "cancel")
            logger.warning(
                "Restart-Recovery: %s verwaiste Jobs bereinigt "
                "(%s cancelled, %s error, %s DS-/Modell-Zaehler aktualisiert)",
                len(stale_rows),
                cancelled,
                len(stale_rows) - cancelled,
                recounted,
            )
        except Exception as exc:
            logger.error("Stale-Job-Recovery (Phase 1) fehlgeschlagen: %s", exc, exc_info=True)
            return []
        return stale_rows

    @staticmethod
    def _recount_stale_job_stats(db, stale_rows) -> int:
        """Zaehlt partielle Ergebnisse abgebrochener Jobs atomar neu."""
        from kiara.backend_selector import normalize_model_name
        from kiara.ingestion.vectordb_ops import _recount_datasource_embeddings

        affected: set[tuple[int, str]] = set()
        for row in stale_rows:
            model = normalize_model_name(str(row.get("embedding_model") or ""))
            if model:
                affected.add((int(row["datasource_id"]), model))

        for datasource_id, embedding_model in sorted(affected):
            _recount_datasource_embeddings(db, datasource_id, embedding_model)
        return len(affected)

    def _recover_single_job(self, _unused_db, srow, *, manual_html_retry=False):
        """Versucht einen einzelnen stale Job per Auto-Resume neu einzureihen.

        Prueft Cancel-Request, Embedding-Modell-Verfuegbarkeit und ob bereits
        ein Pending-Job fuer die Datasource existiert. Erstellt bei Bedarf
        einen neuen incremental Job.

        Args:
            _unused_db: Historischer Signaturplatzhalter; S4 nutzt die vom
                Contract-Guard gefencete Transaktion.
            srow: Row-Dict des stale Jobs (id, datasource_id, embedding_model,
                  requested_action, job_type).

        Returns:
            bool: True wenn ein Resume-Job erstellt wurde, False sonst.
        """
        old_job_id = srow["id"]
        ds_id = srow["datasource_id"]
        emb_model = srow["embedding_model"] or ""

        # V29-FIX: Cancel-Request respektieren
        if not manual_html_retry and srow["requested_action"] == "cancel":
            logger.info(
                "DS %s: Stale Job #%s hatte Cancel-Request — kein Auto-Resume",
                ds_id,
                old_job_id,
            )
            return False

        if not manual_html_retry and srow["requested_action"] == "pause":
            logger.info(
                "DS %s: Stale Job #%s hatte Pause-Request — kein Auto-Resume",
                ds_id,
                old_job_id,
            )
            return False

        if not emb_model:
            logger.warning(
                "DS %s: Stale Job #%s hat kein Embedding-Modell — "
                "markiere als error (kein Auto-Resume moeglich)",
                ds_id,
                old_job_id,
            )
            return False

        # Normalisierung fuer Altdaten
        from kiara.backend_selector import normalize_model_name

        emb_model = normalize_model_name(emb_model)
        if not emb_model:
            logger.warning(
                "DS %s: Stale Job #%s hat nach Normalisierung kein Embedding-Modell — "
                "markiere als error",
                ds_id,
                old_job_id,
            )
            return False

        from kiara.config import load_config
        from kiara.retrieval.index_compatibility import (
            IndexCompatibilityError,
            IndexOperation,
            guard_job_contract,
            validate_job_contract_snapshot,
        )

        try:
            # Auch eine Wiederholung bleibt ein Bestandsauftrag. Eine fehlende
            # Collection darf nicht als erlaubter Erstaufbau behandelt werden.
            with db_query() as read_db:
                retry_row = read_db.execute(
                    "SELECT stats_json FROM ingestion_jobs WHERE id = %s", (old_job_id,)
                ).fetchone()
            html_retry = bool(request_fields(retry_row["stats_json"])) if retry_row else False
            with guard_job_contract(
                load_config(),
                dict(srow),
                operation=IndexOperation.JOB_ENQUEUE,
                write=True,
                allow_initialize=not html_retry,
                validation_scope="stored_binding",
            ) as token:
                db = token.connection
                locked_old = db.execute(
                    "SELECT * FROM ingestion_jobs WHERE id = %s FOR UPDATE",
                    (old_job_id,),
                ).fetchone()
                if not locked_old:
                    return False
                validate_job_contract_snapshot(
                    dict(locked_old),
                    token,
                    operation=IndexOperation.JOB_ENQUEUE,
                )
                if not manual_html_retry and locked_old.get("requested_action") in (
                    "cancel",
                    "pause",
                ):
                    return False
                # Gleiche Lock-Reihenfolge wie enqueue_job(): Contract zuerst,
                # danach Datasource.
                db.execute(
                    "SELECT pg_advisory_xact_lock(%s, %s)",
                    (INGESTION_DATASOURCE_LOCK_NAMESPACE, ds_id),
                )
                statuses = (
                    _ACTIVE_JOB_STATUSES if manual_html_retry else (IngestionStatus.PENDING.value,)
                )
                existing = db.execute(
                    "SELECT id FROM ingestion_jobs WHERE datasource_id = %s AND status = ANY(%s)",
                    (ds_id, list(statuses)),
                ).fetchone()
                if existing:
                    if manual_html_retry:
                        raise ValueError(
                            f"Datenquelle hat bereits einen aktiven Job (#{existing['id']})"
                        )
                    logger.warning(
                        "DS %s: Pending Job #%s existiert bereits — kein Auto-Resume",
                        ds_id,
                        existing["id"],
                    )
                    return False

                orig_job_type = srow.get("job_type") or "incremental"
                migration = selection_from_stats(locked_old.get("stats_json"), ds_id, emb_model)
                html_fields = request_fields(locked_old.get("stats_json"))
                if manual_html_retry:
                    from kiara.ingestion.html_reingest import validate_request, validate_target

                    if not html_fields or locked_old["status"] not in {"error", "cancelled"}:
                        raise ValueError(
                            "Nur fehlgeschlagene oder abgebrochene HTML-Jobs sind wiederholbar"
                        )
                    validate_request(html_fields["html_reingest"], check_current=True)
                    ds = db.execute(
                        "SELECT * FROM datasources WHERE id = %s FOR UPDATE", (ds_id,)
                    ).fetchone()
                    if not ds or not ds["is_active"]:
                        raise ValueError("Datenquelle fehlt oder ist deaktiviert")
                    validate_target(dict(ds), emb_model)
                if (migration is not None or html_fields) and orig_job_type != "incremental":
                    raise ValueError("Migrationsjobs muessen incremental bleiben")
                resume_stats = json.dumps(
                    {
                        "resumed_from_job": old_job_id,
                        **html_fields,
                        **({"html_migration": migration} if migration is not None else {}),
                    },
                    ensure_ascii=False,
                )
                contract = token.job_fields()
                new_job = db.execute(
                    "INSERT INTO ingestion_jobs "
                    "(datasource_id, job_type, triggered_by, embedding_model, stats_json, "
                    "resumed_from_job, queued_at, contract_logical_index_key, "
                    "contract_index_compatibility_id, contract_query_compatibility_id, "
                    "contract_canonical_model_id, contract_profile_id, contract_profile_kind, "
                    "contract_profile_endpoint, contract_snapshot_sha256, contract_fence_version) "
                    "VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, "
                    "%s, %s, %s, %s, %s) RETURNING id",
                    (
                        ds_id,
                        orig_job_type,
                        "manual" if manual_html_retry else "auto-resume",
                        emb_model,
                        resume_stats,
                        old_job_id,
                        contract["contract_logical_index_key"],
                        contract["contract_index_compatibility_id"],
                        contract["contract_query_compatibility_id"],
                        contract["contract_canonical_model_id"],
                        contract["contract_profile_id"],
                        contract["contract_profile_kind"],
                        contract["contract_profile_endpoint"],
                        contract["contract_snapshot_sha256"],
                        contract["contract_fence_version"],
                    ),
                )
                return new_job.fetchone()["id"] if manual_html_retry else True
        except IndexCompatibilityError as exc:
            if manual_html_retry:
                raise ValueError(f"HTML-Wiederholung am Vertragsguard blockiert: {exc}") from exc
            logger.error(
                "DS %s: Auto-Resume von Job #%s am Contract-Guard blockiert: %s",
                ds_id,
                old_job_id,
                exc,
            )
            return False

    def _poll_loop(self):
        """Hauptschleife: Pollt Queue alle POLL_INTERVAL Sekunden."""
        while not self._stop.is_set():
            try:
                job_row = self._fetch_next_job()
                if job_row:
                    self._execute_job(job_row)
                else:
                    self._stop.wait(POLL_INTERVAL)
            except (OSError, ConnectionError) as exc:
                logger.error("Worker Poll-Loop transienter Fehler: %s", exc)
                self._stop.wait(POLL_INTERVAL)
            except Exception as exc:
                import psycopg2

                if isinstance(exc, psycopg2.OperationalError):
                    logger.error("Worker Poll-Loop DB-Fehler: %s", exc)
                    self._stop.wait(POLL_INTERVAL)
                else:
                    logger.critical(
                        "Worker Poll-Loop Code-Fehler — Worker stoppt: %s", exc, exc_info=True
                    )
                    self._stop.set()
                    break

    def _fetch_next_job(self):
        """Holt den naechsten pending Job und claimt ihn atomar."""
        from kiara.config import load_config
        from kiara.retrieval.index_compatibility import (
            IndexCompatibilityError,
            IndexContractUnavailableError,
            IndexOperation,
            guard_job_contract,
            validate_job_contract_snapshot,
        )

        with db_query() as db:
            candidate = db.execute(
                "SELECT j.* FROM ingestion_jobs j "
                "JOIN datasources d ON d.id = j.datasource_id "
                "WHERE j.status = 'pending' AND d.is_active = true "
                "ORDER BY j.queued_at ASC, j.id ASC LIMIT 1"
            ).fetchone()
        if not candidate:
            return None
        candidate = dict(candidate)
        config = load_config()
        try:
            with guard_job_contract(
                config,
                candidate,
                operation=IndexOperation.JOB_CLAIM,
                write=True,
                allow_initialize=True,
            ) as token:
                db = token.connection
                row = db.execute(
                    "SELECT j.* FROM ingestion_jobs j "
                    "JOIN datasources d ON d.id = j.datasource_id "
                    "WHERE j.id = %s AND j.status = 'pending' AND d.is_active = true "
                    "FOR UPDATE OF j SKIP LOCKED",
                    (candidate["id"],),
                ).fetchone()
                if not row:
                    return None
                current = dict(row)
                validate_job_contract_snapshot(
                    current,
                    token,
                    operation=IndexOperation.JOB_CLAIM,
                )
                db.execute(
                    "UPDATE ingestion_jobs SET status = 'initializing', started_at = NOW() "
                    "WHERE id = %s",
                    (candidate["id"],),
                )
                current["status"] = IngestionStatus.INITIALIZING.value
                return current
        except IndexContractUnavailableError as exc:
            # Reine Index-Abhaengigkeitsstoerung ist kein Vertragsbruch.
            # Der Job bleibt pending und wird vom naechsten Poll erneut
            # atomar geclaimt; kein semantischer Providerkontakt findet statt.
            logger.warning(
                "Job %s wartet pending auf Index-Abhaengigkeit: %s",
                candidate["id"],
                exc.detail,
            )
            return None
        except IndexCompatibilityError as exc:
            logger.error(
                "Job %s am Contract-Claim blockiert: %s",
                candidate["id"],
                exc,
                exc_info=True,
            )
            with db_transaction() as db:
                db.execute(
                    "UPDATE ingestion_jobs SET status = %s, finished_at = NOW(), "
                    "error_message = %s, stats_json = COALESCE(stats_json, '{}'::jsonb) || %s::jsonb "
                    "WHERE id = %s AND status = %s",
                    (
                        IngestionStatus.ERROR.value,
                        str(exc),
                        json.dumps(exc.to_dict(), ensure_ascii=False),
                        candidate["id"],
                        IngestionStatus.PENDING.value,
                    ),
                )
            return None

    def _mark_job_running(self, row: dict, config: dict) -> bool:
        """Wechselt einen initialisierten Job atomar in den sichtbaren Running-Status."""
        from kiara.retrieval.index_compatibility import (
            IndexOperation,
            guard_job_contract,
            validate_job_contract_snapshot,
        )

        job_id = row["id"]
        ds_id = row["datasource_id"]
        with guard_job_contract(
            config,
            row,
            operation=IndexOperation.JOB_START,
            write=True,
            allow_initialize=config.get("_html_reingest") is None,
        ) as token:
            db = token.connection
            locked = db.execute(
                "SELECT * FROM ingestion_jobs WHERE id = %s FOR UPDATE",
                (job_id,),
            ).fetchone()
            if not locked:
                return False
            validate_job_contract_snapshot(
                dict(locked),
                token,
                operation=IndexOperation.JOB_START,
            )
            result = db.execute(
                "UPDATE ingestion_jobs SET status = %s "
                "WHERE id = %s AND status = %s AND requested_action IS NULL",
                (
                    IngestionStatus.RUNNING.value,
                    job_id,
                    IngestionStatus.INITIALIZING.value,
                ),
            )
            if result.rowcount <= 0:
                return False
            db.execute(
                "UPDATE datasources SET ingestion_status = %s WHERE id = %s",
                (IngestionStatus.RUNNING.value, ds_id),
            )
        return True

    # ------------------------------------------------------------------
    # _execute_job — Orchestrator + extrahierte Phasen-Methoden
    # ------------------------------------------------------------------

    def _execute_job(self, row):
        """Fuehrt einen einzelnen Ingestion-Job aus (Orchestrator).

        Delegiert an Phasen-Methoden; haelt nur noch den aeusseren
        try/except/finally und die Ablauf-Entscheidungen.
        """
        job_id = row["id"]
        ds_id = row["datasource_id"]
        job_type = row["job_type"]
        embedding_model = row["embedding_model"] or None

        with self._lock:
            self._current_job_id = job_id
            self._processing_tid = threading.get_native_id()

        start_time = time.time()
        _check_control = self._build_check_control(job_id)
        config = None  # fuer except-Block (locals-Fallback)
        stats = None

        try:
            if not embedding_model:
                raise ValueError(
                    f"Job #{job_id} hat kein Embedding-Modell. "
                    "Jeder Ingestion-Job muss ein Embedding-Modell spezifizieren. "
                    "Dieser Job wurde vermutlich vor der Multi-Embedding-Aenderung erstellt."
                )
            # Phase 1: Job-Kontext laden (DS, Config, Connector)
            ctx = self._load_job_context(row, ds_id, job_id)
            config = ctx["config"]
            from kiara.web.cache_refresh import IngestionCacheRefresh

            self._progress_cache = IngestionCacheRefresh(
                ds_id,
                job_id,
                interval=config.get("ingestion", {}).get("admin_cache_refresh_seconds", 3),
            )
            connector_cls = ctx["connector_cls"]
            conn_config = ctx["conn_config"]
            ds_row = ctx["ds_row"]
            tracker = getattr(self, "_activity_tracker", None)
            if tracker is not None:
                tracker.begin_job(job_id, ds_id, ds_row["type"])

            from kiara.ingestion.runtime import _wait_for_index_contract_target_recovery
            from kiara.retrieval.index_compatibility import IndexContractUnavailableError

            contract_start_stats: dict[str, int] = {}
            stats = contract_start_stats
            while True:
                try:
                    marked_running = self._mark_job_running(row, config)
                    break
                except IndexContractUnavailableError as exc:
                    recovered = _wait_for_index_contract_target_recovery(
                        config,
                        exc,
                        control_callback=_check_control,
                        status_callback=lambda phase: self._set_job_phase(job_id, phase),
                        stats=contract_start_stats,
                    )
                    if recovered:
                        continue
                    signal = _check_control()
                    if signal == "cancel":
                        self._finalize_cancelled(job_id, ds_id, contract_start_stats)
                    return

            if not marked_running:
                logger.info(
                    "Job %s wurde vor dem Running-Status abgebrochen oder veraendert",
                    job_id,
                )
                return

            # Phase 2: Callbacks aufbauen
            callbacks = self._build_callbacks(job_id, ds_id)
            _on_progress = callbacks["on_progress"]

            # Phase 3: Backend-Warmup
            warmup_stats_dict = self._run_warmup(
                job_id,
                ds_id,
                ds_row,
                embedding_model,
                config,
                start_time,
                _check_control,
            )
            if warmup_stats_dict is None:
                return  # Warmup fehlgeschlagen oder abgebrochen

            # Phase 4: Scan (parallel oder sequentiell)
            scan_info = self._run_scan_phase(
                job_id,
                ds_id,
                job_type,
                config,
                connector_cls,
                conn_config,
                ds_row,
                _on_progress,
                _check_control,
                embedding_model=embedding_model,
            )
            if scan_info is None:
                return  # Abgebrochen oder fehlgeschlagen waehrend Scan

            # Phase 5: Full-Reingest-Cleanup (falls noetig)
            if job_type == "full":
                self._prepare_full_reingest(job_id, ds_id, config, embedding_model)

            # Phase 6: Resume-Skip ermitteln
            resume_skip_files = self._load_resume_skip_files(
                job_id,
                row["resumed_from_job"],
                job_type,
            )

            # Phase 7: Ingestion ausfuehren (Pass 1)
            overrides = build_datasource_overrides(ds_row) or None
            hash_mode = getattr(connector_cls, "HASH_MODE", "mtime")

            stats, flush_fn = self._run_ingestion(
                job_id,
                ds_id,
                job_type,
                config,
                overrides,
                embedding_model,
                _on_progress,
                _check_control,
                callbacks,
                resume_skip_files,
                hash_mode,
                warmup_stats_dict,
            )
            for key in (
                "contract_discovery_waits",
                "contract_discovery_recoveries",
                "contract_discovery_wait_seconds",
            ):
                stats[key] = stats.get(key, 0) + contract_start_stats.get(key, 0)

            # Phase 8: Parallel-Scan Pass 2 (falls noetig)
            if scan_info.get("scan_thread") is not None:
                self._run_pass2(
                    job_id,
                    ds_id,
                    config,
                    overrides,
                    embedding_model,
                    _on_progress,
                    _check_control,
                    callbacks,
                    hash_mode,
                    warmup_stats_dict,
                    stats,
                    flush_fn,
                    scan_info,
                )

            # Phase 9: Ergebnis auswerten und finalisieren
            duration = int(time.time() - start_time)
            stats["duration_seconds"] = duration
            self._resolve_job_outcome(
                job_id,
                ds_id,
                config,
                embedding_model,
                stats,
                duration,
                _check_control,
            )

        except Exception as exc:
            # Shutdown-Check auch bei unerwarteten Exceptions.
            try:
                self._close_progress_cache()
            except Exception as cache_exc:
                logger.warning("Cache-Abschluss fehlgeschlagen: %s", cache_exc, exc_info=True)
            if self._stop.is_set():
                logger.info("Job %s: Shutdown waehrend Fehler — wird auto-resumed", job_id)
                return
            from kiara.retrieval.index_compatibility import IndexCompatibilityError

            error_message = str(exc)
            secondary_errors = getattr(exc, "ingestion_secondary_errors", ())
            if secondary_errors:
                stats = dict(stats or {})
                stats.setdefault("error_details", []).extend(secondary_errors)
            if isinstance(exc, IndexCompatibilityError):
                diagnostic = exc.to_dict()
                stats = dict(stats or {})
                if stats.get("index_compatibility_error") != diagnostic:
                    stats["errors"] = stats.get("errors", 0) + 1
                    stats.setdefault("error_details", []).append(
                        {
                            "phase": "index_compatibility",
                            "reason_code": exc.reason_code,
                            "operation": exc.operation,
                            "consumer": "ingestion_worker",
                            "diagnostic": diagnostic,
                        }
                    )
                stats["index_compatibility_error"] = diagnostic
                stats["abort_reason"] = (
                    f"Index-Kompatibilitaetsguard: {exc.reason_code} ({exc.detail})"
                )
                error_message = stats["abort_reason"]
            # Worker-Fail-Closed: Job auf FAILED, naechster Job kann laufen.
            # exc_info=True liefert den Traceback fuer Audit, ohne den Loop
            # zu brechen.
            logger.error(
                "Job %s (DS %s): unerwarteter Fehler — Job wird auf FAILED gesetzt: %s",
                job_id,
                ds_id,
                error_message,
                exc_info=True,
            )
            duration = int(time.time() - start_time)
            self._finalize_error(
                job_id,
                ds_id,
                error_message,
                duration,
                config=config,
                embedding_model=embedding_model,
                stats=stats,
            )

        finally:
            try:
                self._close_progress_cache()
            finally:
                tracker = getattr(self, "_activity_tracker", None)
                if tracker is not None:
                    tracker.finish_job(job_id)
                with self._lock:
                    self._current_job_id = None
                    self._processing_tid = None

    # --- Phase: Control-Callback ---

    def _build_check_control(self, job_id):
        """Erzeugt die Control-Callback-Closure (Cancel/Pause/Resume/Shutdown).

        Gibt eine Funktion zurueck, die None | 'cancel' | 'shutdown' liefert.
        Bei 'pause' blockiert sie bis Resume oder Cancel.
        """

        def _check_control():
            # Shutdown-Check ZUERST — vor DB-Read, damit SIGTERM
            # sofort wirkt und nicht erst auf DB-Antwort wartet.
            if self._stop.is_set():
                return "shutdown"
            try:
                # Nur lesen, nicht sperren: Der Callback wird auch innerhalb
                # des Index-Contract-Guards aufgerufen. Dieser haelt den
                # ingestion_jobs-Datensatz bereits per FOR UPDATE. Ein
                # weiterer FOR-UPDATE-Read ueber eine zweite Verbindung
                # wuerde deshalb bis zum statement_timeout auf den eigenen
                # Worker warten. Pause/Cancel-Writer serialisieren ihre
                # Updates weiterhin in ihren eigenen Transaktionen.
                with db_query() as db:
                    ctrl_row = db.execute(
                        "SELECT requested_action FROM ingestion_jobs WHERE id = %s",
                        (job_id,),
                    ).fetchone()
                    action = ctrl_row["requested_action"] if ctrl_row else None
            except Exception:
                # DB-Fehler → Job laeuft weiter; Audit-Spur fuer Diagnose.
                logger.debug(
                    "Job %s: Control-Poll DB-Fehler — Job laeuft weiter",
                    job_id,
                    exc_info=True,
                )
                return None
            if action == "cancel":
                return "cancel"
            pause_poll_interval = 1.0
            while action == "pause":
                if self._stop.is_set():
                    return "shutdown"
                if self._stop.wait(pause_poll_interval):
                    return "shutdown"
                # Schrittweise erhoehen: 1s, 1s, 2s, 2s, 3s, ... bis max 5s
                if pause_poll_interval < 5.0:
                    pause_poll_interval = min(pause_poll_interval + 0.5, 5.0)
                try:
                    with db_query() as db:
                        ctrl_row = db.execute(
                            "SELECT requested_action FROM ingestion_jobs WHERE id = %s",
                            (job_id,),
                        ).fetchone()
                        action = ctrl_row["requested_action"] if ctrl_row else None
                except Exception:
                    # Bei DB-Fehler waehrend Pause NICHT entpausieren.
                    logger.debug(
                        "Job %s: Pause-Poll DB-Fehler — bleibe pausiert",
                        job_id,
                        exc_info=True,
                    )
                    continue
                if action == "cancel":
                    return "cancel"
                if action != "pause":
                    break  # Resume oder NULL → weiter
            return None

        return _check_control

    # --- Phase 1: Job-Kontext laden ---

    def _load_job_context(self, row, ds_id, job_id):
        """Laedt Datasource, Config und Connector-Klasse.

        Returns:
            dict mit keys: ds_row, config, connector_cls, conn_config
        """
        from kiara.config import load_config
        from kiara.connectors.registry import ensure_loaded
        from kiara.connectors.registry import get as get_connector

        ds_row = _ds_repo.get_by_id(ds_id)
        if not ds_row:
            raise ValueError(f"Datenquelle {ds_id} nicht gefunden")

        ensure_loaded()
        config = copy.deepcopy(load_config())
        # Marker: persistierte Config — ColBERT Model-Mismatch-Guard aktiv
        config["_kiara_enforce_persisted_colbert_model"] = True
        connector_cls = get_connector(ds_row["type"])
        if not connector_cls:
            raise ValueError(f"Unbekannter Datenquellen-Typ: {ds_row['type']}")

        conn_config = load_connector_config(ds_row, connector_cls)

        migration = selection_from_stats(row.get("stats_json"), ds_id, row["embedding_model"])
        html_fields = request_fields(row.get("stats_json"))
        if html_fields:
            from kiara.ingestion.html_reingest import validate_request, validate_target

            if row["job_type"] != "incremental":
                raise ValueError("HTML-Neuaufbereitung muss incremental bleiben")
            validate_target(dict(ds_row), row["embedding_model"])
            config["_html_reingest"] = validate_request(
                html_fields["html_reingest"], check_current=True
            )
            if "html_reingest_candidates" in html_fields:
                config["_html_reingest_candidates"] = html_fields["html_reingest_candidates"]
            ds_row = dict(ds_row)
            ds_row["ds_colbert_enabled"] = False
            config.setdefault("retrieval", {}).setdefault("colbert", {})["enabled"] = False
        if migration is not None:
            if row["job_type"] != "incremental":
                raise ValueError("Migrationsjobs muessen incremental bleiben")
            from kiara.backend_selector import normalize_model_name

            model = normalize_model_name(row["embedding_model"])
            config["_html_migration"] = {
                **migration,
                "paths": list(migration["files_by_model"][model]),
                "errors": {},
            }
            # Ausschluss gilt bereits beim Warmup, nur fuer diesen Lauf.
            ds_row = dict(ds_row)
            ds_row["ds_colbert_enabled"] = False
            config.setdefault("retrieval", {}).setdefault("colbert", {})["enabled"] = False

        # Manuell ausgeloeste Jobs: Scan-Cache invalidieren,
        # damit Mailbox-Konnektoren einen frischen IMAP-Scan durchfuehren.
        # Nur fuer Konnektoren mit aktivem Scan-Cache (IMAP/Exchange) — bei
        # Datei-Konnektoren (SMB/Local/WebDAV) zeigt effective_path() in den
        # Mountpoint, der dem kiara-User u.U. nicht beschreibbar ist.
        triggered_by = row["triggered_by"]
        if (
            migration is None
            and not html_fields
            and triggered_by == "manual"
            and getattr(connector_cls, "SCAN_CACHE_MAX_AGE", 0) > 0
        ):
            _marker = os.path.join(
                connector_cls.effective_path(conn_config, dict(ds_row)),
                ".scan_done",
            )
            try:
                os.remove(_marker)
                logger.info("Job %s: Scan-Cache invalidiert (manueller Trigger)", job_id)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning(
                    "Job %s: Scan-Cache-Marker %s konnte nicht entfernt werden: %s",
                    job_id,
                    _marker,
                    exc,
                )

        return {
            "ds_row": ds_row,
            "config": config,
            "connector_cls": connector_cls,
            "conn_config": conn_config,
        }

    # --- Phase 2: Callbacks aufbauen ---

    def _close_progress_cache(self):
        cache = self._progress_cache
        self._progress_cache = None
        if cache is not None:
            cache.close()

    def _build_callbacks(self, job_id, ds_id):
        """Erzeugt Progress-, File-Log-, File-Start- und Status-Callbacks.

        Returns:
            dict mit keys: on_progress, on_file_log, flush_file_logs,
                           on_file_start, on_status
        """

        def _on_progress(processed, total):
            tracker = getattr(self, "_activity_tracker", None)
            if tracker is not None:
                tracker.record(job_id, "file_progress")
            try:
                with db_transaction() as db:
                    db.execute(
                        "UPDATE ingestion_jobs SET progress_processed = %s, progress_total = %s "
                        "WHERE id = %s",
                        (processed, total, job_id),
                    )
                    db.execute(
                        "UPDATE datasources SET processed_files = %s, progress_total = %s "
                        "WHERE id = %s",
                        (processed, total, ds_id),
                    )
            except Exception as exc:
                logger.debug("Progress-Update fehlgeschlagen: %s", exc)
            if self._progress_cache is not None:
                self._progress_cache.request()

        _file_log_buffer = []
        _FILE_LOG_FLUSH = 1

        def _flush_file_logs():
            if not _file_log_buffer:
                return
            try:
                with db_transaction() as _db:
                    _db.executemany(
                        "INSERT INTO ingestion_file_logs "
                        "(job_id, file_path, file_name, file_size, file_ext, "
                        "parser, parser_fallback, chunk_strategy, chunk_count, "
                        "status, error_message, error_phase, duration_ms) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        [
                            (
                                job_id,
                                e.file_path,
                                e.file_name,
                                e.file_size,
                                e.file_ext,
                                e.parser,
                                int(e.parser_fallback),
                                e.chunk_strategy,
                                e.chunk_count,
                                e.status,
                                e.error_message,
                                e.error_phase,
                                e.duration_ms,
                            )
                            for e in _file_log_buffer
                        ],
                    )
            except Exception as exc:
                logger.warning("File-Log flush fehlgeschlagen: %s", exc, exc_info=True)
            _file_log_buffer.clear()

        def _on_file_log(entry):
            _file_log_buffer.append(entry)
            if len(_file_log_buffer) >= _FILE_LOG_FLUSH:
                _flush_file_logs()

        def _on_file_start(file_name, file_path):
            updates = {"current_file": file_name, "phase": "Verarbeite..."}
            tracker = getattr(self, "_activity_tracker", None)
            if tracker is not None:
                activity = tracker.snapshot()
                if activity.job_id == job_id:
                    updates["ingestion_activity"] = activity.for_persistence()
            try:
                with db_transaction() as _db:
                    _update_job_stats(_db, job_id, **updates)
            except Exception as exc:
                logger.debug("Job %s: Datei-Start-Update fehlgeschlagen: %s", job_id, exc)

        def _on_status(phase):
            self._set_job_phase(job_id, phase)

        def _on_activity(event, **details):
            tracker = getattr(self, "_activity_tracker", None)
            if tracker is None:
                return
            tracker.record(
                job_id,
                event,
                phase=details.get("phase"),
                file_name=details.get("file_name"),
                file_path=details.get("file_path"),
                chunks_created=details.get("chunks_created"),
                chunks_written_delta=details.get("chunks_written_delta", 0),
            )

        return {
            "on_progress": _on_progress,
            "on_file_log": _on_file_log,
            "flush_file_logs": _flush_file_logs,
            "on_file_start": _on_file_start,
            "on_status": _on_status,
            "on_activity": _on_activity,
        }

    # --- Phase 3: Backend-Warmup ---

    def _run_warmup(
        self, job_id, ds_id, ds_row, embedding_model, config, start_time, check_control
    ):
        """Fuehrt Backend-Warmup durch.

        Returns:
            warmup_stats_dict bei Erfolg, None bei Fehler/Abbruch
            (Fehler-/Cancel-Finalisierung wird intern erledigt).
        """
        from kiara.backend_warmup import BackendWarmer, build_warmup_tasks
        from kiara.shared.constants import WARMUP_TIMEOUT_DEFAULT

        self._set_job_phase(job_id, "Backend-Warmup...")

        try:
            warmup_tasks = build_warmup_tasks(ds_row, embedding_model, config)
        except RuntimeError as exc:
            duration = int(time.time() - start_time)
            self._finalize_error(job_id, ds_id, str(exc), duration)
            return None

        warmup_result = BackendWarmer().warm(
            warmup_tasks,
            timeout=WARMUP_TIMEOUT_DEFAULT,
            check_control=check_control,
        )

        # Warmup-Ergebnis in stats_json persistieren (fuer Admin-UI / Debugging)
        self._set_warmup_stats(job_id, warmup_result)

        # Cancel-Check ZUERST (warm() kann frueh abgebrochen haben).
        # Warmup/Scan-Phase: Shutdown wird wie Cancel behandelt
        # (kein Verarbeitungs-Fortschritt → Auto-Resume wuerde von vorne beginnen)
        if check_control() in ("cancel", "shutdown"):
            self._finalize_cancelled(
                job_id,
                ds_id,
                {
                    "phase": "warmup",
                    "abort_reason": "Abgebrochen waehrend Backend-Warmup",
                    "duration_seconds": int(time.time() - start_time),
                },
            )
            return None

        if not warmup_result.success:
            duration = int(time.time() - start_time)
            failed = [d for d in warmup_result.details if not d["ok"]]
            # Nur Kategorien ohne OK-Backend melden (nicht einzelne Failover-Fehler)
            failed_categories = {d["category"] for d in warmup_result.details} - {
                d["category"] for d in warmup_result.details if d["ok"]
            }
            critical_failed = [d for d in failed if d["category"] in failed_categories]
            error_msg = "Backend-Warmup fehlgeschlagen: " + "; ".join(
                f"{d['category']}({d.get('model') or d['url']}): {d['error']}"
                for d in critical_failed
            )
            self._finalize_error(
                job_id,
                ds_id,
                error_msg,
                duration,
                config=config,
                embedding_model=embedding_model,
                stats={
                    "warmup": {
                        "success": warmup_result.success,
                        "duration_s": warmup_result.duration_s,
                        "details": warmup_result.details,
                    }
                },
            )
            return None

        # Warmup-Stats als Dict fuer run_ingestion() (DDU-Preflight-Skip).
        # Bewusste Dopplung zu _set_warmup_stats(): DB-Persistenz fuer Admin-UI,
        # Parameter-Uebergabe fuer Pipeline (kein zusaetzlicher DB-Read noetig).
        return {
            "success": warmup_result.success,
            "duration_s": warmup_result.duration_s,
            "details": warmup_result.details,
        }

    # --- Phase 4: Scan (parallel oder sequentiell) ---

    def _run_html_reingest_phase(
        self,
        job_id,
        ds_id,
        config,
        connector_cls,
        conn_config,
        ds_row,
        on_progress,
        check_control,
        embedding_model,
    ):
        from kiara.ingestion._file_utils import ChunkDatasources, HashStore, _CancelledError
        from kiara.ingestion.html_reingest import (
            assert_exclusive_sources,
            collect_candidates,
            prepare_reingestion,
        )
        from kiara.retrieval.vectorstore import VectorStore

        self._set_job_phase(job_id, "HTML-Bestand ermitteln...")
        signal = check_control()
        if signal in {"cancel", "shutdown"}:
            if signal == "cancel":
                self._finalize_cancelled(job_id, ds_id, {})
            return None
        # Bei Dateikonnektoren ist prepare_and_scan nur Pfad-/Mountvorbereitung.
        # Mailkonnektoren duerfen hier keinen normalen Postfachscan starten.
        if ds_row["type"] in {"local", "smb", "webdav"}:
            path = connector_cls.prepare_and_scan(
                conn_config, dict(ds_row), config, control_callback=check_control
            )
            if path is None:
                raise ValueError("Datenquellenpfad fuer HTML-Neuaufbereitung nicht erreichbar")
        candidates = config.get("_html_reingest_candidates")
        if candidates is None:
            try:
                candidates = collect_candidates(
                    config,
                    dict(ds_row),
                    embedding_model,
                    control_callback=check_control,
                    progress_callback=on_progress,
                )
            except _CancelledError:
                if check_control() != "shutdown":
                    self._finalize_cancelled(job_id, ds_id, {})
                return None
            # Vor erster Cachemutation festschreiben. Ein harter Abbruch in der
            # Vorbereitung kann dadurch keine spaeter hinzugekommenen Pfade aufnehmen.
            with db_transaction() as db:
                db.execute(
                    "SELECT pg_advisory_xact_lock(%s, %s)",
                    (INGESTION_DATASOURCE_LOCK_NAMESPACE, ds_id),
                )
                _update_job_stats(db, job_id, html_reingest_candidates=candidates)
            config["_html_reingest_candidates"] = candidates
        # Cacheaufbereitung darf ebenfalls keine fremde DS-Quelle veraendern.
        self._set_job_phase(job_id, "HTML-Dateizuordnungen pruefen...")
        try:
            assert_exclusive_sources(
                VectorStore(config, embedding_model=embedding_model, intent="maintenance"),
                ChunkDatasources(),
                ds_id,
                embedding_model,
                candidates,
                control_callback=check_control,
                progress_callback=on_progress,
            )
        except _CancelledError:
            if check_control() != "shutdown":
                self._finalize_cancelled(job_id, ds_id, {})
            return None
        self._set_job_phase(job_id, "HTML-Inhalte pruefen und aufbereiten...")
        stored_hashes = HashStore().load(ds_id, embedding_model)
        with db_transaction() as db:
            db.execute(
                "SELECT pg_advisory_xact_lock(%s, %s)", (INGESTION_DATASOURCE_LOCK_NAMESPACE, ds_id)
            )
            prepared = prepare_reingestion(
                config,
                connector_cls,
                conn_config,
                dict(ds_row),
                embedding_model,
                config["_html_reingest"],
                candidates,
                stored_hashes,
                on_progress,
                check_control,
                status_callback=lambda phase: self._set_job_phase(job_id, phase),
            )
            _update_job_stats(db, job_id, html_reingest_summary=prepared["summary"])
            if not prepared["cancelled"]:
                _update_job_stats(
                    db,
                    job_id,
                    html_migration=prepared["selection"],
                    html_migration_errors=prepared["errors"],
                )
        config["_html_reingest_summary"] = prepared["summary"]
        config["_html_reingest_prepared_hashes"] = prepared["prepared_hashes"]
        signal = check_control()
        if signal == "shutdown":
            return None
        if prepared["cancelled"] or signal == "cancel":
            self._finalize_cancelled(job_id, ds_id, {"html_reingest_summary": prepared["summary"]})
            return None
        config["_html_migration"] = {
            **prepared["selection"],
            "paths": prepared["paths"],
            "errors": prepared["errors"],
        }
        validate_run_selection(config, ds_id, embedding_model)
        return {"scan_thread": None, "scan_complete": None, "scan_error": [None]}

    def _run_scan_phase(
        self,
        job_id,
        ds_id,
        job_type,
        config,
        connector_cls,
        conn_config,
        ds_row,
        on_progress,
        check_control,
        *,
        embedding_model=None,
    ):
        """Fuehrt die Scan-Phase durch (parallel oder sequentiell).

        Setzt config["datasource"]["path"] als Seiteneffekt.

        Returns:
            dict mit keys: scan_thread, scan_complete, scan_error
            None bei Cancel/Shutdown/Fehler waehrend sequentiellem Scan.
        """
        if config.get("_html_reingest") is not None:
            if job_type != "incremental":
                raise ValueError("HTML-Neuaufbereitung darf keinen Voll-Cleanup starten")
            return self._run_html_reingest_phase(
                job_id,
                ds_id,
                config,
                connector_cls,
                conn_config,
                ds_row,
                on_progress,
                check_control,
                embedding_model,
            )
        migration = validate_run_selection(config, ds_id, embedding_model)
        if migration is not None:
            self._set_job_phase(job_id, "HTML-/Mailbestand aufbereiten...")
            # Nur die lokale/IMAP-Vorbereitung unter der vorhandenen DS-Sperre
            # halten. Vor Pipeline-Indexfences wird sie wieder freigegeben.
            with db_transaction() as db:
                db.execute(
                    "SELECT pg_advisory_xact_lock(%s, %s)",
                    (INGESTION_DATASOURCE_LOCK_NAMESPACE, ds_id),
                )
                prepared = prepare_migration_files(
                    config,
                    connector_cls,
                    conn_config,
                    dict(ds_row),
                    embedding_model,
                    on_progress,
                    check_control,
                )
                _update_job_stats(db, job_id, html_migration_errors=prepared["errors"])
            signal = check_control()
            if signal == "shutdown":
                return None  # Running-Auswahl bleibt fuer Auto-Resume erhalten.
            if prepared["cancelled"] or signal == "cancel":
                self._finalize_cancelled(
                    job_id,
                    ds_id,
                    {
                        "phase": "migration_prepare",
                        "html_migration": selection_spec(migration),
                    },
                )
                return None
            return {"scan_thread": None, "scan_complete": None, "scan_error": [None]}

        cache_path = connector_cls.effective_path(conn_config, dict(ds_row))
        _use_parallel = (
            job_type != "full"
            and getattr(connector_cls, "SUPPORTS_PARALLEL_SCAN", False)
            and os.path.isdir(cache_path)
            and any(True for e in os.scandir(cache_path) if e.is_file() or e.is_dir())
        )

        if _use_parallel:
            return self._start_parallel_scan(
                job_id,
                config,
                connector_cls,
                conn_config,
                ds_row,
                cache_path,
                check_control,
            )

        # Sequentiell: wie bisher
        self._set_job_phase(job_id, "Synchronisierung...")
        path = connector_cls.prepare_and_scan(
            conn_config,
            dict(ds_row),
            config,
            progress_callback=on_progress,
            control_callback=check_control,
        )
        if check_control() in ("cancel", "shutdown"):
            self._finalize_cancelled(job_id, ds_id, {"phase": "scan"})
            return None
        if path is None:
            raise ValueError(f"Vorbereitung fehlgeschlagen fuer Typ {ds_row['type']}")
        config["datasource"]["path"] = path
        return {"scan_thread": None, "scan_complete": None, "scan_error": [None]}

    def _start_parallel_scan(
        self, job_id, config, connector_cls, conn_config, ds_row, cache_path, check_control
    ):
        """Startet einen Parallel-Scan im Hintergrund.

        Returns:
            dict mit keys: scan_thread, scan_complete, scan_error
        """
        scan_error = [None]
        scan_complete = threading.Event()

        def _on_scan_progress(processed, total):
            try:
                with db_transaction() as _db:
                    _update_job_stats(_db, job_id, scan_processed=processed, scan_total=total)
            except Exception as exc:
                logger.debug("Job %s: Scan-Progress-Update fehlgeschlagen: %s", job_id, exc)

        def _background_scan():
            try:
                path = connector_cls.prepare_and_scan(
                    conn_config,
                    dict(ds_row),
                    config,
                    progress_callback=_on_scan_progress,
                    control_callback=check_control,
                )
                if path is None and check_control() not in ("cancel", "shutdown"):
                    scan_error[0] = RuntimeError(
                        f"Scan fuer DS {ds_row['id']} lieferte keinen Cache-Pfad"
                    )
            except Exception as exc:
                scan_error[0] = exc
                logger.error("Background-Scan fehlgeschlagen: %s", exc, exc_info=True)
            finally:
                scan_complete.set()

        # Scan-Cache invalidieren (sonst skip wegen .scan_done).
        # Nur sinnvoll fuer Konnektoren mit aktivem Scan-Cache; Parallel-Scan
        # ist heute auf IMAP/Exchange beschraenkt, der defensive Guard
        # vermeidet Permission-Errors falls das jemals erweitert wird.
        if getattr(connector_cls, "SCAN_CACHE_MAX_AGE", 0) > 0:
            _marker = os.path.join(cache_path, ".scan_done")
            try:
                os.remove(_marker)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning(
                    "Job %s: Scan-Cache-Marker %s konnte nicht entfernt werden: %s",
                    job_id,
                    _marker,
                    exc,
                )

        scan_thread = threading.Thread(
            target=_background_scan,
            daemon=True,
            name="bg-scan",
        )
        scan_thread.start()
        self._set_job_phase(job_id, "Synchronisierung + Verarbeitung...")
        logger.info("Job %s: Parallel-Scan gestartet, Ingestion auf Cache", job_id)

        config["datasource"]["path"] = cache_path
        return {
            "scan_thread": scan_thread,
            "scan_complete": scan_complete,
            "scan_error": scan_error,
        }

    # --- Phase 5: Full-Reingest-Cleanup ---

    @_guarded_worker_job("full_reingest_cleanup")
    def _prepare_full_reingest(self, job_id, ds_id, config, embedding_model):
        """Loescht Chunks und Hashes fuer Voll-Ingestion (referenz-gezaehlt).

        Reihenfolge ist crash-/retry-sicher:
        1. Read-only Preview via ``preview_datasource_removal`` fuer Dense und
           alle DS-relevanten ColBERT-Keys.
        2. Qdrant-Cleanup (Orphans loeschen, Non-Orphan-ds_ids aktualisieren).
           ``VectorDBConnectionError`` bricht VOR jeder DB-/Hash-Mutation ab,
           damit DB-/Hash-Retry-Anker erhalten bleiben.
        3. Elasticsearch-Loeschung fuer alte Dateipfade.
        4. BM25 fuer Dense-Collection invalidieren (nach finaler Dense-Mutation).
        5. EINE DB-/Hash-Transaktion: chunk_datasources (Dense + ColBERT),
           file_hashes, hash_mode_meta und datasource_embeddings atomar
           aktualisieren.
        """
        from kiara.exceptions import VectorDBConnectionError
        from kiara.ingestion._file_utils import ChunkDatasources
        from kiara.ingestion.vectordb_ops import (
            _resolve_colbert_models_for_ds,
            invalidate_bm25,
        )
        from kiara.retrieval.vectorstore import VectorStore

        chunk_ds = ChunkDatasources()

        store = VectorStore(config, embedding_model=embedding_model, intent="maintenance")

        # 1. Read-only Preview: Dense
        dense_orphans, dense_non_orphan_map = chunk_ds.preview_datasource_removal(
            ds_id,
            embedding_model,
        )

        # 1b. Read-only Preview: alle DS-relevanten ColBERT-Keys
        colbert_models = _resolve_colbert_models_for_ds(config, ds_id, chunk_ds)
        colbert_previews = []  # list of (model, key, orphans, non_orphan_map)
        if colbert_models:
            from kiara.retrieval.colbert import colbert_ds_model_key

            for model in colbert_models:
                key = colbert_ds_model_key(model)
                c_orphans, c_non_orphan = chunk_ds.preview_datasource_removal(
                    ds_id,
                    key,
                )
                colbert_previews.append((model, key, c_orphans, c_non_orphan))

        # 2. Qdrant-Cleanup: Dense + ColBERT. VectorDBConnectionError oder
        # Qdrant-Fehler brechen VOR DB-/Hash-Loeschung ab.
        try:
            if dense_orphans:
                store.delete_by_ids(dense_orphans)
            if dense_non_orphan_map:
                store.update_ds_ids(dense_non_orphan_map)
            if colbert_previews:
                from kiara.retrieval.colbert import ColBERTStore

                for model, _key, c_orphans, c_non_orphan in colbert_previews:
                    cstore = ColBERTStore(config, model, intent="maintenance")
                    if c_orphans:
                        cstore.delete_by_ids(c_orphans)
                    if c_non_orphan:
                        cstore.update_ds_ids(c_non_orphan, wait=True)
        except VectorDBConnectionError as exc:
            logger.warning(
                "Job %s: Qdrant-Cleanup fuer Full-Reingest fehlgeschlagen "
                "(DB-/Hash-Retry-Anker bleiben erhalten): %s",
                job_id,
                exc,
            )
            raise

        # Whole-source reset, including entries without a successful file hash.
        from kiara.retrieval.elasticsearch_service import (
            delete_datasource_index,
            model_uses_elasticsearch,
        )

        if model_uses_elasticsearch(config, embedding_model):
            delete_datasource_index(config, datasource_id=ds_id)

        # 4. BM25 NACH Dense-Qdrant-Mutation invalidieren
        try:
            invalidate_bm25(store.get_collection_name())
        except Exception as exc:
            logger.warning(
                "Job %s: BM25-Invalidation nach Full-Reingest fehlgeschlagen: %s",
                job_id,
                exc,
                exc_info=True,
            )

        # 5. Atomare DB-/Hash-Transaktion: chunk_datasources (Dense+ColBERT),
        # file_hashes, hash_mode_meta, datasource_embeddings in einem Commit.
        # All-keyspace Graph-Candidates: jeder Chunk, der in Dense ODER ColBERT
        # fuer diese DS referenziert wurde, muss nach dem Delete durch
        # filter_unreferenced_chunk_ids laufen. Nur Orphans zu sammeln
        # verliert Chunks, die in Dense+ColBERT dieselber DS non-orphan sind
        # (ImplScan V28, Finding 5).
        colbert_keys = [key for _, key, _, _ in colbert_previews]
        graph_candidate_ids = list(
            dict.fromkeys(
                dense_orphans
                + list(dense_non_orphan_map.keys())
                + [cid for _m, _k, c_orphans, _map in colbert_previews for cid in c_orphans]
                + [cid for _m, _k, _orphs, c_map in colbert_previews for cid in c_map]
            )
        )
        self._apply_full_reingest_db_cleanup(
            ds_id,
            embedding_model,
            colbert_keys,
            graph_candidate_ids,
        )
        logger.info("Job %s: Chunks geloescht fuer Voll-Ingestion", job_id)

    def _apply_full_reingest_db_cleanup(
        self,
        ds_id: int,
        embedding_model: str,
        colbert_keys: list[str],
        graph_candidate_ids: list[str] | None = None,
    ) -> None:
        """Atomare DB-/Hash-Mutation fuer Full-Reingest.

        Loescht in EINER Transaktion:
        - ``chunk_datasources`` fuer Dense-Modell und alle ColBERT-Keys
        - ``file_hashes`` und ``hash_mode_meta`` der Datasource (Dense)
        - ``datasource_embeddings``-Counter fuer Dense und ColBERT-Keys

        Die bestehenden no-conn Helper ``ChunkDatasources.delete_datasource_entries``
        und ``HashStore.delete_datasource`` oeffnen eigene Transaktionen und
        werden hier bewusst NICHT sequentiell gerufen — die SQL-Statements
        werden direkt auf demselben ``conn`` ausgefuehrt, damit ein Fehler
        die gesamte Mutation zurueckrollt.
        """
        from kiara.ingestion._file_utils import ChunkDatasources
        from kiara.shared.db import db_transaction

        chunk_ds = ChunkDatasources()
        with db_transaction() as conn:
            chunk_ds.acquire_graph_cleanup_lock(conn)
            # chunk_datasources: Dense
            conn.execute(
                "DELETE FROM chunk_datasources WHERE datasource_id = %s AND embedding_model = %s",
                (ds_id, embedding_model),
            )
            # chunk_datasources: ColBERT-Keys
            for key in colbert_keys:
                conn.execute(
                    "DELETE FROM chunk_datasources "
                    "WHERE datasource_id = %s AND embedding_model = %s",
                    (ds_id, key),
                )
            # file_hashes + hash_mode_meta: nur Dense-Modell
            conn.execute(
                "DELETE FROM file_hashes WHERE datasource_id = %s AND embedding_model = %s",
                (ds_id, embedding_model),
            )
            conn.execute(
                "DELETE FROM hash_mode_meta WHERE datasource_id = %s AND embedding_model = %s",
                (ds_id, embedding_model),
            )
            # datasource_embeddings: Dense + ColBERT-Keys
            conn.execute(
                "DELETE FROM datasource_embeddings "
                "WHERE datasource_id = %s AND embedding_model = %s",
                (ds_id, embedding_model),
            )
            for key in colbert_keys:
                conn.execute(
                    "DELETE FROM datasource_embeddings "
                    "WHERE datasource_id = %s AND embedding_model = %s",
                    (ds_id, key),
                )
            if graph_candidate_ids:
                graph_chunk_ids = chunk_ds.filter_unreferenced_chunk_ids(
                    graph_candidate_ids,
                    conn=conn,
                )
                if graph_chunk_ids:
                    from kiara.graph.store import GraphStore

                    GraphStore().delete_by_chunk_ids(graph_chunk_ids, conn=conn)

    # --- Phase 6: Resume-Skip ---

    def _load_resume_skip_files(self, job_id, resumed_from, job_type):
        """Laedt bereits verarbeitete Dateien aus dem Vorgaenger-Job.

        Returns:
            set von Dateipfaden oder None.
        """
        if not resumed_from or job_type == "full":
            return None
        try:
            with db_query() as _rdb:
                done_rows = _rdb.execute(
                    "SELECT file_path FROM ingestion_file_logs "
                    "WHERE job_id = %s AND status IN ('success', 'skipped')",
                    (resumed_from,),
                ).fetchall()
            if done_rows:
                skip_files = {r["file_path"] for r in done_rows}
                logger.info(
                    "Job %s: Resume von Job #%s — %d bereits verarbeitete Dateien",
                    job_id,
                    resumed_from,
                    len(skip_files),
                )
                return skip_files
        except Exception as exc:
            logger.warning("Resume-Skip-Abfrage fehlgeschlagen: %s", exc, exc_info=True)
        return None

    # --- Phase 7: Ingestion ausfuehren ---

    def _run_ingestion(
        self,
        job_id,
        ds_id,
        job_type,
        config,
        overrides,
        embedding_model,
        on_progress,
        check_control,
        callbacks,
        resume_skip_files,
        hash_mode,
        warmup_stats_dict,
    ):
        """Fuehrt run_ingestion() aus (Pass 1).

        Returns:
            tuple (stats_dict, flush_fn) — flush_fn muss nach Pass 2 erneut
            aufgerufen werden koennen.
        """
        from kiara.ingestion.pipeline import run_ingestion, set_current_job_id

        self._set_job_phase(job_id, "Hashing...")

        incremental = job_type != "full"
        migration = validate_run_selection(config, ds_id, embedding_model)
        if migration is not None:
            resume_skip_files = None  # Ganze Auswahl erneut; keine Teilschritt-Recovery.
        set_current_job_id(job_id)
        try:
            stats = run_ingestion(
                config,
                incremental=incremental,
                datasource_id=ds_id,
                ds_overrides=overrides,
                progress_callback=on_progress,
                control_callback=check_control,
                embedding_model=embedding_model,
                file_log_callback=callbacks["on_file_log"],
                file_start_callback=callbacks["on_file_start"],
                resume_skip_files=resume_skip_files,
                hash_mode=hash_mode,
                status_callback=callbacks["on_status"],
                activity_callback=callbacks.get("on_activity"),
                warmup_stats=warmup_stats_dict,
            )
        finally:
            set_current_job_id(None)

        if migration is not None:
            from kiara_ingestion_core.models import FileLogEntry

            stats["html_migration"] = selection_spec(migration)
            if config.get("_html_reingest") is not None:
                stats["html_reingest"] = config["_html_reingest"]
                stats["html_reingest_summary"] = config["_html_reingest_summary"]
            stats["html_migration_errors"] = migration["errors"]
            stats["errors"] = stats.get("errors", 0) + len(migration["errors"])
            for path, error in migration["errors"].items():
                callbacks["on_file_log"](
                    FileLogEntry(
                        file_path=path,
                        file_name=os.path.basename(path),
                        file_size=0,
                        file_ext=os.path.splitext(path)[1],
                        status="error",
                        error_message=error,
                        error_phase="migration_prepare",
                    )
                )
            expected = set(migration["files_by_model"][embedding_model])
            if set(stats.get("html_migration_completed_paths", [])) != expected:
                stats["abort_reason"] = "Migrationsauswahl nicht vollstaendig verarbeitet"

        # Rest-Buffer flushen
        callbacks["flush_file_logs"]()
        return stats, callbacks["flush_file_logs"]

    # --- Phase 8: Parallel-Scan Pass 2 ---

    def _run_pass2(
        self,
        job_id,
        ds_id,
        config,
        overrides,
        embedding_model,
        on_progress,
        check_control,
        callbacks,
        hash_mode,
        warmup_stats_dict,
        stats,
        flush_fn,
        scan_info,
    ):
        """Fuehrt Pass 2 nach Parallel-Scan durch (neue Dateien verarbeiten).

        Mutiert *stats* in-place (fuehrt Pass-2-Stats zusammen).
        """
        from kiara.ingestion.pipeline import run_ingestion, set_current_job_id

        scan_thread = scan_info["scan_thread"]
        scan_complete = scan_info["scan_complete"]
        scan_error = scan_info["scan_error"]

        # Scan-Thread joinen (warte bis Download fertig)
        if not scan_complete.is_set():
            self._set_job_phase(job_id, "Warte auf Synchronisierung...")
            logger.info("Job %s: Pass 1 fertig, warte auf Scan...", job_id)

        while not scan_complete.is_set():
            scan_complete.wait(timeout=2.0)
            if check_control() in ("cancel", "shutdown"):
                break
        scan_thread.join(timeout=10)

        # Scan-Progress entfernen (scan fertig)
        try:
            with db_transaction() as _db:
                _update_job_stats(_db, job_id, scan_processed=None, scan_total=None)
        except Exception as exc:
            logger.debug("Job %s: Scan-Progress-Reset fehlgeschlagen: %s", job_id, exc)

        # Pass 2 nur wenn: kein Cancel, kein Scan-Fehler, kein Abort
        if (
            not stats.get("abort_reason")
            and check_control() not in ("cancel", "shutdown")
            and not scan_error[0]
        ):
            self._set_job_phase(job_id, "Verarbeite neue Dateien...")
            logger.info("Job %s: Starte Pass 2 (neue Dateien)", job_id)

            set_current_job_id(job_id)
            try:
                stats_p2 = run_ingestion(
                    config,
                    incremental=True,
                    datasource_id=ds_id,
                    ds_overrides=overrides,
                    progress_callback=on_progress,
                    control_callback=check_control,
                    embedding_model=embedding_model,
                    file_log_callback=callbacks["on_file_log"],
                    file_start_callback=callbacks["on_file_start"],
                    resume_skip_files=None,
                    hash_mode=hash_mode,
                    status_callback=callbacks["on_status"],
                    activity_callback=callbacks.get("on_activity"),
                    warmup_stats=warmup_stats_dict,
                )
            finally:
                set_current_job_id(None)
            flush_fn()

            # Stats zusammenfuehren
            for key in (
                "processed_files",
                "total_chunks",
                "errors",
                "deleted_files",
                "dedup_chunks",
                "ddu_files",
                "ddu_remote_files",
                "ddu_remote_errors",
                "ddu_timeout",
                "vlm_chunks",
                "vlm_errors",
                "graph_entities",
                "graph_relations",
                "graph_errors",
                "vectordb_errors",
                "vectordb_recoveries",
                "contract_discovery_waits",
                "contract_discovery_recoveries",
                "contract_discovery_wait_seconds",
            ):
                if key in stats_p2:
                    stats[key] = stats.get(key, 0) + stats_p2[key]
            stats["total_files"] = stats_p2.get("total_files", stats.get("total_files", 0))
            stats["skipped_files"] = stats_p2.get("skipped_files", stats.get("skipped_files", 0))
            if stats_p2.get("abort_reason"):
                stats["abort_reason"] = stats_p2["abort_reason"]
            if stats_p2.get("index_compatibility_error"):
                stats["index_compatibility_error"] = stats_p2["index_compatibility_error"]
            if stats_p2.get("index_compatibility_rollback_errors"):
                stats["index_compatibility_rollback_errors"] = stats_p2[
                    "index_compatibility_rollback_errors"
                ]
            if stats_p2.get("error_details"):
                stats.setdefault("error_details", [])
                stats["error_details"].extend(stats_p2["error_details"])
            stats["parallel_scan"] = True
            stats["pass2_new_files"] = stats_p2.get("processed_files", 0)
            logger.info(
                "Job %s: Pass 2 fertig — %d neue Dateien",
                job_id,
                stats_p2.get("processed_files", 0),
            )
        elif scan_error[0]:
            logger.warning(
                "Job %s: Scan fehlgeschlagen: %s — Pass-1-Ergebnisse bleiben", job_id, scan_error[0]
            )
            stats["abort_reason"] = f"Parallel-Scan fehlgeschlagen: {scan_error[0]}"
            stats["parallel_scan"] = True

    # --- Phase 9: Job-Ergebnis auswerten ---

    def _resolve_job_outcome(
        self, job_id, ds_id, config, embedding_model, stats, duration, check_control
    ):
        """Entscheidet ueber Success/Error/Cancel/Shutdown und finalisiert."""
        # --- Shutdown-Check (VOR Finalize) ---
        if self._stop.is_set():
            # Graceful Shutdown: Job NICHT finalisieren.
            # Bleibt als 'running' in DB → _recover_stale_jobs() auto-resumed.
            logger.info("Job %s: Shutdown — wird beim naechsten Start fortgesetzt", job_id)
            return

        # Race-Window: Wenn SIGTERM zwischen self._stop.is_set() (oben) und
        # _check_control() eintrifft, gibt _check_control() "shutdown" zurueck.
        signal = check_control()
        if signal == "shutdown":
            logger.info("Job %s: Shutdown (late) — wird beim naechsten Start fortgesetzt", job_id)
            return
        if stats.get("html_migration") and stats.get("errors") and not stats.get("abort_reason"):
            stats["abort_reason"] = "Dateifehler bei HTML-/Mailmigration"
        if signal == "cancel":
            self._finalize_cancelled(job_id, ds_id, stats)
        elif stats.get("abort_reason"):
            self._finalize_error(
                job_id,
                ds_id,
                stats["abort_reason"],
                duration,
                config=config,
                embedding_model=embedding_model,
                stats=stats,
            )
        else:
            self._finalize_success(job_id, ds_id, stats, config, embedding_model)

    def _finalize_success(self, job_id, ds_id, stats, config, embedding_model):
        """Job erfolgreich abgeschlossen (Orchestrator).

        Delegiert an:
        - _aggregate_final_stats(): VektorDB-Stats sammeln
        - _update_datasource_metadata(): DB-Updates fuer Job + Datasource
        - _log_job_completion(): Logging, Cache-Invalidierung, Cron-Drift-Check
        """
        self._close_progress_cache()
        chunk_count, finished = self._aggregate_final_stats(
            stats,
            config,
            ds_id,
            embedding_model,
        )

        with db_transaction() as db:
            self._update_datasource_metadata(
                db,
                job_id,
                ds_id,
                stats,
                config,
                embedding_model,
                chunk_count,
                finished,
            )

        self._log_job_completion(
            job_id,
            ds_id,
            stats,
            embedding_model,
        )

    def _aggregate_final_stats(self, stats, config, ds_id, embedding_model):
        """Sammelt finale VektorDB-Statistiken und ergaenzt sie in stats.

        Liest den Chunk-Count aus der ChunkDatasources-SoT und fragt die
        VektorDB nach Gesamtpunktestand, Backend und Collection.

        Args:
            stats: Mutable Stats-Dict (wird in-place ergaenzt).
            config: KIara-Config-Dict.
            ds_id: Datasource-ID.
            embedding_model: Verwendetes Embedding-Modell.

        Returns:
            tuple: (chunk_count, finished_timestamp)
        """
        from kiara.ingestion._file_utils import ChunkDatasources
        from kiara.retrieval.vectorstore import VectorStore

        chunk_ds = ChunkDatasources()
        chunk_count = chunk_ds.count_by_datasource(ds_id, embedding_model)
        store = VectorStore(config, embedding_model=embedding_model, intent="maintenance")
        finished = datetime.now()

        # Vektor-DB Punktestand in Stats erfassen
        stats["vectordb_points_ds"] = chunk_count
        try:
            vdb_stats = store.get_stats()
            stats["vectordb_points_total"] = vdb_stats.get("total_chunks", 0)
            stats["vectordb_backend"] = vdb_stats.get("backend", "")
            stats["vectordb_collection"] = vdb_stats.get("collection", "")
        except Exception as exc:
            from kiara.retrieval.index_compatibility import IndexCompatibilityError

            if isinstance(exc, IndexCompatibilityError):
                raise
            logger.warning("VectorDB-Stats nicht abrufbar: %s", exc, exc_info=True)

        return chunk_count, finished

    def _update_datasource_metadata(
        self, db, job_id, ds_id, stats, config, embedding_model, chunk_count, finished
    ):
        """Aktualisiert Job-Status, datasource_embeddings und datasources in der DB.

        Muss innerhalb einer aktiven DB-Transaktion aufgerufen werden.

        Args:
            db: Aktive DB-Transaktion.
            job_id: Job-ID.
            ds_id: Datasource-ID.
            stats: Stats-Dict (wird als JSON serialisiert).
            config: KIara-Config-Dict (unused, fuer Signatur-Konsistenz).
            embedding_model: Verwendetes Embedding-Modell.
            chunk_count: Anzahl Chunks fuer diese DS/Modell-Kombination.
            finished: ISO-Timestamp des Abschlusszeitpunkts.
        """
        preserve_selection(db, job_id, stats)
        _job_repo.update(
            job_id,
            conn=db,
            status=IngestionStatus.SUCCESS.value,
            requested_action=None,
            finished_at=finished,
            stats_json=json.dumps(stats, ensure_ascii=False, default=str),
        )
        # datasource_embeddings upsert — Modell kommt bereits normalisiert aus enqueue_job()
        if stats.get("html_migration"):
            from kiara.ingestion.vectordb_ops import _recount_datasource_embeddings

            _recount_datasource_embeddings(db, ds_id, embedding_model)
        else:
            _emb_repo.upsert_stats(
                datasource_id=ds_id,
                embedding_model=embedding_model,
                chunk_count=chunk_count,
                file_count=stats.get("total_files", 0),
                conn=db,
            )
        _ds_repo.update(
            ds_id,
            conn=db,
            ingestion_status=IngestionStatus.SUCCESS.value,
            last_error="",
            processed_files=stats.get("processed_files", 0),
        )

    def _log_job_completion(self, job_id, ds_id, stats, embedding_model):
        """Loggt Job-Abschluss und fuehrt Post-Success-Aktionen durch.

        Invalidiert BM25- und Admin-UI-Caches und prueft auf
        Cron-Modell-Drift (Warnung falls Cron-Schedule ein anderes
        Embedding-Modell verwendet).

        Args:
            job_id: Job-ID.
            ds_id: Datasource-ID.
            stats: Stats-Dict (fuer Log-Ausgabe).
            embedding_model: Verwendetes Embedding-Modell.
        """
        logger.info("Job %s (DS %s): Erfolgreich — %s", job_id, ds_id, stats)

        # BM25-Cache invalidieren, damit er beim naechsten Query neu aufgebaut wird
        try:
            from kiara.retrieval.retriever import HybridRetriever

            HybridRetriever.invalidate_bm25_cache()
        except Exception as exc:
            logger.debug("BM25-Cache-Invalidierung fehlgeschlagen: %s", exc)

        # Admin-UI-Caches aktualisieren
        from kiara.web.cache_refresh import (
            refresh_datasource_cache,
            refresh_jobs_cache,
            refresh_vectordb_cache,
        )

        safe_refresh(refresh_datasource_cache, ds_id)
        safe_refresh(refresh_jobs_cache, job_id)
        safe_refresh(refresh_vectordb_cache)

        # Cron-Modell-Drift-Warnung
        # HINWEIS: Dieses Parsing ist gekoppelt an das Cron-Zeilen-Format
        # in write_schedule() (cron_utils.py). Aenderungen am Format muessen
        # an beiden Stellen synchron erfolgen.
        try:
            from kiara.shared.cron_utils import read_schedules

            schedules = read_schedules()
            sched = schedules.get(ds_id)
            if sched:
                cron_line = sched.get("line", "")
                import shlex

                parts = shlex.split(cron_line)
                for i, p in enumerate(parts):
                    if p == "--embedding-model" and i + 1 < len(parts):
                        from kiara.backend_selector import normalize_model_name as _norm

                        cron_model = parts[i + 1]
                        if _norm(cron_model) != _norm(embedding_model):
                            logger.warning(
                                "DS %s: Cron-Schedule verwendet Modell '%s', "
                                "aber aktuelle Ingestion verwendet '%s'. "
                                "Bitte Cron-Schedule in der Admin-UI aktualisieren.",
                                ds_id,
                                cron_model,
                                embedding_model,
                            )
                        break
        except Exception as exc:
            logger.debug("Cron-Drift-Check fehlgeschlagen: %s", exc)

    def _set_warmup_stats(self, job_id, result) -> None:
        """Schreibt Warmup-Ergebnis atomar in stats_json (JSONB-||-Merge).

        Heartbeat-Thread schreibt parallel ins selbe stats_json — daher
        _update_job_stats() nutzen (atomares JSONB-Merge auf DB-Ebene),
        nicht Read-Modify-Write.
        """
        try:
            with db_transaction() as db:
                _update_job_stats(
                    db,
                    job_id,
                    warmup={
                        "success": result.success,
                        "duration_s": result.duration_s,
                        "details": result.details,
                    },
                )
        except Exception:
            logger.debug("Warmup-Stats fuer Job %s nicht geschrieben", job_id, exc_info=True)

    def _finalize_error(
        self, job_id, ds_id, error, duration=0, config=None, embedding_model=None, stats=None
    ):
        """Job fehlgeschlagen — Chunk-Counts trotzdem aktualisieren."""
        self._close_progress_cache()
        if stats is None:
            stats = {}
        stats.update({"error": error, "duration_seconds": duration})
        with db_transaction() as db:
            preserve_selection(db, job_id, stats)
            _job_repo.update(
                job_id,
                conn=db,
                status=IngestionStatus.ERROR.value,
                requested_action=None,
                error_message=str(error)[:1000],
                stats_json=json.dumps(stats, default=str),
            )
            # Use RawSQL for finished_at = NOW()
            db.execute(
                "UPDATE ingestion_jobs SET finished_at = NOW() WHERE id = %s",
                (job_id,),
            )
            # datasource_embeddings trotzdem aktualisieren — erfolgreich
            # verarbeitete Chunks sind bereits in der VektorDB
            if config and embedding_model:
                try:
                    from kiara.ingestion._file_utils import ChunkDatasources

                    chunk_ds = ChunkDatasources()
                    chunk_count = chunk_ds.count_by_datasource(ds_id, embedding_model)
                    _emb_repo.upsert_chunk_count(
                        datasource_id=ds_id,
                        embedding_model=embedding_model,
                        chunk_count=chunk_count,
                        conn=db,
                    )
                except Exception as exc:
                    logger.warning(
                        "Chunk-Count-Update bei Fehler-Job fehlgeschlagen: %s",
                        exc,
                        exc_info=True,
                    )
            _ds_repo.update(
                ds_id,
                conn=db,
                ingestion_status=IngestionStatus.ERROR.value,
                last_error=str(error)[:1000],
            )
        logger.error("Job %s (DS %s): Fehler — %s", job_id, ds_id, error)

        # Admin-UI-Caches aktualisieren
        from kiara.web.cache_refresh import refresh_datasource_cache, refresh_jobs_cache

        safe_refresh(refresh_datasource_cache, ds_id)
        safe_refresh(refresh_jobs_cache, job_id)

    def _finalize_cancelled(self, job_id, ds_id, stats):
        """Job abgebrochen."""
        self._close_progress_cache()
        abort_reason = stats.get("abort_reason", "")
        with db_transaction() as db:
            preserve_selection(db, job_id, stats)
            _job_repo.update(
                job_id,
                conn=db,
                status=IngestionStatus.CANCELLED.value,
                requested_action=None,
                stats_json=json.dumps(stats, ensure_ascii=False, default=str),
            )
            db.execute(
                "UPDATE ingestion_jobs SET finished_at = NOW() WHERE id = %s",
                (job_id,),
            )
            _ds_repo.update(
                ds_id,
                conn=db,
                ingestion_status=IngestionStatus.IDLE.value,
            )
        # Admin-UI-Caches aktualisieren
        from kiara.web.cache_refresh import refresh_datasource_cache, refresh_jobs_cache

        safe_refresh(refresh_datasource_cache, ds_id)
        safe_refresh(refresh_jobs_cache, job_id)

        if abort_reason:
            logger.warning(
                "Job %s (DS %s): Abgebrochen (Pipeline-Abbruch: %s)", job_id, ds_id, abort_reason
            )
        else:
            logger.info("Job %s (DS %s): Abgebrochen", job_id, ds_id)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_worker: IngestionWorker | None = None
_worker_lock = threading.Lock()


def get_worker() -> IngestionWorker:
    """Gibt die Singleton-Worker-Instanz zurueck (erzeugt bei Bedarf, thread-safe)."""
    global _worker
    if _worker is None:
        with _worker_lock:
            if _worker is None:
                _worker = IngestionWorker()
    return _worker


def retry_html_job(job_id: int) -> int:
    """HTML-Auftrag in derselben Resume-Kette mit seiner eingefrorenen Auswahl wiederholen."""
    from kiara.ingestion.html_reingest import validate_request

    with db_query() as db:
        row = db.execute("SELECT * FROM ingestion_jobs WHERE id = %s", (job_id,)).fetchone()
    if not row or row["status"] not in {"error", "cancelled"}:
        raise ValueError("Nur fehlgeschlagene oder abgebrochene HTML-Jobs sind wiederholbar")
    fields = request_fields(row["stats_json"])
    if not fields:
        raise ValueError("Job ist kein HTML-Neuaufbereitungsauftrag")
    validate_request(fields["html_reingest"], check_current=True)
    result = get_worker()._recover_single_job(None, dict(row), manual_html_retry=True)
    if not result:
        raise ValueError("HTML-Job konnte nicht wiederholt werden")
    from kiara.web.cache_refresh import refresh_jobs_cache

    safe_refresh(refresh_jobs_cache, result)
    return result


def enqueue_job(
    ds_id: int,
    job_type: str = "incremental",
    triggered_by: str = "manual",
    *,
    embedding_model: str,
    html_migration: dict | None = None,
    html_reingest: str | None = None,
) -> int:
    """Erstellt einen neuen Job in der Queue.

    embedding_model ist keyword-only ohne Default — fehlende Aufrufe
    scheitern am Funktionsaufruf (TypeError) statt erst zur Laufzeit.

    Raises:
        TypeError: Wenn embedding_model nicht uebergeben wird (Python-nativ).
        ValueError: Wenn embedding_model leer ist oder ein Job bereits existiert.
    """
    from kiara.backend_selector import normalize_model_name
    from kiara.config import load_config
    from kiara.retrieval.index_compatibility import (
        IndexOperation,
        guard_index,
        primary_index_target,
        raw_qdrant_client,
    )

    embedding_model = normalize_model_name(embedding_model)
    initial_stats = {}
    if html_reingest is not None:
        from kiara.ingestion.html_reingest import build_request, validate_target

        if job_type != "incremental" or html_migration is not None:
            raise ValueError("HTML-Modus braucht einen eigenen inkrementellen Job")
        initial_stats["html_reingest"] = build_request(html_reingest)
        with db_query() as db:
            ds = db.execute("SELECT * FROM datasources WHERE id = %s", (ds_id,)).fetchone()
            if not ds:
                raise ValueError(f"Datenquelle {ds_id} fehlt")
            validate_target(dict(ds), embedding_model)
            stock = db.execute(
                "SELECT EXISTS (SELECT 1 FROM file_hashes WHERE datasource_id = %s AND embedding_model = %s) "
                "OR EXISTS (SELECT 1 FROM chunk_datasources WHERE datasource_id = %s AND embedding_model = %s) AS present",
                (ds_id, embedding_model, ds_id, embedding_model),
            ).fetchone()
            if not stock["present"]:
                raise ValueError("Kein bestehender Bestand fuer das gewaehlte Modell")
    if html_migration is not None:
        if job_type != "incremental":
            raise ValueError("Migrationsjobs muessen incremental bleiben")
        initial_stats["html_migration"] = validate_selection(html_migration, ds_id, embedding_model)
    if not embedding_model:
        raise ValueError(
            f"DS {ds_id}: Kein Embedding-Modell angegeben. "
            "Jeder Ingestion-Job muss ein Embedding-Modell spezifizieren."
        )

    config = load_config()
    target = primary_index_target(config, embedding_model)
    with guard_index(
        config,
        target,
        operation=IndexOperation.JOB_ENQUEUE,
        write=True,
        allow_initialize=html_reingest is None,
        qdrant_client=raw_qdrant_client(config),
    ) as contract_token:
        contract = contract_token.job_fields()
        db = contract_token.connection
        db.execute(
            "SELECT pg_advisory_xact_lock(%s, %s)",
            (INGESTION_DATASOURCE_LOCK_NAMESPACE, ds_id),
        )
        ds_row = db.execute(
            "SELECT id, is_active FROM datasources WHERE id = %s FOR UPDATE",
            (ds_id,),
        ).fetchone()
        if not ds_row:
            raise ValueError(f"DS {ds_id}: Datenquelle nicht gefunden")
        if not ds_row["is_active"]:
            raise ValueError(f"DS {ds_id}: Datenquelle ist deaktiviert")

        status_placeholders = ", ".join(["%s"] * len(_ACTIVE_JOB_STATUSES))
        existing = db.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s "
            f"AND status IN ({status_placeholders}) "
            "ORDER BY id ASC LIMIT 1 FOR UPDATE",
            (ds_id, *_ACTIVE_JOB_STATUSES),
        ).fetchone()
        if existing:
            raise ValueError(f"DS {ds_id} hat bereits einen aktiven Job (#{existing['id']})")
        job_id = insert_returning(
            db,
            "INSERT INTO ingestion_jobs (datasource_id, job_type, triggered_by, "
            "embedding_model, stats_json, queued_at, contract_logical_index_key, "
            "contract_index_compatibility_id, contract_query_compatibility_id, "
            "contract_canonical_model_id, contract_profile_id, contract_profile_kind, "
            "contract_profile_endpoint, contract_snapshot_sha256, contract_fence_version) "
            "VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                ds_id,
                job_type,
                triggered_by,
                embedding_model,
                json.dumps(initial_stats, ensure_ascii=False),
                contract["contract_logical_index_key"],
                contract["contract_index_compatibility_id"],
                contract["contract_query_compatibility_id"],
                contract["contract_canonical_model_id"],
                contract["contract_profile_id"],
                contract["contract_profile_kind"],
                contract["contract_profile_endpoint"],
                contract["contract_snapshot_sha256"],
                contract["contract_fence_version"],
            ),
        )

    logger.info(
        "Job #%s enqueued (DS %s, %s, model=%s, by %s)",
        job_id,
        ds_id,
        job_type,
        embedding_model,
        triggered_by,
    )

    # Admin-UI-Caches aktualisieren
    from kiara.web.cache_refresh import refresh_datasource_cache, refresh_jobs_cache

    safe_refresh(refresh_jobs_cache, job_id)
    safe_refresh(refresh_datasource_cache, ds_id)

    return job_id
