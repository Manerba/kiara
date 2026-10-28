"""CLI for KIara."""

import sys

import click
import yaml

from kiara.config import DEFAULT_PATHS, get_embedding_model, load_config
from kiara.shared.constants import ADMIN_PORT_DEFAULT, WORKER_LOCK_PATH
from kiara.shared.secret_redact import redact_secrets_recursive


@click.group()
@click.option("--config", "config_path", default=None, help="Path to config YAML file.")
@click.pass_context
def main(ctx, config_path):
    """KIara – KI-Assistent fuer Retrieval und Analyse."""
    import os

    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    if config_path:
        os.environ["KIARA_CONFIG"] = config_path


def _get_config(ctx) -> dict:
    return load_config(ctx.obj.get("config_path"))


def _prepare_datasource_index(
    config: dict, datasource_id: int
) -> tuple[dict, dict, dict, str, str]:
    """Build the effective ingestion config for one DB-backed datasource."""
    import copy

    from kiara.connectors.registry import ensure_loaded
    from kiara.connectors.registry import get as get_connector
    from kiara.ingestion.datasource_runtime import build_datasource_overrides, load_connector_config
    from kiara.repositories.datasource import DatasourceRepository

    ensure_loaded()

    row = DatasourceRepository().get_by_id(datasource_id)
    if not row:
        raise click.ClickException(f"Datenquelle {datasource_id} nicht gefunden.")
    if not row.get("is_active"):
        raise click.ClickException(f"Datenquelle {datasource_id} ({row['name']}) ist deaktiviert.")

    connector_cls = get_connector(row["type"])
    if not connector_cls:
        raise click.ClickException(
            f"Unbekannter Datenquellen-Typ {row['type']!r} fuer ID {datasource_id}."
        )

    ds_config = copy.deepcopy(config)
    conn_config = load_connector_config(row, connector_cls)
    path = connector_cls.safe_prepare_and_scan(conn_config, dict(row), ds_config)
    if path is None:
        raise click.ClickException(f"Datenquelle {datasource_id} ist nicht lesbar.")

    ds_config["datasource"]["path"] = path
    ds_overrides = build_datasource_overrides(row)
    hash_mode = getattr(connector_cls, "HASH_MODE", "mtime")
    return ds_config, row, ds_overrides, hash_mode, path


@main.command()
@click.option("--incremental", is_flag=True, help="Only process changed/new files.")
@click.option("--datasource-id", type=int, default=None, help="Index a specific datasource by ID.")
@click.option("--all-datasources", is_flag=True, help="Index all active datasources.")
@click.pass_context
def index(ctx, incremental, datasource_id, all_datasources):
    """Index the knowledge base (full or incremental)."""
    from kiara.ingestion.pipeline import run_all_datasources, run_ingestion

    config = _get_config(ctx)
    # CLI-Policy: Kein --config → persistiert (markieren),
    # explizites --config → kein Marker (Custom-Config-Laeufe)
    if not ctx.obj.get("config_path"):
        config["_kiara_enforce_persisted_colbert_model"] = True
    mode = "incremental" if incremental else "full"

    try:
        if all_datasources:
            click.echo(f"Starting {mode} indexing for all active datasources...")
            click.echo(f"Embedding model: {get_embedding_model(config)}")
            click.echo()

            stats = run_all_datasources(config, incremental=incremental)

            click.echo()
            click.echo("Indexing complete (all datasources):")
            click.echo(f"  Datasources processed: {stats.get('datasources_processed', 0)}")
        elif datasource_id:
            ds_config, row, ds_overrides, hash_mode, path = _prepare_datasource_index(
                config, datasource_id
            )
            click.echo(f"Starting {mode} indexing...")
            click.echo(f"Datasource: {path}")
            click.echo(f"Datasource ID: {datasource_id} ({row['name']})")
            click.echo(f"Embedding model: {get_embedding_model(ds_config)}")
            click.echo()

            stats = run_ingestion(
                ds_config,
                incremental=incremental,
                datasource_id=datasource_id,
                ds_overrides=ds_overrides or None,
                hash_mode=hash_mode,
            )
        else:
            click.echo(f"Starting {mode} indexing...")
            click.echo(f"Datasource: {config['datasource']['path']}")
            click.echo(f"Embedding model: {get_embedding_model(config)}")
            click.echo()

            stats = run_ingestion(config, incremental=incremental)

        click.echo()
        click.echo("Indexing complete:")
        click.echo(f"  Total files found:  {stats['total_files']}")
        click.echo(f"  Files processed:    {stats['processed_files']}")
        click.echo(f"  Files skipped:      {stats['skipped_files']}")
        click.echo(f"  Files deleted:      {stats['deleted_files']}")
        click.echo(f"  Chunks created:     {stats['total_chunks']}")
        click.echo(f"  Errors:             {stats['errors']}")
    except KeyboardInterrupt:
        click.echo("\nAbgebrochen.")
        sys.exit(130)


@main.command()
@click.argument("question")
@click.option("--filter-dir", default=None, help="Filter by directory (substring match).")
@click.option("--filter-type", default=None, help="Filter by file type (e.g. .ps1).")
@click.option("-k", default=5, help="Number of context chunks.")
@click.pass_context
def query(ctx, question, filter_dir, filter_type, k):
    """Query the knowledge base."""
    from kiara.generation.backends import create_backend
    from kiara.generation.chain import RAGChain
    from kiara.retrieval.index_compatibility import IndexCompatibilityError

    try:
        config = _get_config(ctx)
        backend = create_backend(config)
        chain = RAGChain(config, backend=backend)
        result = chain.query(
            question=question,
            k=k,
            filter_dir=filter_dir,
            filter_type=filter_type,
        )
        if not isinstance(result, dict) or "answer" not in result:
            raise ValueError("Query-Ergebnis enthaelt keine Antwort")

        click.echo()
        click.echo(result["answer"])
        click.echo()

        sources = result.get("sources") or []
        if sources:
            click.echo("--- Quellen ---")
            for i, src in enumerate(sources, 1):
                score_val = src.get("score")
                score_str = f"{score_val:.3f}" if score_val is not None else "N/A"
                click.echo(f"  [{i}] {src['source']} (Relevanz: {score_str})")
    except KeyboardInterrupt:
        click.echo("\nAbgebrochen.", err=True)
        sys.exit(130)
    except IndexCompatibilityError as exc:
        import json

        click.echo(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True), err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Query fehlgeschlagen: {exc}", err=True)
        sys.exit(1)


@main.command()
@click.pass_context
def status(ctx):
    """Show index statistics."""
    from kiara.retrieval.index_compatibility import IndexCompatibilityError
    from kiara.retrieval.retriever import Retriever

    try:
        config = _get_config(ctx)
        retriever = Retriever(config)
        stats = retriever.get_stats()
        if not isinstance(stats, dict):
            raise ValueError("Statusdaten nicht verfuegbar")

        click.echo("KIara Status:")
        click.echo(f"  Collection:    {stats.get('collection', 'unbekannt')}")
        click.echo(f"  Total chunks:  {stats.get('total_chunks', 'unbekannt')}")
        click.echo(f"  VectorDB path: {stats.get('vectordb_path', 'unbekannt')}")
    except KeyboardInterrupt:
        click.echo("\nAbgebrochen.", err=True)
        sys.exit(130)
    except IndexCompatibilityError as exc:
        import json

        click.echo(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True), err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Status fehlgeschlagen: {exc}", err=True)
        sys.exit(1)


@main.command("config")
@click.pass_context
def show_config(ctx):
    """Show current configuration."""
    config = _get_config(ctx)
    redacted = redact_secrets_recursive(config)
    click.echo(yaml.dump(redacted, default_flow_style=False, allow_unicode=True))


@main.command()
@click.pass_context
def serve(ctx):
    """Start the KIara Web UI (FastAPI)."""
    import os

    import uvicorn

    config = _get_config(ctx)
    host = config["web"]["host"]
    port = config["web"]["port"]

    if ctx.obj.get("config_path"):
        os.environ["KIARA_CONFIG"] = ctx.obj["config_path"]

    click.echo(f"Starting KIara Web UI on {host}:{port}")

    uvicorn.run(
        "kiara.web.app:app",
        host=host,
        port=port,
        log_level="info",
        workers=config.get("web", {}).get("workers", 4),
    )


@main.command("serve-admin")
@click.pass_context
def serve_admin(ctx):
    """Start the KIara Admin UI (FastAPI)."""
    import os

    import uvicorn

    config = _get_config(ctx)
    host = config["web"]["host"]
    port = config["web"].get("admin_port", ADMIN_PORT_DEFAULT)

    if ctx.obj.get("config_path"):
        os.environ["KIARA_CONFIG"] = ctx.obj["config_path"]

    click.echo(f"Starting KIara Admin UI on {host}:{port}")

    uvicorn.run(
        "kiara.web.admin_app:app",
        host=host,
        port=port,
        log_level="info",
    )


@main.command("serve-worker")
@click.pass_context
def serve_worker(ctx):
    """Startet den Ingestion-Worker als eigenstaendigen Daemon."""
    import fcntl
    import os
    import signal

    from kiara.web.database import init_db
    from kiara.web.faulthandler_setup import enable_faulthandler
    from kiara.web.ingestion_worker import get_worker

    # Erlaubt im Incident-Fall einen nicht terminierenden Python-Stackdump via
    # SIGUSR1. Der automatische D-State-Collector persistiert den Stack zusaetzlich.
    enable_faulthandler()

    # --- Singleton-Schutz: Nur EIN Worker-Prozess gleichzeitig ---
    lock_path = WORKER_LOCK_PATH
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        click.echo("FEHLER: Ein anderer kiara-worker Prozess laeuft bereits.", err=True)
        raise SystemExit(1)
    lock_fd.write(str(os.getpid()))
    lock_fd.flush()

    config = _get_config(ctx)

    if ctx.obj.get("config_path"):
        os.environ["KIARA_CONFIG"] = ctx.obj["config_path"]

    # DB-Backend initialisieren (muss VOR init_db() passieren)
    from kiara.shared.db import init_db_backend

    init_db_backend(config)

    init_db()

    # Der eigenstaendige Worker braucht den DB-Handler explizit, damit
    # D-State-Zustandsereignisse im Admin-Tab System/System-Log erscheinen.
    from kiara.web.logging_config import setup_security_logger

    setup_security_logger()

    from kiara.shared.tracing import init_tracing

    init_tracing(config)

    worker = get_worker()

    # Signal-Handler VOR worker.start() registrieren
    def _handle_signal(signum, frame):
        worker._stop.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    worker.start()

    click.echo("Ingestion-Worker gestartet (PID %s)" % os.getpid())
    try:
        # Blockiert bis _stop gesetzt wird (durch Signal-Handler)
        worker._stop.wait()
        # Worker-Thread beenden — KEIN Timeout, systemd's TimeoutStopSec greift
        if worker._thread and worker._thread.is_alive():
            worker._thread.join()
        if worker._heartbeat_thread and worker._heartbeat_thread.is_alive():
            worker._heartbeat_thread.join(timeout=5)
        if worker._stall_watchdog_thread and worker._stall_watchdog_thread.is_alive():
            worker._stall_watchdog_thread.join(timeout=5)
    finally:
        lock_fd.close()


def _prepare_mount_path(row, ds_type: str, config: dict) -> tuple[str | None, str]:
    """Bereitet den Dateipfad fuer eine Datasource vor (inkl. Mount falls noetig).

    Returns:
        Tuple (path, error_message). path ist None bei Fehler.
    """
    import os

    from kiara.connectors.registry import auto_mount_point as _auto_mount_point_from_row
    from kiara.shared.crypto import decrypt, is_encrypted
    from kiara.shared.enums import ConnectorType
    from kiara.shared.mount import is_mounted

    if ds_type == ConnectorType.SMB:
        from kiara.shared.mount import mount_smb

        mount_point = _auto_mount_point_from_row(row)
        if not is_mounted(mount_point):
            password = row["smb_password"]
            if password and is_encrypted(password):
                password = decrypt(password)
            ok, msg = mount_smb(
                server=row["smb_server"],
                share=row["smb_share"],
                username=row["smb_username"],
                password=password or "",
                domain=row["smb_domain"] or "",
                mount_point=mount_point,
            )
            if not ok:
                return None, f"SMB mount failed: {msg}"
        sub_path = (row["smb_mount_point"] or "").strip("/")
        return (os.path.join(mount_point, sub_path) if sub_path else mount_point), ""

    if ds_type == ConnectorType.WEBDAV:
        from kiara.web.webdav_utils import mount_webdav

        path = _auto_mount_point_from_row(row)
        if not is_mounted(path):
            password = row["webdav_password"]
            if password and is_encrypted(password):
                password = decrypt(password)
            webdav_url = (row["webdav_url"] or "").rstrip("/") + (row["webdav_base_path"] or "/")
            ok, msg = mount_webdav(
                url=webdav_url,
                username=row["webdav_username"] or "",
                password=password or "",
                mount_point=path,
            )
            if not ok:
                return None, f"WebDAV mount failed: {msg}"
        return path, ""

    if ds_type == ConnectorType.LOCAL and row["path"]:
        return row["path"], ""

    return config["datasource"]["path"], ""


@main.command("queue-ingest")
@click.option("--ds-id", type=int, required=True, help="Datasource-ID.")
@click.option("--full", is_flag=True, help="Voll-Ingestion statt inkrementell.")
@click.option(
    "--triggered-by",
    type=click.Choice(["manual", "cron", "api"]),
    default="manual",
    help="Ausloser des Jobs.",
)
@click.option("--embedding-model", required=True, help="Embedding-Modell fuer die Vektorisierung.")
@click.pass_context
def queue_ingest(ctx, ds_id, full, triggered_by, embedding_model):
    """Stellt einen Ingestion-Job in die Queue (non-blocking)."""
    from kiara.shared.db import db_query, init_db_backend
    from kiara.web.ingestion_worker import enqueue_job

    config = _get_config(ctx)
    init_db_backend(config)

    # Datasource pruefen
    with db_query() as db:
        row = db.execute(
            "SELECT id, name, is_active FROM datasources WHERE id = %s", (ds_id,)
        ).fetchone()
    if not row:
        click.echo(f"Fehler: Datenquelle {ds_id} nicht gefunden.", err=True)
        sys.exit(1)
    if not row["is_active"]:
        click.echo(f"Fehler: Datenquelle {ds_id} ({row['name']}) ist deaktiviert.", err=True)
        sys.exit(1)

    # Job einstellen via enqueue_job() (zentrale Validierung + Normalisierung)
    job_type = "full" if full else "incremental"
    try:
        job_id = enqueue_job(ds_id, job_type, triggered_by, embedding_model=embedding_model)
        click.echo(
            f"Job #{job_id} eingestellt: DS {ds_id} ({row['name']}), "
            f"Typ={job_type}, Modell={embedding_model}, Ausloser={triggered_by}"
        )
    except ValueError as exc:
        click.echo(f"Fehler: {exc}", err=True)
        sys.exit(1)


@main.command("init-db")
@click.pass_context
def init_db_cmd(ctx):
    """Initialize database and create default admin user."""
    from kiara.shared.db import init_db_backend
    from kiara.web.database import init_db

    config = _get_config(ctx)
    init_db_backend(config)

    click.echo("Initializing database...")
    init_db()
    click.echo("Database initialized. Default admin user: admin / admin")


@main.command("cleanup-system-events")
@click.option(
    "--days",
    type=click.IntRange(min=1),
    default=None,
    help="Loesche Eintraege aelter als N Tage (Default: retention_days aus Config).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Zaehlt betroffene Eintraege, ohne sie zu loeschen.",
)
@click.pass_context
def cleanup_system_events_cmd(ctx, days, dry_run):
    """Bereinigt alte Eintraege in system_events (Retention-Job)."""
    from kiara.config import get_system_events_config
    from kiara.repositories.system_events import SystemEventRepository
    from kiara.shared.db import init_db_backend

    config = _get_config(ctx)
    init_db_backend(config)

    effective_days = (
        days if days is not None else get_system_events_config(config)["retention_days"]
    )
    repo = SystemEventRepository()

    if dry_run:
        count = repo.count_older_than(effective_days)
        click.echo(
            f"Dry-Run: {count} Eintraege aelter als {effective_days} Tage wuerden geloescht."
        )
        return

    deleted = repo.cleanup_older_than(effective_days)
    click.echo(f"{deleted} Eintraege aelter als {effective_days} Tage geloescht.")


@main.command("log-service-failure")
@click.argument("service_name")
@click.pass_context
def log_service_failure_cmd(ctx, service_name):
    """Loggt einen Service-Failure als SERVICE_DIED-Event in system_events.

    Wird von systemd via OnFailure=kiara-failure-logger@%n.service getriggert,
    wenn ein KIara-Service unkontrolliert stirbt (OOM-Kill, Segfault, nicht
    abgefangene Exception). Liest Exit-Status, journalctl-Tail und dmesg-OOM-
    Hinweise und persistiert sie strukturiert ueber den security-Logger.
    """
    import shutil
    import subprocess

    if not service_name or not service_name.endswith(".service"):
        click.echo("Ungueltiger Service-Name (erwartet: <name>.service)", err=True)
        sys.exit(2)
    safe_name = service_name.replace("\n", "").replace("\r", "")
    if not safe_name.replace(".service", "").replace("-", "").replace("_", "").isalnum():
        click.echo("Service-Name enthaelt unerlaubte Zeichen", err=True)
        sys.exit(2)

    config = _get_config(ctx)
    from kiara.shared.db import init_db_backend

    init_db_backend(config)
    from kiara.web.logging_config import setup_security_logger

    try:
        security_log = setup_security_logger(strict_db=True)
    except Exception as exc:
        click.echo(
            f"SERVICE_DIED-Logger konnte nicht initialisiert werden: {exc}",
            err=True,
        )
        sys.exit(1)

    def _run(args):
        try:
            return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return ""

    show_out = _run(
        [
            "systemctl",
            "show",
            safe_name,
            "-p",
            "Result",
            "-p",
            "ExecMainStatus",
            "-p",
            "ExecMainCode",
            "-p",
            "ActiveEnterTimestamp",
            "-p",
            "MainPID",
        ]
    )
    props = {}
    for line in show_out.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            props[key.strip()] = value.strip()

    journal_tail = _run(["journalctl", "-u", safe_name, "-n", "30", "--no-pager"])

    dmesg_oom = ""
    if shutil.which("dmesg"):
        dmesg_full = _run(["dmesg", "--ctime"])
        oom_lines = [
            line
            for line in dmesg_full.splitlines()[-200:]
            if "out of memory" in line.lower() or "killed process" in line.lower()
        ]
        dmesg_oom = "\n".join(oom_lines[-10:]) if oom_lines else ""

    result = props.get("Result", "unknown")
    exit_code = props.get("ExecMainStatus", "?")
    main_pid = props.get("MainPID", "?")
    is_oom = bool(
        dmesg_oom and (safe_name.removesuffix(".service") in dmesg_oom or main_pid in dmesg_oom)
    )
    reason = "oom_killed" if is_oom else result

    extra_dict = {
        "journal_tail": journal_tail[-4000:],
        "dmesg_oom": dmesg_oom or None,
    }
    try:
        security_log.error(
            "SERVICE_DIED service=%s reason=%s exit_code=%s pid=%s",
            safe_name,
            reason,
            exit_code,
            main_pid,
            extra={"kiara_details": extra_dict},
        )
    except Exception as exc:
        click.echo(f"SERVICE_DIED konnte nicht persistiert werden: {exc}", err=True)
        sys.exit(1)
    click.echo(
        f"SERVICE_DIED {safe_name} reason={reason} exit_code={exit_code} pid={main_pid} logged."
    )


@main.command("prepare-cifs-flight-recorder", hidden=True)
def prepare_cifs_flight_recorder_cmd():
    """Initialisiert den root-only CIFS-Trace-Ringpuffer."""
    import json
    import os

    if os.geteuid() != 0:
        raise click.ClickException("CIFS-Flight-Recorder muss als root gestartet werden")

    from kiara.web.ingestion_forensics import configure_cifs_flight_recorder

    result = configure_cifs_flight_recorder()
    click.echo(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    if not result.get("available"):
        raise click.ClickException("CIFS-Flight-Recorder konnte nicht aktiviert werden")


@main.command("collect-ingestion-stall-forensics", hidden=True)
@click.argument("pid", type=click.IntRange(min=1, max=4_194_304))
@click.argument("tid", type=click.IntRange(min=1, max=4_194_304))
def collect_ingestion_stall_forensics_cmd(pid, tid):
    """Sammelt Root-Forensik fuer einen validierten Ingestion-Worker-Thread."""
    import json
    import os

    if os.geteuid() != 0:
        raise click.ClickException("Ingestion-Stall-Forensik muss als root laufen")

    from kiara.web.ingestion_forensics import (
        ForensicsTargetError,
        collect_ingestion_stall_forensics,
    )

    try:
        result = collect_ingestion_stall_forensics(pid, tid)
    except ForensicsTargetError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


@main.command("log-worker-death")
@click.argument("worker_pid")
@click.argument("service_name")
@click.pass_context
def log_worker_death_cmd(ctx, worker_pid, service_name):
    """Loggt einen uvicorn-Worker-Tod als WORKER_DIED-Event in system_events.

    Wird vom kiara-worker-death-watcher.service via journalctl-Tail
    getriggert, wenn der uvicorn-Master "Child process [PID] died" loggt.
    Im Gegensatz zu log-service-failure: kein Exit-Code/Signal verfuegbar
    (externer Watcher hat keine Parent-Child-Relation zum Worker), nur
    Kontext-Snapshot fuer Korrelation (dmesg-Tail, CIFS-Status, Memory,
    journal-Tail).
    """
    import subprocess

    if not worker_pid.isdigit() or len(worker_pid) > 12:
        click.echo("Ungueltige Worker-PID", err=True)
        sys.exit(2)
    if not service_name or not service_name.endswith(".service"):
        click.echo("Ungueltiger Service-Name", err=True)
        sys.exit(2)
    safe_service = service_name.replace("\n", "").replace("\r", "")
    if not safe_service.replace(".service", "").replace("-", "").replace("_", "").isalnum():
        click.echo("Service-Name enthaelt unerlaubte Zeichen", err=True)
        sys.exit(2)

    config = _get_config(ctx)
    from kiara.shared.db import init_db_backend

    init_db_backend(config)
    from kiara.web.logging_config import setup_security_logger

    try:
        security_log = setup_security_logger(strict_db=True)
    except Exception as exc:
        click.echo(
            f"WORKER_DIED-Logger konnte nicht initialisiert werden: {exc}",
            err=True,
        )
        sys.exit(1)

    def _run(args):
        try:
            return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return ""

    dmesg_tail = ""
    try:
        out = _run(["dmesg", "--ctime"])
        lines = out.splitlines()[-200:]
        cifs_lines = [line for line in lines if "CIFS" in line or "cifs" in line]
        oom_lines = [
            line
            for line in lines
            if "out of memory" in line.lower() or "killed process" in line.lower()
        ]
        snippet_lines = (cifs_lines[-15:] + oom_lines[-10:])[-25:]
        dmesg_tail = "\n".join(snippet_lines) if snippet_lines else ""
    except Exception:
        pass

    cifs_status = _run(["mount", "-t", "cifs"])[:2000]

    cgroup_memory_current = ""
    cgroup_memory_peak = ""
    cgroup_oom_kill = ""
    cgroup_base = f"/sys/fs/cgroup/system.slice/{safe_service}"
    try:
        with open(f"{cgroup_base}/memory.current") as fh:
            cgroup_memory_current = fh.read().strip()
        with open(f"{cgroup_base}/memory.peak") as fh:
            cgroup_memory_peak = fh.read().strip()
        with open(f"{cgroup_base}/memory.events") as fh:
            for line in fh:
                if line.startswith("oom_kill "):
                    cgroup_oom_kill = line.split()[1]
                    break
    except (FileNotFoundError, PermissionError, OSError):
        pass

    journal_tail = _run(["journalctl", "-u", safe_service, "-n", "20", "--no-pager"])

    extra_dict = {
        "service": safe_service,
        "dmesg_tail": dmesg_tail or None,
        "cifs_mounts": cifs_status or None,
        "memory_current": cgroup_memory_current or None,
        "memory_peak": cgroup_memory_peak or None,
        "cgroup_oom_kill": cgroup_oom_kill or None,
        "journal_tail": journal_tail[-3000:] or None,
    }

    cifs_indicator = "tcon failed" in (dmesg_tail or "")
    reason = "cifs_unreachable_correlation" if cifs_indicator else "unknown"

    try:
        security_log.error(
            "WORKER_DIED service=%s pid=%s reason=%s",
            safe_service,
            worker_pid,
            reason,
            extra={"kiara_details": extra_dict},
        )
    except Exception as exc:
        click.echo(f"WORKER_DIED konnte nicht persistiert werden: {exc}", err=True)
        sys.exit(1)
    click.echo(f"WORKER_DIED {safe_service} pid={worker_pid} reason={reason} logged.")


@main.command()
@click.option(
    "--output", "-o", default=None, help=f"Ausgabepfad (Default: {DEFAULT_PATHS['backups']}/)"
)
@click.option(
    "--include-vectordb/--no-vectordb",
    default=False,
    help="File-backed VectorDB-Daten einschliessen (Qdrant wird nicht per Dateipfad gesichert)",
)
@click.pass_context
def backup(ctx, output, include_vectordb):
    """Erstellt ein vollstaendiges Backup: DB + Config + optional VectorDB-Daten."""
    from kiara.backup import create_backup

    config = _get_config(ctx)
    click.echo("Erstelle Backup...")

    if not include_vectordb:
        click.echo("  VectorDB-Daten werden NICHT eingeschlossen.")

    try:
        path = create_backup(config, output_path=output, include_vectordb_data=include_vectordb)
        click.echo(f"Backup erstellt: {path}")
    except Exception as exc:
        click.echo(f"Fehler beim Backup: {exc}", err=True)
        sys.exit(1)


@main.command()
@click.argument("backup_path")
@click.option("--force/--no-force", default=False, help="Bestehende Daten ueberschreiben")
@click.option("--dry-run/--no-dry-run", default=False, help="Nur pruefen, nicht wiederherstellen")
@click.pass_context
def restore(ctx, backup_path, force, dry_run):
    """Stellt ein Backup wieder her."""
    from kiara.backup import restore_backup, validate_backup

    config = _get_config(ctx)

    # Validierung
    click.echo(f"Validiere Backup: {backup_path}")
    is_valid, manifest, issues = validate_backup(backup_path)

    if issues:
        click.echo("Probleme gefunden:")
        for issue in issues:
            click.echo(f"  - {issue}")

    if not is_valid:
        click.echo("Backup-Validierung fehlgeschlagen. Restore abgebrochen.", err=True)
        sys.exit(1)

    if is_valid:
        click.echo("Backup ist gueltig.")
        click.echo(f"  Erstellt: {manifest.get('created_at', 'unbekannt')}")
        click.echo(f"  Dateien: {len(manifest.get('files', {}))}")
        include_vdb = manifest.get("include_vectordb_data", manifest.get("include_chromadb"))
        click.echo(f"  VectorDB-Daten: {'ja' if include_vdb else 'nein'}")

    if dry_run:
        click.echo("\n--- Dry-Run: Keine Aenderungen ---")

    result = restore_backup(backup_path, config, force=force, dry_run=dry_run)

    if result["success"]:
        if dry_run:
            click.echo("Wuerde wiederherstellen:")
            for item in result.get("would_restore", []):
                click.echo(f"  - {item}")
        else:
            click.echo("Wiederherstellung erfolgreich:")
            for item in result.get("restored", []):
                click.echo(f"  - {item}")
    else:
        click.echo(f"Fehler: {result.get('message', 'Unbekannt')}", err=True)
        for issue in result.get("issues", []):
            click.echo(f"  - {issue}", err=True)
        sys.exit(1)


@main.command("migrate-source-key")
@click.pass_context
def migrate_source_key_cmd(ctx):
    """Qdrant-Punkte um source_key-Feld ergaenzen (einmalige Migration)."""
    config = _get_config(ctx)
    backend = config.get("vectordb", {}).get("backend", "qdrant")
    if backend != "qdrant":
        click.echo(f"source_key-Migration nur fuer Qdrant-Backend relevant (aktuell: {backend}).")
        return
    from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

    store = QdrantVectorStore(config, intent="maintenance")
    click.echo(f"Starte source_key-Migration fuer Collection {store._collection}...")
    stats = store.migrate_source_key()
    click.echo(
        f"Fertig: {stats['total']} total, {stats['migrated']} migriert, "
        f"{stats['skipped']} uebersprungen, {stats['errors']} Fehler"
    )
    if stats["errors"] == 0:
        click.echo(
            "Migration erfolgreich. Bitte Services neustarten: systemctl restart kiara kiara-admin"
        )
    else:
        click.echo(
            f"WARNUNG: {stats['errors']} Fehler. Migration erneut ausfuehren oder Fehler pruefen."
        )


@main.command("index-contracts-backfill")
@click.option(
    "--apply",
    is_flag=True,
    help="Evidence-/Contractmetadaten atomar schreiben (Default: read-only Dry-run).",
)
@click.option(
    "--rollback-run-id",
    default=None,
    help="Nur Metadaten dieses Apply-Runs entfernen; Vektoren bleiben read-only.",
)
@click.pass_context
def index_contracts_backfill_cmd(ctx, apply, rollback_run_id):
    """S4-Bestandsaudit/Backfill mit maschinenlesbarer JSON-Ausgabe."""
    import json

    from kiara.retrieval.index_compatibility import IndexCompatibilityError
    from kiara.retrieval.index_contract_backfill import (
        rollback_backfill_metadata,
        run_index_contract_backfill,
    )
    from kiara.shared.db import init_db_backend

    if apply and rollback_run_id:
        raise click.ClickException("--apply und --rollback-run-id sind gegenseitig exklusiv")
    config = _get_config(ctx)
    init_db_backend(config)
    try:
        if rollback_run_id:
            result = rollback_backfill_metadata(config, rollback_run_id)
        else:
            result = run_index_contract_backfill(config, apply=apply)
    except IndexCompatibilityError as exc:
        raise click.ClickException(
            json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True)
        ) from exc
    except Exception as exc:
        raise click.ClickException(f"Index-Contract-Backfill fehlgeschlagen: {exc}") from exc
    click.echo(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, default=str))


@main.command("index-contracts-schema")
@click.option("--model", required=True, help="Freigegebenes lokales Modell (P64/task:189).")
@click.option(
    "--apply", is_flag=True, help="Geprueften Uebergang atomar protokollieren und anwenden."
)
@click.option(
    "--rollback-evidence",
    type=int,
    default=None,
    help="Aktuellen Uebergang anhand seiner Evidence-ID zuruecknehmen.",
)
@click.pass_context
def index_contracts_schema_cmd(ctx, model, apply, rollback_evidence):
    """Begrenzter Schemauebergang; ohne --apply ausschliesslich Vorschau."""
    import json

    from kiara.retrieval.index_compatibility import (
        IndexCompatibilityError,
        stored_primary_index_target,
    )
    from kiara.retrieval.index_schema_transition import (
        rollback_schema_transition,
        transition_schema,
    )
    from kiara.shared.db import init_db

    config = _get_config(ctx)
    # This command operates on an existing binding and must not run unrelated
    # schema migrations, especially during its default preview.
    init_db(config["database"]["postgresql"]["dsn"], force=True)
    try:
        target = stored_primary_index_target(config, model)
        if rollback_evidence is None:
            result = transition_schema(config, target, apply=apply)
        else:
            result = rollback_schema_transition(config, target, rollback_evidence, apply=apply)
    except IndexCompatibilityError as exc:
        raise click.ClickException(json.dumps(exc.to_dict(), ensure_ascii=False)) from exc
    click.echo(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


@main.command("cleanup-colbert")
@click.option("--dry-run", is_flag=True, help="Nur stale ColBERT-Daten anzeigen, nichts loeschen.")
@click.option("--ds-id", type=int, default=None, help="Nur diese Datasource pruefen/bereinigen.")
@click.pass_context
def cleanup_colbert(ctx, dry_run, ds_id):
    """Stale ColBERT-Collections und verwaiste colbert::*-DB-Eintraege bereinigen."""
    from kiara.ingestion._file_utils import ChunkDatasources
    from kiara.ingestion.pipeline import (
        cleanup_stale_colbert_collections_without_db,
        cleanup_stale_colbert_models,
    )
    from kiara.integrity.checks import check_stale_colbert_collections
    from kiara.retrieval.colbert import colbert_ds_model_key, model_name_from_colbert_ds_key
    from kiara.shared.db import db_query, init_db_backend

    config = _get_config(ctx)
    init_db_backend(config)
    chunk_ds = ChunkDatasources()
    detection = check_stale_colbert_collections(config)

    def _filter_to_ds(items):
        if ds_id is None:
            return list(items)
        filtered = []
        for item in items:
            datasources = [d for d in item.get("datasources", []) if d.get("id") == ds_id]
            if datasources:
                filtered.append({**item, "datasources": datasources})
        return filtered

    stale_collections = _filter_to_ds(detection.get("stale_collections", []))
    orphaned_db_entries = _filter_to_ds(detection.get("orphaned_db_entries", []))
    stale_set = {c["model"] for c in stale_collections} | {e["model"] for e in orphaned_db_entries}
    allow_current_models = {
        e["model"] for e in orphaned_db_entries if e.get("reason") == "collection_missing"
    }
    collection_only_stale = [c for c in stale_collections if not c.get("datasources")]

    details = detection.get("details", "ColBERT-Cleanup-Pruefung abgeschlossen")
    if ds_id is not None:
        details = (
            f"DS {ds_id}: {len(stale_collections)} stale Collections, "
            f"{len(orphaned_db_entries)} verwaiste DB-Eintraege"
        )
    click.echo(details)
    if dry_run:
        click.echo(f"Stale Collections: {len(stale_collections)}")
        click.echo(f"Verwaiste DB-Eintraege: {len(orphaned_db_entries)}")
        return
    if detection.get("status") == "warning" and not stale_set:
        click.echo(
            "Cleanup uebersprungen: Detection ist warning und keine stale Modelle sind sicher klassifiziert."
        )
        return

    db_colbert_keys = [
        m for m in chunk_ds.distinct_models() if m and str(m).startswith("colbert::")
    ]
    stale_models = []
    for key in db_colbert_keys:
        try:
            model = model_name_from_colbert_ds_key(key)
        except ValueError:
            continue
        if model in stale_set:
            stale_models.append(model)

    if ds_id is not None:
        stale_models = [
            model
            for model in stale_models
            if chunk_ds.count_by_datasource(ds_id, colbert_ds_model_key(model)) > 0
        ]
        affected_ds_ids = [ds_id] if stale_models else []
    else:
        stale_ds_keys = [colbert_ds_model_key(m) for m in stale_models]
        if not stale_ds_keys:
            affected_ds_ids = []
        else:
            with db_query() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT datasource_id FROM chunk_datasources "
                    "WHERE embedding_model = ANY(%s)",
                    (stale_ds_keys,),
                ).fetchall()
            affected_ds_ids = [r["datasource_id"] for r in rows]

    invoked = 0
    lock_skips = 0
    current_model_skips = 0
    db_only_current_cleanups = 0
    wrapper_errors = 0
    collection_only = cleanup_stale_colbert_collections_without_db(config, collection_only_stale)
    for affected_ds_id in affected_ds_ids:
        result = cleanup_stale_colbert_models(
            config,
            affected_ds_id,
            chunk_ds,
            stale_models,
            allow_current_models=allow_current_models,
        )
        if result["lock_conflict"]:
            lock_skips += 1
        if result["cleanup_invoked"]:
            invoked += 1
        current_model_skips += len(result.get("skipped_current_models", []))
        db_only_current_cleanups += len(result.get("db_only_current_models", []))
        wrapper_errors += len(result.get("errors", []))

    click.echo(
        f"Cleanup gestartet fuer {invoked} Datasource(s), "
        f"Lock-Skips: {lock_skips}, Current-Model-Skips: {current_model_skips}, "
        f"DB-only Current-Model-Cleanups: {db_only_current_cleanups}, "
        f"Wrapper-Fehler: {wrapper_errors}, "
        f"Collection-only geloescht: {len(collection_only.get('deleted', []))}, "
        f"Collection-only Fehler: {len(collection_only.get('errors', []))}"
    )


@main.group("config-snapshot")
def config_snapshot_group():
    """Config-Snapshots verwalten (versionierte kiara.yaml-Staende)."""


@config_snapshot_group.command("list")
def config_snapshot_list_cmd():
    """Snapshots mit Labels, Hashes und verknuepften Benchmark-Laeufen auflisten."""
    from kiara.admin.services.config_snapshot_service import list_snapshots

    snapshots = list_snapshots()
    if not snapshots:
        click.echo("Keine Config-Snapshots vorhanden.")
        return
    for s in snapshots:
        marker = " [AKTIV]" if s.get("is_current") else ""
        runs = ", ".join(s.get("benchmark_runs") or []) or "-"
        click.echo(
            f"{s['timestamp']}  {s['config_sha256'][:8]}  "
            f"{(s.get('label') or '-'):24} trigger={s.get('trigger', '-'):11} "
            f"runs={runs}{marker}"
        )


@config_snapshot_group.command("create")
@click.option("--label", default="", help="Kurzname (a-z, 0-9, . _ -)")
@click.option("--comment", default="", help="Freitext-Kommentar")
def config_snapshot_create_cmd(label, comment):
    """Aktuellen Config-Stand als Snapshot sichern."""
    from kiara.admin.services.config_snapshot_service import ConfigSnapshotError, create_snapshot

    try:
        entry = create_snapshot(label=label, comment=comment, trigger="manual", user="cli")
    except ConfigSnapshotError as exc:
        raise click.ClickException(str(exc))
    click.echo(f"Snapshot angelegt: {entry['file']} (sha {entry['config_sha256'][:8]})")


@config_snapshot_group.command("diff")
@click.argument("ref")
@click.option("--against", default="current", help="Vergleichsziel (Default: current)")
def config_snapshot_diff_cmd(ref, against):
    """Unterschiede eines Snapshots zur aktuellen Config zeigen (Secrets maskiert)."""
    from kiara.admin.services.config_snapshot_service import ConfigSnapshotError, diff_snapshots

    try:
        changes = diff_snapshots(ref, against)
    except ConfigSnapshotError as exc:
        raise click.ClickException(str(exc))
    if not changes:
        click.echo("Keine Unterschiede.")
        return
    for change in changes[1:]:
        click.echo(f"{change['key']}: {change['snapshot']} -> {change['against']}")


@config_snapshot_group.command("restore")
@click.argument("ref")
@click.option("--restart/--no-restart", default=False, help="Services danach neu starten")
@click.confirmation_option(prompt="Aktive kiara.yaml wirklich durch den Snapshot ersetzen?")
def config_snapshot_restore_cmd(ref, restart):
    """Snapshot als aktive Config wiederherstellen (sichert aktuellen Stand vorher)."""
    from kiara.admin.services.config_snapshot_service import (
        ConfigSnapshotError,
        restore_snapshot,
        trigger_service_restart,
    )

    try:
        result = restore_snapshot(ref, user="cli")
    except ConfigSnapshotError as exc:
        raise click.ClickException(str(exc))
    click.echo(f"Wiederhergestellt: {result['restored']['file']}")
    click.echo(f"Vorheriger Stand gesichert als: {result['pre_restore_snapshot']}")
    if restart:
        trigger_service_restart(user="cli")
        click.echo("Service-Restart ausgeloest.")
    else:
        click.echo(
            "Hinweis: Services neu starten, damit die Config wirkt (./deploy.sh oder systemctl)."
        )


if __name__ == "__main__":
    main()
