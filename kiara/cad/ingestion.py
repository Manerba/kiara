"""Route PKG imports to independently selected, independently retained targets."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from functools import wraps
from pathlib import Path

from kiara.shared.db import db_query, db_transaction


def pkg(path):
    return Path(path).suffix.lower() == ".pkg"


def targets(row):
    return {
        key: row.get("ds_" + key, True) for key in ("cad_vector_enabled", "cad_relational_enabled")
    }


def sql_only_source(row, config):
    """Only an explicit PKG-only selection can dispense with vector infrastructure."""
    types = row.get("ds_file_types") or config.get("datasource", {}).get("file_types", [])
    if isinstance(types, str):
        types = json.loads(types)
    return (
        targets(row)["cad_vector_enabled"] is False
        and bool(types)
        and {x.lower() for x in types} == {".pkg"}
        and not row.get("ds_accept_all_text", False)
    )


def job_is_sql_only(row):
    stats = row.get("stats_json") or {}
    if isinstance(stats, str):
        stats = json.loads(stats)
    return stats.get("cad_sql_only") is True


def _version(source_hash):
    from kiara.cad import revision

    return hashlib.sha256(json.dumps([source_hash, revision()]).encode()).hexdigest()


def _audit_version(source_hash):
    """Veröffentlichung mit ungeklärten Feldformen: eigene Version, damit ein
    Lauf mit der Regel skip_file die Datei erneut streng prüft."""
    from kiara.cad import revision

    return hashlib.sha256(json.dumps([source_hash, revision(), "open-fields"]).encode()).hexdigest()


def unknown_fields_audit(config):
    """Schalter ingestion.cad_unknown_fields: skip (Standard) oder audit."""
    mode = config.get("ingestion", {}).get("cad_unknown_fields", "skip")
    if mode not in {"skip", "audit"}:
        raise ValueError(f"Ungültiger Wert für ingestion.cad_unknown_fields: {mode!r}")
    return mode == "audit"


def ingest_relational(
    config,
    datasource_id,
    *,
    incremental=True,
    progress_callback=None,
    control_callback=None,
    file_log_callback=None,
    file_start_callback=None,
    status_callback=None,
    **_,
):
    from kiara_ingestion_core.models import FileLogEntry

    from kiara.cad.adapter import load_pkg
    from kiara.cad.repository import publish
    from kiara.ingestion._file_utils import scan_files

    root = os.path.abspath(config["datasource"]["path"])
    if not os.path.isdir(root):
        raise ValueError("CAD-Datenquellenverzeichnis ist nicht erreichbar")
    files = [p for p in scan_files(config) if pkg(p)]
    audit = unknown_fields_audit(config)
    stats = dict(
        total_files=len(files),
        processed_files=0,
        skipped_files=0,
        errors=0,
        total_chunks=0,
        cad_relational_files=0,
        cad_relational_skipped=0,
        cad_relational_errors=0,
        cad_relational_failed_sources=[],
        cad_error_policy="audit_unknown_fields" if audit else "skip_file",
        cad_audit_files=0,
        cad_audit_findings=0,
    )
    enabled = config.get("ingestion", {}).get("cad_relational_enabled", True)
    if status_callback:
        status_callback(
            "CAD: relationale Daten verarbeiten…" if enabled else "PKG-Dateien überspringen…"
        )
    for index, path in enumerate(files):
        if control_callback and control_callback() in {"cancel", "shutdown"}:
            stats["cancelled"] = True
            break
        started = time.monotonic()
        if file_start_callback:
            file_start_callback(Path(path).name, path)
        entry = FileLogEntry(path, Path(path).name, 0, ".pkg", parser="cad-pkg-sql")
        try:
            entry.file_size = Path(path).stat().st_size
            unchanged = False
            if enabled:
                with open(path, "rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                with db_query() as conn:
                    old = conn.execute(
                        "SELECT m.processing_version FROM cad_models m JOIN cad_model_owners o "
                        "ON o.model_id=m.id WHERE o.datasource_id=%s AND o.source=%s",
                        (datasource_id, path),
                    ).fetchone()
                accepted = {_version(digest)} | ({_audit_version(digest)} if audit else set())
                unchanged = incremental and old and old["processing_version"] in accepted
            if not enabled or unchanged:
                stats["skipped_files"] += 1
                stats["cad_relational_skipped"] += 1
                entry.status = "skipped"
            else:
                snapshot = load_pkg(path, open_fields=audit)
                if snapshot["source_hash"] != digest:
                    raise ValueError("PKG-Datei wurde während des Imports geändert")
                if control_callback and control_callback() in {"cancel", "shutdown"}:
                    stats["cancelled"] = True
                    break
                with db_transaction() as conn:
                    # Serializes writers for this datasource, never locks a vector index.
                    conn.execute("SELECT pg_advisory_xact_lock(%s,%s)", (1128350803, datasource_id))
                    publish(
                        conn,
                        "",
                        path,
                        "",
                        _version(digest) if snapshot["verification_complete"] else _audit_version(digest),
                        snapshot,
                        datasource_id=datasource_id,
                    )
                stats["processed_files"] += 1
                stats["cad_relational_files"] += 1
                findings = snapshot["open_findings"]
                if findings:
                    # Audit: veröffentlicht, Fundstellen bleiben im Dateilog nachvollziehbar.
                    stats["cad_audit_files"] += 1
                    stats["cad_audit_findings"] += len(findings)
                    entry.error_phase = "cad_audit"
                    entry.error_message = (
                        f"Audit: {len(findings)} ungeklärte Ingest-Felder, Datei veröffentlicht: "
                        + json.dumps(findings[:32], ensure_ascii=True)
                        + (f" (+{len(findings) - 32} weitere)" if len(findings) > 32 else "")
                    )
        except Exception as exc:
            # Reject this file atomically, retain its previous publication, and
            # keep processing the remaining source. The file log carries details.
            entry.status, entry.error_phase, entry.error_message = (
                "error",
                "cad_relational",
                str(exc),
            )
            stats["errors"] += 1
            stats["cad_relational_errors"] += 1
            stats["cad_relational_failed_sources"].append(path)
        entry.duration_ms = int((time.monotonic() - started) * 1000)
        if file_log_callback:
            file_log_callback(entry)
        if progress_callback:
            progress_callback(index + 1, len(files))
    if enabled and not stats.get("errors") and not stats.get("cancelled"):
        with db_transaction() as conn:
            existing = conn.execute(
                "SELECT source FROM cad_model_owners WHERE datasource_id=%s",
                (datasource_id,),
            ).fetchall()
            current = set(files)
            removed = [
                r["source"]
                for r in existing
                if pkg(r["source"])
                and os.path.commonpath([root, os.path.abspath(r["source"])]) == root
                and r["source"] not in current
            ]
            if removed:
                conn.execute(
                    "DELETE FROM cad_model_owners WHERE datasource_id=%s AND source=ANY(%s)",
                    (datasource_id, removed),
                )
                conn.execute(
                    "DELETE FROM cad_models m WHERE source=ANY(%s) AND NOT EXISTS "
                    "(SELECT 1 FROM cad_model_owners o WHERE o.model_id=m.id)",
                    (removed,),
                )
            stats["cad_relational_removed"] = len(removed)
    return stats


def route_cad_ingestion(function):
    """SQL has its own completion hashes; disabling a target never clears it."""

    @wraps(function)
    def wrapped(config=None, *args, **kwargs):
        # The established API is keyword-based beyond config. Preserve old positional callers.
        if args:
            import inspect

            bound = inspect.signature(function).bind(config, *args, **kwargs)
            kwargs = dict(bound.arguments)
            config = kwargs.pop("config", None)
        if config is None:
            from kiara.config import load_config

            config = load_config()
            config["_kiara_enforce_persisted_colbert_model"] = True
        config = copy.deepcopy(config)
        overrides = kwargs.get("ds_overrides") or {}
        ing = config.setdefault("ingestion", {})
        for key in ("cad_vector_enabled", "cad_relational_enabled"):
            ing[key] = overrides.get(key, ing.get(key, True))
        scan = config.setdefault("datasource", {})
        for key in ("file_types", "exclude_patterns", "accept_all_text"):
            if key in overrides:
                scan[key] = overrides[key]
        if scan.get("file_types"):
            scan["file_types"] = [".pkg" if x.lower() == ".pkg" else x for x in scan["file_types"]]
            if "file_types" in overrides:
                overrides = {**overrides, "file_types": scan["file_types"]}
                kwargs["ds_overrides"] = overrides
        cad_only = (
            ing["cad_vector_enabled"] is False
            and {x.lower() for x in scan.get("file_types", [])} == {".pkg"}
            and not scan.get("accept_all_text")
        )
        ds_id = kwargs.get("datasource_id")
        pkg_selected = "file_types" in scan and (
            not scan["file_types"] or ".pkg" in {x.lower() for x in scan["file_types"]}
        )
        if ds_id is None and pkg_selected:
            from kiara.ingestion.pipeline import _ensure_cli_datasource

            ds_id = kwargs["datasource_id"] = _ensure_cli_datasource()
        if not overrides.get("index_attachments", True):
            scan["exclude_patterns"] = [*scan.get("exclude_patterns", []), "*/attachments/*"]
        sql_stats = None
        # HTML migrations must remain bounded to their explicit selection.
        if (
            pkg_selected
            and not config.get("_html_migration")
            and not config.get("_html_reingest")
            and (ing["cad_relational_enabled"] or cad_only)
            and ds_id is not None
        ):
            sql_kwargs = {k: v for k, v in kwargs.items() if k != "datasource_id"}
            # Successful both-path imports have one vector file log. Failed SQL
            # files are logged here and excluded from vector cleanup/publication.
            if not cad_only and ing["cad_vector_enabled"]:
                log = sql_kwargs.get("file_log_callback")
                if log is not None:

                    def log_sql_error(entry):
                        if entry.status == "error":
                            log(entry)

                    sql_kwargs["file_log_callback"] = log_sql_error
            sql_stats = ingest_relational(config, ds_id, **sql_kwargs)
            if cad_only or sql_stats.get("cancelled"):
                return sql_stats
            config["_cad_failed_sources"] = sql_stats["cad_relational_failed_sources"]
        result = function(config, **kwargs)
        if sql_stats:
            for key in (
                "cad_relational_files",
                "cad_relational_skipped",
                "cad_relational_errors",
                "cad_relational_failed_sources",
                "cad_error_policy",
                "cad_audit_files",
                "cad_audit_findings",
            ):
                result[key] = sql_stats[key]
            if not ing["cad_vector_enabled"]:
                for key in ("total_files", "processed_files", "skipped_files", "errors"):
                    result[key] = result.get(key, 0) + sql_stats.get(key, 0)
            else:
                # Failed SQL files were excluded from the vector scan and have
                # exactly one error record; count them once in the combined job.
                result["total_files"] += sql_stats["cad_relational_errors"]
                result["errors"] += sql_stats["cad_relational_errors"]
        return result

    return wrapped
