"""Begrenzte HTML-/Mail-Bestandsmigration fuer Projekt 64 ueber bestehende Jobs."""

import copy
import json
import os
from pathlib import Path

from kiara.backend_selector import normalize_model_name

MIGRATION_REVISION = "html-markdown-p64-329-v1"
MIGRATION_MODELS = ("nomic-embed-text", "hellord/e5-mistral-7b-instruct:q4_0")
_SELECTION_KEYS = {"project_id", "revision", "datasource_id", "models", "files_by_model"}
REINGEST_SELECTION_REVISION = "html-reingest-selection-v1"
_REINGEST_SELECTION_KEYS = _SELECTION_KEYS - {"project_id"}


def build_reingest_selection(datasource_id: int, model: str, paths: list[str]) -> dict:
    """Normale HTML-Bestandsaktion; auch eine explizit leere Auswahl ist verbindlich."""
    if isinstance(datasource_id, bool) or not isinstance(datasource_id, int) or datasource_id <= 0:
        raise ValueError("Ungueltige HTML-Datasource")
    if not isinstance(model, str) or normalize_model_name(model) not in MIGRATION_MODELS:
        raise ValueError("Modell nicht fuer HTML-Neuaufbereitung freigegeben")
    if not isinstance(paths, list):
        raise ValueError("HTML-Auswahl braucht eine explizite Pfadliste")
    model = normalize_model_name(model)
    return {
        "revision": REINGEST_SELECTION_REVISION,
        "datasource_id": datasource_id,
        "models": [model],
        "files_by_model": {model: sorted({_absolute_path(path) for path in paths})},
    }


def selection_spec(value: dict) -> dict:
    """Nur die validierte unveraenderliche Auswahl aus einem Lauf uebernehmen."""
    keys = (
        _REINGEST_SELECTION_KEYS
        if value.get("revision") == REINGEST_SELECTION_REVISION
        else _SELECTION_KEYS
    )
    return validate_selection({key: value[key] for key in keys})


def _stats(value) -> dict:
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Migrationsstatistik muss ein Objekt sein")
    return value


def _absolute_path(value) -> str:
    if (
        not isinstance(value, str)
        or not Path(value).is_absolute()
        or os.path.abspath(value) != value
    ):
        raise ValueError("Migrationspfade muessen absolute normalisierte Bestandspfade sein")
    return value


def build_selection(datasource_id: int, files_by_model: dict) -> dict:
    """Explizite Dateiauswahl kanonisieren; niemals auf weitere Modelle erweitern."""
    if isinstance(datasource_id, bool) or not isinstance(datasource_id, int) or datasource_id <= 0:
        raise ValueError("Ungueltige Migrations-Datasource")
    if not isinstance(files_by_model, dict) or not files_by_model:
        raise ValueError("Migration braucht explizite Modell-/Dateiauswahl")
    normalized = {}
    for raw_model, paths in files_by_model.items():
        if not isinstance(raw_model, str):
            raise ValueError("Ungueltiges Migrationsmodell")
        model = normalize_model_name(raw_model)
        if model not in MIGRATION_MODELS or model in normalized:
            raise ValueError(f"Modell nicht eindeutig im Projektumfang: {raw_model}")
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"Modell {model} braucht bekannte Bestandspfade")
        normalized[model] = sorted({_absolute_path(path) for path in paths})
    models = [model for model in MIGRATION_MODELS if model in normalized]
    return {
        "project_id": 64,
        "revision": MIGRATION_REVISION,
        "datasource_id": datasource_id,
        "models": models,
        "files_by_model": {model: normalized[model] for model in models},
    }


def validate_selection(value: dict, datasource_id=None, model=None) -> dict:
    if isinstance(value, dict) and value.get("revision") == REINGEST_SELECTION_REVISION:
        if (
            set(value) != _REINGEST_SELECTION_KEYS
            or not isinstance(value["models"], list)
            or len(value["models"]) != 1
        ):
            raise ValueError("Ungueltige HTML-Bestandsauswahl")
        selected_model = value["models"][0]
        if (
            not isinstance(selected_model, str)
            or not isinstance(value["files_by_model"], dict)
            or set(value["files_by_model"]) != {selected_model}
        ):
            raise ValueError("Modellliste widerspricht HTML-Dateiauswahl")
        selected = build_reingest_selection(
            value["datasource_id"], selected_model, value["files_by_model"][selected_model]
        )
        if datasource_id is not None and selected["datasource_id"] != datasource_id:
            raise ValueError("HTML-Auswahl gehoert zu anderer Datenquelle")
        if model is not None and normalize_model_name(model) not in selected["models"]:
            raise ValueError("Jobmodell gehoert nicht zur HTML-Auswahl")
        return selected
    if not isinstance(value, dict) or set(value) != _SELECTION_KEYS:
        raise ValueError("Unvollstaendige oder erweiterte Migrationsauswahl")
    if value["project_id"] != 64 or value["revision"] != MIGRATION_REVISION:
        raise ValueError("Unbekanntes Migrationsprojekt oder Revision")
    selection = build_selection(value["datasource_id"], value["files_by_model"])
    if not isinstance(value["models"], list) or not all(
        isinstance(item, str) for item in value["models"]
    ):
        raise ValueError("Ungueltige Modellliste der Migration")
    models = [normalize_model_name(item) for item in value["models"]]
    if len(models) != len(set(models)) or set(models) != set(selection["models"]):
        raise ValueError("Modellliste widerspricht Dateiauswahl")
    if datasource_id is not None and selection["datasource_id"] != datasource_id:
        raise ValueError("Migrationsauswahl gehoert zu anderer Datenquelle")
    if model is not None and normalize_model_name(model) not in selection["models"]:
        raise ValueError("Jobmodell gehoert nicht zur Migrationsauswahl")
    return selection


def validate_run_selection(config: dict, datasource_id, model: str) -> dict | None:
    """Vor Scan/Guards nur die immutable Auswahl plus begrenzte Aufbereitung zulassen."""
    if "_html_migration" not in config:
        return None
    value = config["_html_migration"]
    keys = (
        _REINGEST_SELECTION_KEYS
        if isinstance(value, dict) and value.get("revision") == REINGEST_SELECTION_REVISION
        else _SELECTION_KEYS
    )
    if not isinstance(value, dict) or set(value) != keys | {"paths", "errors"}:
        raise ValueError("Ungueltige Migrationslauf-Auswahl")
    selection = validate_selection({key: value[key] for key in keys}, datasource_id, model)
    model = normalize_model_name(model)
    paths, errors = value["paths"], value["errors"]
    if not isinstance(paths, list) or not isinstance(errors, dict):
        raise ValueError("Migrationspfade/-fehler muessen Liste/Objekt sein")
    successful = {_absolute_path(path) for path in paths}
    failed = {_absolute_path(path) for path in errors}
    expected = set(selection["files_by_model"][model])
    if (
        len(successful) != len(paths)
        or successful & failed
        or successful | failed != expected
        or not all(isinstance(error, str) and error for error in errors.values())
    ):
        raise ValueError("Aufbereitete Pfade und Fehler widersprechen der Jobauswahl")
    return {**selection, "paths": list(paths), "errors": copy.deepcopy(errors)}


def selection_from_stats(stats, datasource_id=None, model=None) -> dict | None:
    stats = _stats(stats)
    if "html_migration" not in stats:
        return None
    return validate_selection(stats["html_migration"], datasource_id, model)


def preserve_selection(db, job_id: int, stats: dict) -> None:
    """Persistierte Auswahl auch bei fruehen Fehlern/Abbruch unveraendert erhalten."""
    row = db.execute("SELECT stats_json FROM ingestion_jobs WHERE id = %s", (job_id,)).fetchone()
    existing = _stats(row["stats_json"]) if row else {}
    from kiara.ingestion.html_reingest import request_fields

    for key, value in request_fields(existing).items():
        if key == "html_reingest_summary":
            stats.setdefault(key, value)
        else:
            stats[key] = value
    selection = selection_from_stats(existing)
    if selection is not None:
        stats["html_migration"] = selection
        if "html_migration_errors" in existing:
            stats.setdefault(
                "html_migration_errors", copy.deepcopy(existing["html_migration_errors"])
            )


def migration_completion(selection: dict, jobs: list[dict]) -> dict:
    """Gesamtabschluss benoetigt dieselbe Auswahl und jede Datei in jedem Zielmodell."""
    selection = validate_selection(selection)
    latest = {}
    ordered = sorted(enumerate(jobs), key=lambda item: item[1].get("id", item[0]))
    for _index, job in ordered:
        model = normalize_model_name(str(job.get("embedding_model") or ""))
        if model not in selection["models"]:
            continue
        stats = _stats(job.get("stats_json"))
        if stats.get("html_migration") == selection:
            latest[model] = job
    completed = set()
    for model, job in latest.items():
        stats = _stats(job.get("stats_json"))
        if (
            job.get("status") != "success"
            or stats.get("errors")
            or stats.get("abort_reason")
            or stats.get("html_migration_errors")
        ):
            continue
        paths = stats.get("html_migration_completed_paths", [])
        if isinstance(paths, list) and set(paths) == set(selection["files_by_model"][model]):
            completed.add(model)
    pending = [model for model in selection["models"] if model not in completed]
    return {"complete": not pending, "pending_models": pending}


def prepare_files(
    config, connector_cls, conn_config, row, model, progress_callback, control_callback
):
    """Vor bestehendem Pipeline-Cleanup Originale aufbereiten und Konvertierung pruefen."""
    selection = validate_run_selection(config, row["id"], model)
    paths = selection["files_by_model"][normalize_model_name(model)]
    root = Path(connector_cls.effective_path(conn_config, row)).resolve()
    config["datasource"]["path"] = str(root)
    from kiara.ingestion.loader import load_file

    result = {"files": [], "errors": {}, "cancelled": False}
    mail_paths = []
    if hasattr(connector_cls, "reprocess_cached"):
        mail_paths = [path for path in paths if Path(path).suffix.lower() not in (".html", ".htm")]
        if mail_paths:
            result = connector_cls.reprocess_cached(
                conn_config,
                row,
                config,
                cache_paths=mail_paths,
                progress_callback=(
                    (lambda done, _total: progress_callback(done, len(paths)))
                    if progress_callback
                    else None
                ),
                control_callback=control_callback,
            )
    html_paths = [path for path in paths if path not in set(mail_paths)]
    for index, path_text in enumerate(html_paths):
        if result["cancelled"] or (
            control_callback and control_callback() in ("cancel", "shutdown")
        ):
            result["cancelled"] = True
            break
        try:
            path = Path(path_text)
            if (
                not path.is_file()
                or path.resolve() != path
                or not path.is_relative_to(root)
                or path.suffix.lower() not in (".html", ".htm")
            ):
                raise ValueError("HTML-Datei ist kein bestehender Pfad der Datenquelle")
            document = load_file(path_text, ddu_enabled=False)
            if not document or not document.get("text", "").strip():
                raise ValueError("HTML-Konvertierung liefert keinen indexierbaren Inhalt")
            if document["metadata"].get("parser") != "html-to-markdown":
                raise ValueError("HTML-Konvertierung verwendet nicht das freigegebene Profil")
            result["files"].append(path_text)
        except Exception as exc:
            result["errors"][path_text] = str(exc)
        if progress_callback:
            progress_callback(len(mail_paths) + index + 1, len(paths))
    successful = set(result["files"])
    result["files"] = [path for path in paths if path in successful]
    if not result["cancelled"]:
        config["_html_migration"] = {
            **{key: selection[key] for key in _SELECTION_KEYS},
            "paths": result["files"],
            "errors": result["errors"],
        }
        validate_run_selection(config, row["id"], model)
    return result


def preview_migration(
    config: dict,
    datasource_ids: list[int],
    *,
    paths=None,
    limit=None,
    models=None,
    control_callback=None,
    progress_callback=None,
) -> dict:
    """Bekannte Dateien modellbezogen lesen; ausgeschlossene Bestaende nur benennen."""
    from kiara.connectors.registry import ensure_loaded
    from kiara.connectors.registry import get as get_connector
    from kiara.ingestion._file_utils import ChunkDatasources, HashStore, _CancelledError
    from kiara.ingestion.datasource_runtime import load_connector_config
    from kiara.repositories.file_hash import FileHashRepository
    from kiara.retrieval.vectorstore import VectorStore
    from kiara.shared.constants import QDRANT_DELETE_BATCH_SIZE
    from kiara.shared.db import db_query

    def checkpoint():
        if control_callback and control_callback() in {"cancel", "shutdown"}:
            raise _CancelledError()

    def source_batch(scanned):
        checkpoint()
        if progress_callback:
            progress_callback(scanned, 0)

    read_options = (
        {"batch_callback": source_batch}
        if control_callback is not None or progress_callback is not None
        else {}
    )
    if not datasource_ids or (limit is not None and limit <= 0):
        raise ValueError("Vorschau braucht Datenquellen und ein positives Dateilimit")
    explicit_paths = {_absolute_path(path) for path in paths} if paths else None
    if models is not None and (
        not isinstance(models, list)
        or not models
        or any(model not in MIGRATION_MODELS for model in models)
    ):
        raise ValueError("Ungueltiger Modellumfang der HTML-Vorschau")
    report = {
        "project_id": 64,
        "revision": MIGRATION_REVISION,
        "selections": [],
        "excluded": [],
        "unresolved": [],
        "limited": bool(paths or limit),
        "inventory_scope": "selected_paths" if explicit_paths is not None else "datasources",
    }
    found_explicit = set()
    ensure_loaded()
    for ds_id in dict.fromkeys(datasource_ids):
        checkpoint()
        with db_query() as db:
            row = db.execute("SELECT * FROM datasources WHERE id = %s", (ds_id,)).fetchone()
            inventory = db.execute(
                "SELECT embedding_model FROM datasource_embeddings WHERE datasource_id = %s "
                "UNION SELECT embedding_model FROM file_hashes WHERE datasource_id = %s "
                "UNION SELECT embedding_model FROM chunk_datasources WHERE datasource_id = %s",
                (ds_id, ds_id, ds_id),
            ).fetchall()
        if not row:
            raise ValueError(f"Datenquelle {ds_id} fehlt")
        row = dict(row)
        connector = get_connector(row["type"])
        if connector is None:
            raise ValueError(f"Unbekannter Connector fuer Datenquelle {ds_id}")
        conn_config = load_connector_config(row, connector)
        root = Path(connector.effective_path(conn_config, row)).resolve()
        registry_paths = None
        if hasattr(connector, "reprocess_cached"):
            from kiara.ingestion.mail_migration import _registry_entries

            registry_paths = set(_registry_entries(root))
        chosen = {}
        for item in inventory:
            checkpoint()
            stored_model = item["embedding_model"]
            model = normalize_model_name(stored_model)
            if models is not None and model not in models:
                continue
            if model not in MIGRATION_MODELS:
                report["excluded"].append({"datasource_id": ds_id, "model": stored_model})
                continue
            try:
                store = VectorStore(config, embedding_model=model, intent="maintenance")
                checkpoint()
                chunks = ChunkDatasources()
                if explicit_paths is None:
                    known = set(HashStore().load(ds_id, stored_model))
                    checkpoint()
                    known.update(
                        entry["source"]
                        for entry in store.list_sources_for_datasource(ds_id, **read_options)
                    )
                    checkpoint()
                    ids = chunks.get_chunk_ids(ds_id, stored_model)
                else:
                    selected_sources = sorted(explicit_paths)
                    hashes = FileHashRepository()
                    with db_query() as db:
                        known = {
                            path
                            for path in selected_sources
                            if hashes.get_hash(ds_id, stored_model, path, conn=db) is not None
                        }
                    known.update(
                        entry["source"]
                        for entry in store.list_sources_for_datasource(
                            ds_id, sources=selected_sources, **read_options
                        )
                    )
                    checkpoint()
                    # Fehlende Payload-ds_ids duerfen die bestaetigte SoT-Zuordnung
                    # nicht verstecken. Die Registry allein belegt keine Zugehoerigkeit.
                    candidates = store.get_ids_by_sources(selected_sources)
                    ownership = chunks.get_datasource_ids_for_chunks(candidates, stored_model)
                    ids = [chunk for chunk in candidates if ds_id in ownership.get(chunk, [])]
                missing = []
                for start in range(0, len(ids), QDRANT_DELETE_BATCH_SIZE):
                    checkpoint()
                    batch = ids[start : start + QDRANT_DELETE_BATCH_SIZE]
                    records = store.get_documents_by_ids(batch, include=["metadatas"])
                    resolved = {}
                    for chunk_id, metadata in zip(
                        records["ids"], records["metadatas"], strict=True
                    ):
                        if metadata.get("source"):
                            resolved[chunk_id] = metadata["source"]
                    known.update(resolved.values())
                    missing.extend(set(batch) - set(resolved))
                    if progress_callback:
                        progress_callback(start + len(batch), len(ids))
                checkpoint()
                if missing:
                    report["unresolved"].append(
                        {"datasource_id": ds_id, "model": model, "chunk_ids": sorted(set(missing))}
                    )
                selected = sorted(
                    path
                    for path in known
                    if isinstance(path, str)
                    and path
                    and (
                        Path(path).suffix.lower() in (".html", ".htm")
                        or (registry_paths is not None and path in registry_paths)
                    )
                    and (explicit_paths is None or path in explicit_paths)
                )
                found_explicit.update(selected)
                if limit is not None:
                    selected = selected[:limit]
                if selected:
                    chosen.setdefault(model, []).extend(selected)
            except _CancelledError:
                raise
            except Exception as exc:
                report["unresolved"].append(
                    {"datasource_id": ds_id, "model": model, "error": str(exc)}
                )
        if chosen:
            report["selections"].append(build_selection(ds_id, chosen))
    if explicit_paths:
        report["unresolved"].extend(
            {"path": path, "error": "Expliziter Pfad ohne freigegebenen Modellbestand"}
            for path in sorted(explicit_paths - found_explicit)
        )
    return report


def enqueue_next(selection: dict) -> dict:
    """Genau den naechsten offenen Modelljob einreihen; vorhandene Queue bleibt zustaendig."""
    from kiara.config import load_config
    from kiara.connectors.registry import ensure_loaded
    from kiara.connectors.registry import get as get_connector
    from kiara.ingestion._file_utils import HashStore, compute_file_hash
    from kiara.ingestion.datasource_runtime import build_datasource_overrides
    from kiara.ingestion.processing_revision import processing_file_hash
    from kiara.shared.db import db_query
    from kiara.web.ingestion_worker import enqueue_job

    selection = validate_selection(selection)
    ds_id = selection["datasource_id"]
    with db_query() as db:
        jobs = db.execute(
            "SELECT id, embedding_model, status, stats_json FROM ingestion_jobs "
            "WHERE datasource_id = %s ORDER BY id",
            (ds_id,),
        ).fetchall()
        row = db.execute("SELECT * FROM datasources WHERE id = %s", (ds_id,)).fetchone()
    if not row:
        raise ValueError(f"Datenquelle {ds_id} fehlt")
    ensure_loaded()
    connector_cls = get_connector(row["type"])
    if not connector_cls:
        raise ValueError(f"Unbekannter Datenquellen-Typ: {row['type']}")
    # Scoped Replay nutzt den aktuellen Connectormodus, ohne hash_mode_meta umzuschreiben.
    hash_mode = getattr(connector_cls, "HASH_MODE", "mtime")
    completion = migration_completion(selection, jobs)
    config = copy.deepcopy(load_config())
    overrides = build_datasource_overrides(dict(row))
    for key in ("chunk_size", "chunk_overlap"):
        if overrides.get(key) is not None:
            config.setdefault("ingestion", {})[key] = overrides[key]
    hashes = HashStore()
    pending = set(completion["pending_models"])
    for model in selection["models"]:
        if model in pending:
            continue
        confirmed = hashes.load(ds_id, model)
        try:
            if any(
                confirmed.get(path)
                != processing_file_hash(config, path, compute_file_hash(path, hash_mode))
                for path in selection["files_by_model"][model]
            ):
                pending.add(model)
        except (OSError, ValueError):
            pending.add(model)
    completion = {
        "complete": not pending,
        "pending_models": [model for model in selection["models"] if model in pending],
    }
    if completion["complete"]:
        return {"datasource_id": ds_id, "complete": True, "job_id": None}
    model = completion["pending_models"][0]
    job_id = enqueue_job(
        ds_id,
        "incremental",
        "manual",
        embedding_model=model,
        html_migration=selection,
    )
    return {"datasource_id": ds_id, "complete": False, "model": model, "job_id": job_id}
