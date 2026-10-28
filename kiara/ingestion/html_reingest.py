"""HTML-Bestandsauswahl und Vorbereitung innerhalb eines normalen Ingestion-Jobs."""

import copy
from pathlib import Path

from kiara.ingestion.html_migration import (
    MIGRATION_MODELS,
    _absolute_path,
    _stats,
    build_reingest_selection,
    preview_migration,
)

REQUEST_REVISION = "html-reingest-v1"
SUPPORTED_CONNECTORS = {"local", "smb", "webdav", "imap", "exchange_online"}


def build_request(mode: str) -> dict:
    from kiara_document_loaders.html import conversion_revision

    from kiara.ingestion.markdown_chunks import HTML_CHUNKING_REVISION

    if not isinstance(mode, str) or mode not in {"outdated", "all"}:
        raise ValueError("HTML-Modus muss outdated oder all sein")
    return {
        "revision": REQUEST_REVISION,
        "mode": mode,
        "preparation_revision": list(conversion_revision()),
        "chunking_revision": HTML_CHUNKING_REVISION,
    }


def validate_request(value, *, check_current=False) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "revision",
        "mode",
        "preparation_revision",
        "chunking_revision",
    }:
        raise ValueError("Ungueltiger HTML-Jobauftrag")
    current = build_request(value["mode"])
    if (
        value["revision"] != REQUEST_REVISION
        or not isinstance(value["preparation_revision"], list)
        or len(value["preparation_revision"]) != 2
        or not all(isinstance(x, str) and x for x in value["preparation_revision"])
        or not isinstance(value["chunking_revision"], str)
        or not value["chunking_revision"]
    ):
        raise ValueError("Ungueltige HTML-Jobrevision")
    if check_current and value != current:
        raise ValueError("HTML-Revision seit Jobstart geaendert; neuen HTML-Job starten")
    return copy.deepcopy(value)


def request_from_stats(stats) -> dict | None:
    value = _stats(stats)
    if "html_reingest_candidates" in value and "html_reingest" not in value:
        raise ValueError("HTML-Kandidaten ohne Jobauftrag")
    return validate_request(value["html_reingest"]) if "html_reingest" in value else None


def request_fields(stats) -> dict:
    """Auftrag und Kandidaten auch vor fertiger Klassifizierung wiederaufnehmen."""
    value = _stats(stats)
    request = request_from_stats(value)
    if request is None:
        return {}
    result = {"html_reingest": request}
    if "html_reingest_candidates" in value:
        result["html_reingest_candidates"] = validate_candidates(value["html_reingest_candidates"])
    if "html_reingest_summary" in value:
        result["html_reingest_summary"] = copy.deepcopy(value["html_reingest_summary"])
    return result


def validate_candidates(value) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("HTML-Kandidaten muessen eine gespeicherte Liste sein")
    paths = [_absolute_path(path) for path in value]
    if len(set(paths)) != len(paths):
        raise ValueError("Doppelte HTML-Kandidaten")
    return paths


def validate_target(row: dict, model: str) -> None:
    if row["type"] not in SUPPORTED_CONNECTORS:
        raise ValueError("Datenquellentyp unterstuetzt keine HTML-Bestandsaufbereitung")
    if model not in MIGRATION_MODELS:
        raise ValueError("HTML-Neuaufbereitung ist nur fuer Nomic und Hellord freigegeben")


def collect_candidates(
    config: dict, row: dict, model: str, *, control_callback=None, progress_callback=None
) -> list[str]:
    validate_target(row, model)
    report = preview_migration(
        config,
        [row["id"]],
        models=[model],
        control_callback=control_callback,
        progress_callback=progress_callback,
    )
    if report["unresolved"]:
        raise ValueError("HTML-Bestand nicht eindeutig aufloesbar: " + str(report["unresolved"]))
    return sorted(
        {
            path
            for selection in report["selections"]
            for path in selection["files_by_model"].get(model, [])
        }
    )


def assert_exclusive_sources(
    store, chunk_ds, datasource_id, model, paths, *, control_callback=None, progress_callback=None
):
    """Der begrenzte Job darf keinen gemeinsam genutzten Bestand invalidieren."""
    from kiara.ingestion._file_utils import _CancelledError
    from kiara.shared.constants import QDRANT_DELETE_BATCH_SIZE

    for index, path in enumerate(paths):
        if control_callback and control_callback() in {"cancel", "shutdown"}:
            raise _CancelledError()
        ids = store.get_ids_by_sources([path])
        for start in range(0, len(ids), QDRANT_DELETE_BATCH_SIZE):
            if control_callback and control_callback() in {"cancel", "shutdown"}:
                raise _CancelledError()
            batch = ids[start : start + QDRANT_DELETE_BATCH_SIZE]
            foreign = set(chunk_ds.get_other_datasources(batch, datasource_id, model))
            records = store.get_documents_by_ids(batch, include=["metadatas"])
            if set(records["ids"]) != set(batch):
                raise ValueError(f"HTML-Eigentuemerpruefung unvollstaendig: {path}")
            for metadata in records["metadatas"]:
                foreign.update(set(metadata.get("ds_ids", [])) - {datasource_id})
            if foreign:
                raise ValueError(
                    f"HTML-Datei wird auch von anderen Datenquellen verwendet: {path}; DS {sorted(foreign)}"
                )
        if progress_callback:
            progress_callback(index + 1, len(paths))


def prepared_source_hash(config, path):
    from kiara.ingestion._file_utils import compute_file_hash
    from kiara.ingestion.processing_revision import processing_file_hash

    return processing_file_hash(config, path, compute_file_hash(path, "content"))


def validate_prepared_sources(config, paths):
    """Nach der Vorpruefung veraenderte Eingaben vor jeglichem Index-Cleanup abweisen."""
    if config.get("_html_reingest") is None:
        return
    expected = config.get("_html_reingest_prepared_hashes")
    if not isinstance(expected, dict) or set(expected) != set(paths):
        raise ValueError("HTML-Vorpruefung deckt die Cleanup-Auswahl nicht ab")
    for path in paths:
        if prepared_source_hash(config, path) != expected[path]:
            raise ValueError(f"HTML-Quelle seit Vorpruefung geaendert: {path}")


def _preflight_document(path: str, config: dict, row: dict) -> None:
    from kiara.config import get_mail_archive_config
    from kiara.ingestion.datasource_runtime import build_datasource_overrides
    from kiara.ingestion.loader import load_file
    from kiara.ingestion.runtime import _determine_chunks

    doc = load_file(path, ddu_enabled=False, mail_archive=get_mail_archive_config(config))
    if not doc or not doc.get("text", "").strip():
        raise ValueError("HTML-Aufbereitung liefert keinen indexierbaren Inhalt")
    if (
        doc["metadata"].get("parser") != "html-to-markdown"
        or doc["metadata"].get("source_mime_type") != "text/html"
    ):
        raise ValueError("HTML-Herkunft der aufbereiteten Datei nicht bestaetigt")
    ingestion = config.get("ingestion", {})
    chunks = _determine_chunks(
        doc,
        build_datasource_overrides(row),
        ingestion.get("chunk_size", 1000),
        ingestion.get("chunk_overlap", 200),
    )
    if not chunks:
        raise ValueError("HTML-Aufbereitung liefert keine Chunks")
    for chunk in chunks:
        start, end = chunk["char_start"], chunk["char_end"]
        if (
            chunk["text"] != doc["text"][start:end]
            or len(chunk["text"]) > chunk["metadata"]["chunk_size_used"]
        ):
            raise ValueError("HTML-Chunkposition oder Budget ungueltig")


def prepare_reingestion(
    config,
    connector_cls,
    conn_config,
    row,
    model,
    request,
    candidates,
    stored_hashes,
    progress_callback,
    control_callback,
    status_callback=None,
):
    """Klassifizieren/konvertieren vor Index-Cleanup; Fremdbestand bleibt unberuehrt."""
    from kiara.ingestion._file_utils import compute_file_hash
    from kiara.ingestion.datasource_runtime import build_datasource_overrides
    from kiara.ingestion.processing_revision import processing_file_hash

    request = validate_request(request, check_current=True)
    validate_target(row, model)
    paths = validate_candidates(candidates)
    root = Path(connector_cls.effective_path(conn_config, row)).resolve()
    config.setdefault("datasource", {})["path"] = str(root)
    overrides = build_datasource_overrides(row)
    for key in ("chunk_size", "chunk_overlap"):
        if overrides.get(key) is not None:
            config.setdefault("ingestion", {})[key] = overrides[key]
    summary = dict.fromkeys(
        (
            "inspected",
            "selected",
            "current",
            "not_html",
            "missing_original",
            "empty_body",
            "errors",
        ),
        0,
    )
    result = {"paths": [], "errors": {}, "cancelled": False, "summary": summary}
    selected = set()
    mail_paths = [path for path in paths if Path(path).suffix.lower() not in {".htm", ".html"}]
    hash_mode = getattr(connector_cls, "HASH_MODE", "mtime")
    if mail_paths:
        if not hasattr(connector_cls, "reprocess_cached"):
            raise ValueError("HTML-Auswahl enthaelt nicht unterstuetzte Dateipfade")
        prepared = connector_cls.reprocess_cached(
            conn_config,
            row,
            config,
            cache_paths=mail_paths,
            progress_callback=(lambda done, _total: progress_callback(done, len(paths)))
            if progress_callback
            else None,
            control_callback=control_callback,
            html_options={
                "mode": request["mode"],
                "stored_hashes": stored_hashes,
                "hash_mode": hash_mode,
                "config": config,
            },
        )
        result["cancelled"] = prepared["cancelled"]
        result["errors"].update(prepared["errors"])
        result["paths"].extend(prepared["files"])
        summary["inspected"] = prepared["examined"]
        summary["not_html"] = prepared["skipped"]["plain"]
        summary["current"] = prepared["skipped"]["current"]
        for kind in prepared.get("error_kinds", {}).values():
            if kind in ("missing_original", "empty_body"):
                summary[kind] += 1
        selected.update(prepared["files"])
        selected.update(prepared["errors"])
    mail_path_set = set(mail_paths)
    for path_text in (path for path in paths if path not in mail_path_set):
        if result["cancelled"] or (
            control_callback and control_callback() in {"cancel", "shutdown"}
        ):
            result["cancelled"] = True
            break
        summary["inspected"] += 1
        try:
            path = Path(path_text)
            if not path.is_file() or path.resolve() != path or not path.is_relative_to(root):
                raise ValueError("HTML-Datei fehlt oder liegt ausserhalb der Datenquelle")
            if request["mode"] == "outdated" and stored_hashes.get(
                path_text
            ) == processing_file_hash(config, path_text, compute_file_hash(path_text, hash_mode)):
                summary["current"] += 1
            else:
                selected.add(path_text)
                result["paths"].append(path_text)
        except Exception as exc:
            selected.add(path_text)
            result["errors"][path_text] = str(exc)
        if progress_callback:
            progress_callback(summary["inspected"], len(paths))
    # Auch Mail-Markdown muss die echten Chunk-Vertraege bestehen, bevor alte
    # Indexdaten geloescht werden duerfen. Parser-Erfolg allein genuegt nicht.
    if status_callback:
        status_callback("HTML-Konvertierung und Chunks pruefen...")
    valid = []
    prepared_hashes = {}
    for index, path in enumerate(result["paths"]):
        if result["cancelled"] or (
            control_callback and control_callback() in {"cancel", "shutdown"}
        ):
            result["cancelled"] = True
            break
        try:
            before = prepared_source_hash(config, path)
            _preflight_document(path, config, row)
            if prepared_source_hash(config, path) != before:
                raise ValueError("HTML-Quelle waehrend Vorpruefung geaendert")
            valid.append(path)
            prepared_hashes[path] = before
        except Exception as exc:
            result["errors"][path] = str(exc)
        if progress_callback:
            progress_callback(index + 1, len(result["paths"]))
    result["paths"] = valid
    result["prepared_hashes"] = prepared_hashes
    summary["selected"] = len(selected)
    summary["errors"] = len(result["errors"])
    result["selection"] = build_reingest_selection(row["id"], model, sorted(selected))
    return result
