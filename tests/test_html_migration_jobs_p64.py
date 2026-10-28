"""Projekt-64-Migration: Datei-/Modellabschluss und echter PG-Queuevertrag."""

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import pytest


def _selection(tmp_path):
    from kiara.ingestion.html_migration import build_selection

    first, second = tmp_path / "a.html", tmp_path / "b.html"
    first.write_text("<p>A</p>")
    second.write_text("<p>B</p>")
    return build_selection(
        25,
        {
            "nomic-embed-text": [str(first), str(second)],
            "hellord/e5-mistral-7b-instruct:q4_0": [str(first)],
        },
    )


def test_one_model_success_never_completes_whole_selection(tmp_path):
    from kiara.ingestion.html_migration import migration_completion

    selection = _selection(tmp_path)
    job = {
        "embedding_model": "nomic-embed-text",
        "status": "success",
        "stats_json": {
            "html_migration": selection,
            "html_migration_completed_paths": selection["files_by_model"]["nomic-embed-text"],
        },
    }
    result = migration_completion(selection, [job])
    assert result["complete"] is False
    assert result["pending_models"] == ["hellord/e5-mistral-7b-instruct:q4_0"]


def test_all_model_jobs_need_every_selected_file_and_same_revision(tmp_path):
    from kiara.ingestion.html_migration import migration_completion

    selection = _selection(tmp_path)
    jobs = [
        {
            "embedding_model": model,
            "status": "success",
            "stats_json": {
                "html_migration": copy.deepcopy(selection),
                "html_migration_completed_paths": paths,
            },
        }
        for model, paths in selection["files_by_model"].items()
    ]
    assert migration_completion(selection, jobs)["complete"] is True
    jobs[0]["stats_json"]["html_migration_completed_paths"] = []
    assert migration_completion(selection, jobs)["complete"] is False
    jobs[0]["stats_json"]["html_migration_completed_paths"] = selection["files_by_model"][
        "nomic-embed-text"
    ]
    jobs[0]["stats_json"]["html_migration"]["revision"] = "other-revision"
    assert migration_completion(selection, jobs)["complete"] is False


def test_excluded_model_cannot_be_selected(tmp_path):
    from kiara.ingestion.html_migration import build_selection

    with pytest.raises(ValueError, match="Modell"):
        build_selection(25, {"text-embedding-3-large": [str(tmp_path / "a.html")]})


def test_error_job_cannot_complete_despite_file_list(tmp_path):
    from kiara.ingestion.html_migration import migration_completion

    selection = _selection(tmp_path)
    jobs = [
        {
            "embedding_model": model,
            "status": "error",
            "stats_json": {"html_migration": selection, "html_migration_completed_paths": paths},
        }
        for model, paths in selection["files_by_model"].items()
    ]
    assert migration_completion(selection, jobs)["complete"] is False


def test_finalization_retains_stored_immutable_selection(monkeypatch):
    import kiara.web.ingestion_worker as module

    selection = {
        "project_id": 64,
        "revision": "html-markdown-p64-329-v1",
        "datasource_id": 25,
        "models": ["nomic-embed-text"],
        "files_by_model": {"nomic-embed-text": ["/cache/a.html"]},
    }
    db = Mock()
    db.execute.return_value.fetchone.return_value = {"stats_json": {"html_migration": selection}}
    jobs = Mock()
    monkeypatch.setattr(module, "_job_repo", jobs)
    monkeypatch.setattr(module, "_emb_repo", Mock())
    monkeypatch.setattr(module, "_ds_repo", Mock())
    recount = Mock()
    monkeypatch.setattr("kiara.ingestion.vectordb_ops._recount_datasource_embeddings", recount)
    worker = module.IngestionWorker.__new__(module.IngestionWorker)
    worker._update_datasource_metadata(
        db, 91, 25, {"processed_files": 1}, {}, "nomic-embed-text", 10, "2026-09-28"
    )
    persisted = json.loads(jobs.update.call_args.kwargs["stats_json"])
    assert persisted["html_migration"] == selection
    recount.assert_called_once_with(db, 25, "nomic-embed-text")


def test_migration_file_error_forces_error_outcome():
    from kiara.web.ingestion_worker import IngestionWorker

    worker = IngestionWorker.__new__(IngestionWorker)
    worker._stop = Mock()
    worker._stop.is_set.return_value = False
    worker._finalize_success = Mock()
    worker._finalize_error = Mock()
    stats = {"html_migration": {"project_id": 64}, "errors": 1}
    worker._resolve_job_outcome(91, 25, {}, "nomic-embed-text", stats, 1, lambda: None)
    worker._finalize_success.assert_not_called()
    worker._finalize_error.assert_called_once()


def _successful_jobs(selection):
    return [
        {
            "id": index + 1,
            "embedding_model": model,
            "status": "success",
            "stats_json": {"html_migration": selection, "html_migration_completed_paths": paths},
        }
        for index, (model, paths) in enumerate(selection["files_by_model"].items())
    ]


def test_latest_matching_job_invalidates_older_success(tmp_path):
    from kiara.ingestion.html_migration import migration_completion

    selection = _selection(tmp_path)
    jobs = _successful_jobs(selection)
    jobs.append({**copy.deepcopy(jobs[0]), "id": 99, "status": "error"})
    assert migration_completion(selection, jobs)["complete"] is False


def test_enqueue_next_persists_existing_trigger_and_selection(
    test_db, s4_guard_passthrough, tmp_path, monkeypatch
):
    from kiara.ingestion.html_migration import build_selection, enqueue_next

    config_path = tmp_path / "kiara.yaml"
    config_path.write_text("{}\n")
    monkeypatch.setenv("KIARA_CONFIG", str(config_path))
    assert test_db.execute(
        "SELECT 1 FROM pg_constraint WHERE conrelid = 'ingestion_jobs'::regclass "
        "AND conname = 'ingestion_jobs_triggered_by_check' AND contype = 'c'"
    ).fetchone()
    datasource_id = test_db.execute(
        "INSERT INTO datasources (name, type, path, is_active) "
        "VALUES ('P64 migration enqueue', 'local', %s, true) RETURNING id",
        (str(tmp_path),),
    ).fetchone()["id"]
    test_db.commit()
    selection = build_selection(datasource_id, _selection(tmp_path)["files_by_model"])

    result = enqueue_next(selection)

    row = test_db.execute(
        "SELECT * FROM ingestion_jobs WHERE id = %s", (result["job_id"],)
    ).fetchone()
    assert row["datasource_id"] == datasource_id
    assert row["triggered_by"] == "manual"
    assert row["status"] == "pending"
    assert row["job_type"] == "incremental"
    assert row["embedding_model"] == "nomic-embed-text"
    stats = row["stats_json"]
    if isinstance(stats, str):
        stats = json.loads(stats)
    assert stats["html_migration"] == selection
    assert result["complete"] is False


@pytest.mark.parametrize("change", ["source", "revision", "unchanged_imap"])
def test_enqueue_next_rechecks_current_input_and_processing_revision(tmp_path, monkeypatch, change):
    import kiara.config as config_module
    import kiara.connectors.registry as registry
    import kiara.shared.db as db_module
    import kiara.web.ingestion_worker as worker_module
    from kiara.connectors.imap_connector import IMAPConnector
    from kiara.ingestion import processing_revision
    from kiara.ingestion._file_utils import HashStore, compute_file_hash
    from kiara.ingestion.html_migration import enqueue_next

    selection = _selection(tmp_path)
    jobs = _successful_jobs(selection)
    config = {"ingestion": {"chunk_size": 640, "chunk_overlap": 100}}
    hashes = {
        model: {
            path: processing_revision.processing_file_hash(
                config, path, compute_file_hash(path, "content")
            )
            for path in paths
        }
        for model, paths in selection["files_by_model"].items()
    }
    monkeypatch.setattr(config_module, "load_config", lambda: copy.deepcopy(config))
    monkeypatch.setattr(registry, "ensure_loaded", lambda: None)
    monkeypatch.setattr(registry, "get", lambda _kind: IMAPConnector)
    monkeypatch.setattr(HashStore, "load", lambda _self, _ds, model: hashes[model])
    # Scoped Replay schreibt Contenthashes, ohne alte/fehlende Mode-Metadaten zu migrieren.
    stored_mode = Mock(return_value="mtime")
    monkeypatch.setattr(HashStore, "get_hash_mode", stored_mode)
    monkeypatch.setattr(
        "kiara.ingestion.datasource_runtime.build_datasource_overrides", lambda _row: {}
    )
    db = Mock()
    db.execute.return_value.fetchall.return_value = jobs
    db.execute.return_value.fetchone.return_value = {"id": 25, "type": "imap"}

    @contextmanager
    def read():
        yield db

    monkeypatch.setattr(db_module, "db_query", read)
    enqueue = Mock(return_value=77)
    monkeypatch.setattr(worker_module, "enqueue_job", enqueue)
    if change == "source":
        Path(selection["files_by_model"]["nomic-embed-text"][0]).write_text("<p>Geaendert</p>")
    elif change == "revision":
        monkeypatch.setattr(
            processing_revision, "processing_file_hash", lambda *_args: "new-profile"
        )
    result = enqueue_next(selection)
    if change == "unchanged_imap":
        assert result == {"datasource_id": 25, "complete": True, "job_id": None}
        enqueue.assert_not_called()
        stored_mode.assert_not_called()
        return
    assert result["complete"] is False
    assert result["job_id"] == 77
    enqueue.assert_called_once()


def test_mail_preparation_includes_indexed_html_attachment(tmp_path):
    from kiara.ingestion.html_migration import build_selection, prepare_files

    mail = tmp_path / "body.txt"
    mail.write_text("Mailbody")
    attachment = tmp_path / "attachment.html"
    attachment.write_text("<p>Vertragsbedingungen V42</p>")
    selection = build_selection(25, {"nomic-embed-text": [str(mail), str(attachment)]})
    config = {
        "datasource": {},
        "_html_migration": {
            **selection,
            "paths": selection["files_by_model"]["nomic-embed-text"],
            "errors": {},
        },
    }
    connector = Mock()
    connector.effective_path.return_value = str(tmp_path)
    connector.reprocess_cached.return_value = {
        "files": [str(mail)],
        "errors": {},
        "cancelled": False,
    }
    result = prepare_files(
        config, connector, {}, {"id": 25}, "nomic-embed-text", None, lambda: None
    )
    assert result["errors"] == {}
    assert result["files"] == selection["files_by_model"]["nomic-embed-text"]
    assert connector.reprocess_cached.call_args.kwargs["cache_paths"] == [str(mail)]


@pytest.mark.parametrize("explicit_missing", [False, True])
def test_preview_keeps_html_attachments_and_reports_unowned_explicit_paths(
    tmp_path, monkeypatch, explicit_missing
):
    import kiara.connectors.registry as registry
    import kiara.shared.db as db_module
    from kiara.ingestion._file_utils import ChunkDatasources, HashStore
    from kiara.ingestion.html_migration import preview_migration
    from kiara.repositories.file_hash import FileHashRepository

    mail, attachment = str(tmp_path / "body.txt"), str(tmp_path / "attachment.html")
    missing = str(tmp_path / "unknown.html")
    connector = Mock()
    connector.effective_path.return_value = str(tmp_path)
    monkeypatch.setattr(registry, "ensure_loaded", lambda: None)
    monkeypatch.setattr(registry, "get", lambda _kind: connector)
    monkeypatch.setattr(
        "kiara.ingestion.datasource_runtime.load_connector_config", lambda *_args: {}
    )
    monkeypatch.setattr(
        "kiara.ingestion.mail_migration._registry_entries", lambda _root: {mail: []}
    )
    monkeypatch.setattr(HashStore, "load", lambda *_args: {mail: "old", attachment: "old"})
    monkeypatch.setattr(FileHashRepository, "get_hash", lambda *_args, **_kwargs: None)
    chunk_ids = [f"chunk-{index}" for index in range(501)]
    monkeypatch.setattr(ChunkDatasources, "get_chunk_ids", lambda *_args: chunk_ids)
    monkeypatch.setattr(ChunkDatasources, "get_datasource_ids_for_chunks", lambda *_args: {})
    store = Mock()
    store.list_sources_for_datasource.return_value = []
    store.get_ids_by_sources.return_value = []
    store.get_documents_by_ids.side_effect = lambda ids, **_kwargs: {
        "ids": ids,
        "metadatas": [{"source": mail} for _chunk in ids],
    }
    factory = Mock(return_value=store)
    monkeypatch.setattr("kiara.retrieval.vectorstore.VectorStore", factory)
    db = Mock()
    db.execute.return_value.fetchone.return_value = {"id": 25, "type": "imap"}
    db.execute.return_value.fetchall.return_value = [
        {"embedding_model": "nomic-embed-text"},
        {"embedding_model": "text-embedding-3-large"},
    ]

    @contextmanager
    def read():
        yield db

    monkeypatch.setattr(db_module, "db_query", read)
    report = preview_migration({}, [25], paths=[missing] if explicit_missing else None)
    if explicit_missing:
        assert report["selections"] == []
        assert report["unresolved"][0]["path"] == missing
        store.get_documents_by_ids.assert_not_called()
    else:
        assert report["unresolved"] == []
        assert report["selections"][0]["files_by_model"]["nomic-embed-text"] == sorted(
            [mail, attachment]
        )
        assert [len(call.args[0]) for call in store.get_documents_by_ids.call_args_list] == [500, 1]
    assert report["excluded"] == [{"datasource_id": 25, "model": "text-embedding-3-large"}]
    factory.assert_called_once_with({}, embedding_model="nomic-embed-text", intent="maintenance")


def test_explicit_preview_uses_sot_ownership_without_trusting_registry(tmp_path, monkeypatch):
    import kiara.connectors.registry as registry
    import kiara.shared.db as db_module
    from kiara.ingestion._file_utils import ChunkDatasources, HashStore
    from kiara.ingestion.html_migration import preview_migration
    from kiara.repositories.file_hash import FileHashRepository

    mail, foreign = str(tmp_path / "owned.txt"), str(tmp_path / "foreign.txt")
    hashed, payload = str(tmp_path / "hashed.html"), str(tmp_path / "payload.html")
    paths = sorted([mail, foreign, hashed, payload])
    connector = Mock()
    connector.effective_path.return_value = str(tmp_path)
    monkeypatch.setattr(registry, "ensure_loaded", lambda: None)
    monkeypatch.setattr(registry, "get", lambda _kind: connector)
    monkeypatch.setattr(
        "kiara.ingestion.datasource_runtime.load_connector_config", lambda *_args: {}
    )
    monkeypatch.setattr(
        "kiara.ingestion.mail_migration._registry_entries", lambda _root: {mail: [], foreign: []}
    )
    full_hashes = Mock(
        side_effect=AssertionError("Explizite Vorschau darf keine Vollhashliste lesen")
    )
    full_ids = Mock(side_effect=AssertionError("Explizite Vorschau darf keine Gesamt-DS lesen"))
    monkeypatch.setattr(HashStore, "load", full_hashes)
    monkeypatch.setattr(ChunkDatasources, "get_chunk_ids", full_ids)
    monkeypatch.setattr(
        FileHashRepository,
        "get_hash",
        lambda _self, _ds, _model, path, **_kwargs: "old" if path == hashed else None,
    )
    ownership = Mock(return_value={"owned-chunk": [25], "foreign-chunk": [99]})
    monkeypatch.setattr(ChunkDatasources, "get_datasource_ids_for_chunks", ownership)
    store = Mock()
    store.list_sources_for_datasource.return_value = [{"source": payload}]
    store.get_ids_by_sources.return_value = ["owned-chunk", "foreign-chunk"]
    store.get_documents_by_ids.return_value = {
        "ids": ["owned-chunk"],
        "metadatas": [{"source": mail}],  # Keine Payload-ds_ids: SoT bleibt massgeblich.
    }
    monkeypatch.setattr("kiara.retrieval.vectorstore.VectorStore", Mock(return_value=store))
    db = Mock()
    db.execute.return_value.fetchone.return_value = {"id": 25, "type": "imap"}
    db.execute.return_value.fetchall.return_value = [{"embedding_model": "nomic-embed-text"}]

    @contextmanager
    def read():
        yield db

    monkeypatch.setattr(db_module, "db_query", read)
    report = preview_migration({}, [25], paths=paths)
    assert report["selections"][0]["files_by_model"]["nomic-embed-text"] == sorted(
        [mail, hashed, payload]
    )
    assert report["unresolved"] == [
        {"path": foreign, "error": "Expliziter Pfad ohne freigegebenen Modellbestand"}
    ]
    assert report["limited"] is True
    assert report["inventory_scope"] == "selected_paths"
    store.list_sources_for_datasource.assert_called_once_with(25, sources=paths)
    store.get_ids_by_sources.assert_called_once_with(paths)
    ownership.assert_called_once_with(["owned-chunk", "foreign-chunk"], "nomic-embed-text")
    store.get_documents_by_ids.assert_called_once_with(["owned-chunk"], include=["metadatas"])
    full_hashes.assert_not_called()
    full_ids.assert_not_called()


def test_early_error_preserves_prepared_file_diagnostics(tmp_path):
    from kiara.ingestion.html_migration import preserve_selection

    selection = _selection(tmp_path)
    errors = {selection["files_by_model"]["nomic-embed-text"][0]: "Konvertierung fehlgeschlagen"}
    db = Mock()
    db.execute.return_value.fetchone.return_value = {
        "stats_json": {"html_migration": selection, "html_migration_errors": errors}
    }
    stats = {"error": "Pipeline vor Rueckgabe abgebrochen"}
    preserve_selection(db, 91, stats)
    assert stats["html_migration"] == selection
    assert stats["html_migration_errors"] == errors


@pytest.mark.parametrize("migration", [True, False])
def test_job_context_scopes_real_warmup_without_changing_datasource(
    tmp_path, monkeypatch, migration
):
    import kiara.backend_selector as selector
    import kiara.config as config_module
    import kiara.connectors.registry as registry
    import kiara.web.ingestion_worker as module
    from kiara.backend_warmup import build_warmup_tasks

    selection = _selection(tmp_path)
    ds_row = {"id": 25, "type": "local", "ds_colbert_enabled": True}
    config = {
        "datasource": {"path": str(tmp_path)},
        "graph": {"enabled": False},
        "retrieval": {"colbert": {"enabled": True, "model": "excluded-colbert"}},
    }
    original_config, original_row = copy.deepcopy(config), copy.deepcopy(ds_row)
    repository = Mock()
    repository.get_by_id.return_value = ds_row
    monkeypatch.setattr(module, "_ds_repo", repository)
    monkeypatch.setattr(config_module, "load_config", lambda: config)
    monkeypatch.setattr(registry, "ensure_loaded", lambda: None)
    monkeypatch.setattr(registry, "get", lambda _kind: Mock(SCAN_CACHE_MAX_AGE=0))
    monkeypatch.setattr(module, "load_connector_config", lambda *_args: {})
    backend = {
        "model": "nomic-embed-text",
        "provider": "ollama",
        "url": "http://warmup.invalid",
        "extra_config": {"embedding_profile_id": "nomic-selected"},
    }
    monkeypatch.setattr(selector, "get_all_active", lambda _category: [backend])
    monkeypatch.setattr(
        "kiara.retrieval.colbert._resolve_colbert_url", lambda _config: "http://colbert.invalid"
    )
    job = {
        "id": 91,
        "embedding_model": "nomic-embed-text",
        "job_type": "incremental",
        "triggered_by": "manual" if migration else "cron",
        "stats_json": {"html_migration": selection} if migration else {},
    }
    worker = module.IngestionWorker.__new__(module.IngestionWorker)
    context = worker._load_job_context(job, 25, 91)
    tasks = build_warmup_tasks(context["ds_row"], "nomic-embed-text", context["config"])
    assert [task.category for task in tasks] == (
        ["embedding"] if migration else ["embedding", "colbert"]
    )
    assert ds_row == original_row
    assert config == original_config
