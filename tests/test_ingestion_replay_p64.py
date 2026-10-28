"""P64: repeat incomplete files and retain excluded model state."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from kiara.ingestion import pipeline
from kiara.ingestion._file_utils import HashStore, compute_file_hash

_CLEANUP_BODY = pipeline._cleanup_stale_chunks.__wrapped__


@pytest.fixture
def diff_run(monkeypatch, tmp_path):
    path = tmp_path / "document.html"
    path.write_text("<p>Artikel A: 17 EUR</p>")
    config = {
        "datasource": {"path": str(tmp_path), "file_types": [".html"], "exclude_patterns": []},
        "ingestion": {"chunk_size": 640, "chunk_overlap": 100},
    }
    hashes = Mock(spec=HashStore)
    hashes.load.return_value = {}
    hashes.get_hash_mode.return_value = "mtime"
    hashes.get_missing_size_paths.return_value = set()
    monkeypatch.setattr(pipeline, "HashStore", lambda: hashes)
    cleanup = Mock()
    monkeypatch.setattr(pipeline, "_cleanup_stale_chunks", cleanup)
    return config, str(path), hashes, cleanup


def test_resume_log_without_completed_hash_cannot_skip_file(diff_run):
    config, path, hashes, cleanup = diff_run
    result = pipeline._load_and_diff_files(
        config, Mock(), True, 1, "nomic-embed-text", resume_skip_files={path}
    )
    assert result[1] == [path]
    assert result[5] == {}
    cleanup.assert_called_once()
    assert cleanup.call_args.args[-1] == []


def test_file_without_hash_is_prepared_before_deciding_cleanup(diff_run):
    config, path, hashes, cleanup = diff_run
    result = pipeline._load_and_diff_files(config, Mock(), True, 1, "nomic-embed-text")
    assert result[1] == [path]
    assert cleanup.call_args.args[-1] == []


def test_unchanged_html_gets_processing_revision_hash(diff_run):
    config, path, hashes, cleanup = diff_run
    hashes.load.return_value = {path: compute_file_hash(path)}
    first = pipeline._load_and_diff_files(config, Mock(), True, 1, "nomic-embed-text")
    assert first[1] == [path]
    hashes.load.return_value = {path: first[2][path][0]}
    second = pipeline._load_and_diff_files(config, Mock(), True, 1, "nomic-embed-text")
    assert second[1] == []
    config["ingestion"]["chunk_size"] = 800
    third = pipeline._load_and_diff_files(config, Mock(), True, 1, "nomic-embed-text")
    assert third[1] == [path]
    assert third[2][path][0] != first[2][path][0]


def test_title_correction_invalidates_completed_unchanged_html(diff_run, monkeypatch):
    from pathlib import Path

    from kiara_document_loaders import html

    from kiara.ingestion.processing_revision import processing_file_hash

    config, path, hashes, cleanup = diff_run
    source = Path(path).read_bytes()
    stat = Path(path).stat()
    with monkeypatch.context() as old:
        old.setattr(html, "_PROFILE_REVISION", "kiara-html-3")
        previous = processing_file_hash(config, path, compute_file_hash(path))
    hashes.load.return_value = {path: previous}
    result = pipeline._load_and_diff_files(config, Mock(), True, 1, "nomic-embed-text")
    assert result[1] == [path]
    assert result[2][path][0] != previous
    assert Path(path).read_bytes() == source
    assert Path(path).stat().st_mtime_ns == stat.st_mtime_ns
    assert cleanup.call_args.args[-1] == []
    hashes.load.return_value = {path: result[2][path][0]}
    assert pipeline._load_and_diff_files(config, Mock(), True, 1, "nomic-embed-text")[1] == []


def test_scoped_hash_write_keeps_other_model_sizes_and_uses_given_transaction():
    from kiara_ingestion_core import index_state

    conn = Mock()
    index_state.HashStore().upsert_many(
        1,
        "nomic-embed-text",
        {"/cache/mail.txt": ("new", 90)},
        conn=conn,
        sync_model_sizes=False,
    )
    assert conn.executemany.call_count == 1
    assert "INSERT INTO file_hashes" in conn.executemany.call_args.args[0]


def test_file_completion_hash_and_recount_share_transaction(monkeypatch):
    events = []
    conn = object()

    @contextmanager
    def transaction():
        events.append("begin")
        yield conn
        events.append("commit")

    hashes = Mock()
    hashes.upsert_many.side_effect = lambda *a, **kw: events.append(("hash", kw["conn"]))
    monkeypatch.setattr(pipeline, "db_transaction", transaction)
    monkeypatch.setattr(
        "kiara.ingestion.vectordb_ops._recount_datasource_embeddings",
        lambda c, ds, model: events.append(("recount", c, model)),
    )
    pipeline._complete_file(
        hashes, 1, "nomic-embed-text", "/cache/mail.txt", ("v2", 90), sync_model_sizes=False
    )
    assert events == ["begin", ("hash", conn), ("recount", conn, "nomic-embed-text"), "commit"]


def _select_migration(config, path):
    config["_html_migration"] = {
        "project_id": 64,
        "revision": "html-markdown-p64-329-v1",
        "datasource_id": 1,
        "models": ["nomic-embed-text"],
        "files_by_model": {"nomic-embed-text": [path]},
        "paths": [path],
        "errors": {},
    }


def test_selected_file_replays_even_with_current_completed_hash(diff_run):
    from kiara.ingestion.processing_revision import processing_file_hash

    config, path, hashes, cleanup = diff_run
    _select_migration(config, path)
    digest = processing_file_hash(config, path, compute_file_hash(path))
    hashes.load.return_value = {path: digest}
    result = pipeline._load_and_diff_files(
        config, Mock(), True, 1, "nomic-embed-text", resume_skip_files={path}
    )
    assert result[1] == [path]
    assert cleanup.call_args.args[-1] == [path]


def test_migration_selection_does_not_call_normal_scan(diff_run, monkeypatch):
    config, path, _, _ = diff_run
    _select_migration(config, path)
    scan = Mock(side_effect=AssertionError("Normaler Scan ausserhalb der Auswahl"))
    monkeypatch.setattr(pipeline, "scan_files", scan)
    result = pipeline._load_and_diff_files(config, Mock(), True, 1, "nomic-embed-text")
    assert result[0] == [path]
    scan.assert_not_called()


def test_unselected_existing_and_missing_files_are_not_cleaned(diff_run, tmp_path):
    config, path, hashes, cleanup = diff_run
    _select_migration(config, path)
    unselected = tmp_path / "unselected.html"
    unselected.write_text("<p>Bestand ausserhalb der Auswahl</p>")
    missing = str(tmp_path / "missing-unselected.html")
    hashes.load.return_value = {str(unselected): "old", missing: "old"}
    result = pipeline._load_and_diff_files(config, Mock(), True, 1, "nomic-embed-text")
    assert result[0] == [path]
    assert result[1] == [path]
    assert set(result[2]) == {path}
    assert cleanup.call_args.args[-1] == [path]
    assert str(unselected) not in result[3]
    assert missing not in result[3]


def test_migration_skips_global_hash_mode_and_size_healing(diff_run):
    from kiara.ingestion.processing_revision import processing_file_hash

    config, path, hashes, _ = diff_run
    _select_migration(config, path)
    hashes.load.return_value = {path: processing_file_hash(config, path, compute_file_hash(path))}
    hashes.get_hash_mode.return_value = "content"
    hashes.get_missing_size_paths.return_value = {path}
    pipeline._load_and_diff_files(config, Mock(), True, 1, "nomic-embed-text", hash_mode="mtime")
    hashes.migrate_hash_mode.assert_not_called()
    hashes.get_missing_size_paths.assert_not_called()
    hashes.upsert_many.assert_not_called()


def _loaded_hash(config, path):
    from kiara.ingestion._file_utils import compute_file_hash
    from kiara.ingestion.processing_revision import processing_file_hash

    return processing_file_hash(config, path, compute_file_hash(path)), Path(path).stat().st_size


@pytest.fixture
def scoped_pipeline_run(diff_run, tmp_path, monkeypatch):
    """Isolate outer pipeline wiring while exercising its real completion callback."""
    config, path, hashes, _ = diff_run
    _select_migration(config, path)
    config["vectordb"] = {"path": str(tmp_path / "vectors")}
    config["ingestion"]["batch_size"] = 8
    monkeypatch.setattr(pipeline, "_apply_ds_overrides", Mock())
    monkeypatch.setattr(pipeline, "_check_ddu_preflight", Mock())
    monkeypatch.setattr(pipeline, "VectorStore", Mock())
    monkeypatch.setattr(pipeline, "ChunkDatasources", Mock())
    diff = Mock(return_value=([path], [path], {path: ("new", 25)}, [], hashes, {}))
    monkeypatch.setattr(pipeline, "_load_and_diff_files", diff)
    completion = Mock()
    monkeypatch.setattr(pipeline, "_complete_file", completion)
    monkeypatch.setattr(pipeline, "_persist_hashes_checkpoint", Mock())
    colbert_cleanup = Mock(return_value=True)
    monkeypatch.setattr(pipeline, "_run_colbert_model_switch_cleanup", colbert_cleanup)
    monkeypatch.setattr("kiara.ingestion.vectordb_ops._recount_datasource_embeddings", Mock())

    @contextmanager
    def transaction():
        yield Mock()

    monkeypatch.setattr(pipeline, "db_transaction", transaction)

    def process_file(context, current_path):
        with context.file_loading_context(current_path):
            Path(current_path).read_bytes()
        context.file_completion_callback(current_path)
        context.stats["processed_files"] += 1
        return True

    monkeypatch.setattr(pipeline, "_process_file_in_pipeline", process_file)
    return config, path, hashes, diff, completion, colbert_cleanup


def test_migration_completion_callback_never_updates_other_model_sizes(scoped_pipeline_run):
    config, path, hashes, _, completion, _ = scoped_pipeline_run
    stats = pipeline.run_ingestion.__wrapped__(
        config, incremental=True, datasource_id=1, embedding_model="nomic-embed-text"
    )
    completion.assert_called_once_with(
        hashes, 1, "nomic-embed-text", path, _loaded_hash(config, path), sync_model_sizes=False
    )
    assert stats["html_migration_completed_paths"] == [path]
    hashes.set_hash_mode.assert_not_called()


@pytest.mark.parametrize("has_validated_paths", [True, False])
def test_migration_never_runs_historical_colbert_cleanup(scoped_pipeline_run, has_validated_paths):
    config, path, hashes, diff, _, colbert_cleanup = scoped_pipeline_run
    if not has_validated_paths:
        config["_html_migration"]["paths"] = []
        config["_html_migration"]["errors"] = {path: "Konvertierung fehlgeschlagen"}
        diff.return_value = ([], [], {}, [], hashes, {})
    pipeline.run_ingestion.__wrapped__(
        config, incremental=True, datasource_id=1, embedding_model="nomic-embed-text"
    )
    colbert_cleanup.assert_not_called()


def test_migration_cleanup_guard_fences_only_selected_primary_model(diff_run, monkeypatch):
    from kiara.retrieval import index_compatibility as contract

    config, path, hashes, _ = diff_run
    _select_migration(config, path)
    target = SimpleNamespace(logical_index_key="selected-primary")
    monkeypatch.setattr(contract, "stored_primary_index_target", lambda *_args: target)
    historical_models = Mock(
        side_effect=AssertionError("Historische ColBERT-Bestaende sind ausgeschlossen")
    )
    monkeypatch.setattr(
        "kiara.ingestion.vectordb_ops._resolve_colbert_models_for_ds", historical_models
    )
    guards = []

    @contextmanager
    def guard(_config, current_target, **kwargs):
        guards.append((current_target, kwargs))
        yield Mock()

    monkeypatch.setattr(contract, "guard_index", guard)
    cleanup_body = Mock()
    guarded = pipeline._guard_stale_chunk_cleanup(cleanup_body)
    store, chunk_ds = Mock(), Mock()
    guarded(config, store, hashes, chunk_ds, 1, "nomic-embed-text", [path])
    historical_models.assert_not_called()
    assert guards == [
        (
            target,
            {
                "operation": "incremental_reingest_cleanup",
                "write": True,
                "validation_scope": "stored_binding",
            },
        )
    ]
    cleanup_body.assert_called_once_with(
        config, store, hashes, chunk_ds, 1, "nomic-embed-text", [path], colbert_keys=frozenset()
    )


@pytest.mark.parametrize("colbert_recount_fails", [False, True])
def test_completion_recounts_active_colbert_in_same_transaction_before_commit(
    monkeypatch, colbert_recount_fails
):
    events = []
    conn = object()
    colbert_model = "colbert::active-model"

    @contextmanager
    def transaction():
        events.append("begin")
        try:
            yield conn
        except Exception:
            events.append("rollback")
            raise
        else:
            events.append("commit")

    def recount(connection, datasource_id, model):
        assert connection is conn
        assert datasource_id == 1
        events.append(("recount", connection, model))
        if colbert_recount_fails and model == colbert_model:
            raise RuntimeError("ColBERT-Recount fehlgeschlagen")

    hashes = Mock()
    hashes.upsert_many.side_effect = lambda *args, **kwargs: events.append(("hash", kwargs["conn"]))
    monkeypatch.setattr(pipeline, "db_transaction", transaction)
    monkeypatch.setattr("kiara.ingestion.vectordb_ops._recount_datasource_embeddings", recount)

    def complete():
        pipeline._complete_file(
            hashes,
            1,
            "nomic-embed-text",
            "/cache/mail.txt",
            ("v2", 90),
            sync_model_sizes=False,
            additional_models=(colbert_model,),
        )

    if colbert_recount_fails:
        with pytest.raises(RuntimeError, match="ColBERT-Recount"):
            complete()
    else:
        complete()
    assert events == [
        "begin",
        ("hash", conn),
        ("recount", conn, "nomic-embed-text"),
        ("recount", conn, colbert_model),
        "rollback" if colbert_recount_fails else "commit",
    ]


@pytest.mark.parametrize("replacement", ["missing", "directory"])
def test_validated_selected_file_disappearing_fails_before_cleanup(diff_run, replacement):
    from pathlib import Path

    config, path, hashes, cleanup = diff_run
    _select_migration(config, path)
    Path(path).unlink()
    if replacement == "directory":
        Path(path).mkdir()
    with pytest.raises(ValueError, match="Vorvalidierte HTML-Migrationsdatei"):
        pipeline._load_and_diff_files(config, Mock(), True, 1, "nomic-embed-text")
    cleanup.assert_not_called()
    hashes.upsert_many.assert_not_called()
    hashes.remove_files.assert_not_called()


def test_invalid_selection_fails_before_primary_store_guard(scoped_pipeline_run, monkeypatch):
    config, _, _, _, _, _ = scoped_pipeline_run
    config["_html_migration"]["datasource_id"] = 99
    store = Mock(side_effect=AssertionError("Primary guard must not run for invalid scope"))
    monkeypatch.setattr(pipeline, "VectorStore", store)
    with pytest.raises(ValueError):
        pipeline.run_ingestion.__wrapped__(
            config, incremental=True, datasource_id=1, embedding_model="nomic-embed-text"
        )
    store.assert_not_called()


def test_cleanup_outside_selection_is_rejected_before_any_guard(diff_run, monkeypatch):
    from kiara.retrieval import index_compatibility as contracts

    config, path, hashes, _ = diff_run
    _select_migration(config, path)
    target = Mock(side_effect=AssertionError("Out-of-scope cleanup reached target construction"))
    monkeypatch.setattr(contracts, "stored_primary_index_target", target)
    cleanup = Mock()
    guarded = pipeline._guard_stale_chunk_cleanup(cleanup)
    with pytest.raises(ValueError, match="ausserhalb"):
        guarded(config, Mock(), hashes, Mock(), 1, "nomic-embed-text", [path + ".unselected"])
    target.assert_not_called()
    cleanup.assert_not_called()


def test_selected_cleanup_body_preserves_historical_colbert(diff_run, monkeypatch):
    from kiara.ingestion import vectordb_ops

    config, path, hashes, _ = diff_run
    _select_migration(config, path)
    store, chunk_ds = Mock(), Mock()
    remaining = ["old-primary-chunk"]
    store.get_ids_by_sources.side_effect = lambda sources: list(remaining)
    store.delete_by_sources.side_effect = lambda sources: remaining.clear()
    chunk_ds.get_other_datasources.return_value = []
    chunk_ds.get_datasource_ids_for_chunks.return_value = {}
    chunk_ds.filter_unreferenced_chunk_ids.return_value = []
    historical = Mock(side_effect=AssertionError("Historical ColBERT is excluded"))
    monkeypatch.setattr(vectordb_ops, "_resolve_colbert_models_for_ds", historical)
    monkeypatch.setattr(vectordb_ops, "delete_from_elasticsearch", Mock())
    monkeypatch.setattr(vectordb_ops, "invalidate_bm25", Mock())
    connection = Mock()

    @contextmanager
    def transaction():
        yield connection

    monkeypatch.setattr(pipeline, "db_transaction", transaction)
    _CLEANUP_BODY(config, store, hashes, chunk_ds, 1, "nomic-embed-text", [path])
    historical.assert_not_called()
    chunk_ds.delete_by_chunk_ids.assert_called_once_with(
        ["old-primary-chunk"], "nomic-embed-text", conn=connection
    )
    store.delete_by_sources.assert_called_once_with([path])
    assert remaining == []


def test_migration_disables_colbert_writer_without_mutating_datasource_overrides(
    scoped_pipeline_run, monkeypatch
):
    config, path, _, _, _, _ = scoped_pipeline_run
    config["retrieval"] = {"colbert": {"enabled": True, "model": "active-model"}}
    overrides = {"colbert_enabled": True}
    active_writers = []

    def process(context, current_path):
        active_writers.append(pipeline._colbert_enabled_for_run(context))
        with context.file_loading_context(current_path):
            Path(current_path).read_bytes()
        context.file_completion_callback(current_path)
        return True

    monkeypatch.setattr(pipeline, "_process_file_in_pipeline", process)
    stats = pipeline.run_ingestion.__wrapped__(
        config,
        incremental=True,
        datasource_id=1,
        embedding_model="nomic-embed-text",
        ds_overrides=overrides,
    )
    assert active_writers == [False]
    assert overrides == {"colbert_enabled": True}
    assert stats["html_migration_completed_paths"] == [path]


def test_failed_completion_never_reports_selected_path(scoped_pipeline_run, monkeypatch):
    config, _, _, _, completion, _ = scoped_pipeline_run
    completion.side_effect = RuntimeError("Completion transaction failed")
    contexts = []

    def process(context, current_path):
        contexts.append(context)
        with context.file_loading_context(current_path):
            Path(current_path).read_bytes()
        context.file_completion_callback(current_path)

    monkeypatch.setattr(pipeline, "_process_file_in_pipeline", process)
    with pytest.raises(RuntimeError, match="Completion transaction failed"):
        pipeline.run_ingestion.__wrapped__(
            config, incremental=True, datasource_id=1, embedding_model="nomic-embed-text"
        )
    assert contexts[0].stats["html_migration_completed_paths"] == []


@pytest.mark.parametrize(
    "globally_enabled,ds_enabled", [(True, True), (True, False), (False, True)]
)
def test_normal_completion_recounts_only_actually_active_colbert(
    scoped_pipeline_run, globally_enabled, ds_enabled
):
    config, path, hashes, _, completion, _ = scoped_pipeline_run
    del config["_html_migration"]
    config["retrieval"] = {"colbert": {"enabled": globally_enabled, "model": "Active-Model:latest"}}
    pipeline.run_ingestion.__wrapped__(
        config,
        incremental=True,
        datasource_id=1,
        embedding_model="nomic-embed-text",
        ds_overrides={"colbert_enabled": ds_enabled},
    )
    expected = {"sync_model_sizes": True}
    if globally_enabled and ds_enabled:
        expected["additional_models"] = ("colbert::active-model",)
    completion.assert_called_once_with(
        hashes, 1, "nomic-embed-text", path, _loaded_hash(config, path), **expected
    )
