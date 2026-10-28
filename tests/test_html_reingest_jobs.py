"""HTML-Bestandsjobs: begrenzte Auswahl, Revisionen und sichere Wiederaufnahme."""

import copy
import json
from unittest.mock import Mock

import pytest

MODEL = "nomic-embed-text"


def test_empty_html_selection_is_explicit_and_cannot_become_full_scan(tmp_path, monkeypatch):
    from kiara.ingestion import pipeline
    from kiara.ingestion.html_migration import build_reingest_selection, validate_run_selection

    selection = build_reingest_selection(12, MODEL, [])
    config = {
        "datasource": {"path": str(tmp_path)},
        "_html_migration": {**selection, "paths": [], "errors": {}},
    }
    assert validate_run_selection(config, 12, MODEL)["paths"] == []
    scan = Mock(side_effect=AssertionError("Keine normale Dateisuche"))
    hashes = Mock()
    hashes.load.side_effect = AssertionError("Keine unselektierten Dateihashes")
    cleanup = Mock()
    monkeypatch.setattr(pipeline, "scan_files", scan)
    monkeypatch.setattr(pipeline, "HashStore", lambda: hashes)
    monkeypatch.setattr(pipeline, "_cleanup_stale_chunks", cleanup)
    result = pipeline._load_and_diff_files(config, Mock(), True, 12, MODEL)
    assert result[0] == result[1] == result[3] == []
    assert cleanup.call_args.args[-1] == []
    scan.assert_not_called()
    hashes.load.assert_not_called()


@pytest.mark.parametrize("mode", ["outdated", "all"])
def test_request_binds_parser_and_chunking_revision(mode):
    from kiara.ingestion.html_reingest import build_request, validate_request

    request = build_request(mode)
    assert validate_request(request, check_current=True) == request
    old = copy.deepcopy(request)
    old["preparation_revision"][1] = "old-profile"
    with pytest.raises(ValueError, match="Revision"):
        validate_request(old, check_current=True)


@pytest.mark.parametrize("mode", ["", "full", None, True, [], {}])
def test_invalid_html_mode_is_rejected(mode):
    from kiara.ingestion.html_reingest import build_request

    with pytest.raises(ValueError):
        build_request(mode)


def test_generic_selection_is_not_project_bound_and_preserves_model_limit(tmp_path):
    from kiara.ingestion.html_migration import build_reingest_selection, validate_selection

    selected = build_reingest_selection(12, MODEL, [str(tmp_path / "a.html")])
    assert "project_id" not in selected
    assert validate_selection(selected) == selected
    with pytest.raises(ValueError):
        build_reingest_selection(12, "text-embedding-3-large", [])


def test_html_request_survives_error_before_classification():
    from kiara.ingestion.html_migration import preserve_selection
    from kiara.ingestion.html_reingest import build_request

    request = build_request("outdated")
    stored = {"html_reingest": request, "html_reingest_candidates": ["/cache/a.txt"]}
    db = Mock()
    db.execute.return_value.fetchone.return_value = {"stats_json": stored}
    result = {"errors": 1}
    preserve_selection(db, 5, result)
    assert result["html_reingest"] == request
    assert result["html_reingest_candidates"] == ["/cache/a.txt"]


def test_prepare_rejects_bad_table_before_returning_indexable_path(tmp_path):
    from kiara.connectors.local_connector import LocalConnector
    from kiara.ingestion.html_reingest import build_request, prepare_reingestion

    good = tmp_path / "good.html"
    good.write_text("<title>Guter Titel</title><p>Beleg V17</p>")
    bad = tmp_path / "bad.html"
    bad.write_text(
        "<table><tr><th>"
        + "Zu langer Tabellenkopf " * 30
        + "</th></tr><tr><td>42</td></tr></table>"
    )
    config = {"ingestion": {"chunk_size": 100, "chunk_overlap": 10}}
    row = {"id": 12, "type": "local", "ds_chunk_size": 100, "ds_chunk_overlap": 10}
    result = prepare_reingestion(
        config,
        LocalConnector,
        {"path": str(tmp_path)},
        row,
        MODEL,
        build_request("all"),
        [str(good), str(bad)],
        {},
        None,
        None,
    )
    assert result["paths"] == [str(good)]
    assert str(bad) in result["errors"]
    assert result["summary"]["selected"] == 2


def test_outdated_html_uses_processing_hash_and_all_forces_replay(tmp_path):
    from kiara.connectors.local_connector import LocalConnector
    from kiara.ingestion._file_utils import compute_file_hash
    from kiara.ingestion.html_reingest import build_request, prepare_reingestion
    from kiara.ingestion.processing_revision import processing_file_hash

    path = tmp_path / "a.html"
    path.write_text("<p>Aktueller Inhalt</p>")
    config = {"ingestion": {"chunk_size": 500, "chunk_overlap": 50}}
    row = {"id": 12, "type": "local", "ds_chunk_size": 500, "ds_chunk_overlap": 50}
    hashes = {
        str(path): processing_file_hash(config, str(path), compute_file_hash(str(path), "mtime"))
    }
    result = prepare_reingestion(
        config,
        LocalConnector,
        {"path": str(tmp_path)},
        row,
        MODEL,
        build_request("outdated"),
        [str(path)],
        hashes,
        None,
        None,
    )
    assert result["paths"] == []
    assert result["summary"]["current"] == 1
    result = prepare_reingestion(
        config,
        LocalConnector,
        {"path": str(tmp_path)},
        row,
        MODEL,
        build_request("all"),
        [str(path)],
        hashes,
        None,
        None,
    )
    assert result["paths"] == [str(path)]


def test_enqueue_html_does_not_initialize_missing_model_stock(
    test_db, s4_guard_passthrough, tmp_path, monkeypatch
):
    from kiara.web.ingestion_worker import enqueue_job

    config = tmp_path / "test.yaml"
    config.write_text("{}")
    monkeypatch.setenv("KIARA_CONFIG", str(config))
    ds = test_db.execute(
        "INSERT INTO datasources(name,type,path,is_active) VALUES ('html selection','local',%s,true) RETURNING id",
        (str(tmp_path),),
    ).fetchone()["id"]
    test_db.commit()
    with pytest.raises(ValueError, match="Bestand"):
        enqueue_job(ds, embedding_model=MODEL, html_reingest="outdated")
    assert (
        test_db.execute(
            "SELECT count(*) AS n FROM ingestion_jobs WHERE datasource_id=%s", (ds,)
        ).fetchone()["n"]
        == 0
    )


def test_html_mode_cannot_be_combined_with_full_cleanup():
    from kiara.web.ingestion_worker import enqueue_job

    with pytest.raises(ValueError):
        enqueue_job(12, job_type="full", embedding_model=MODEL, html_reingest="all")


@pytest.mark.parametrize("owner_layer", ["postgres", "payload"])
def test_shared_sources_are_rejected_without_deleting_any_layer(tmp_path, owner_layer):
    from kiara.ingestion.html_migration import build_reingest_selection
    from kiara.ingestion.html_reingest import build_request, prepared_source_hash
    from kiara.ingestion.pipeline import _cleanup_stale_chunks

    path = str(tmp_path / "shared.html")
    (tmp_path / "shared.html").write_text("<p>Gemeinsamer Inhalt</p>")
    config = {"_html_reingest": build_request("all")}
    config["_html_reingest_prepared_hashes"] = {path: prepared_source_hash(config, path)}
    config["_html_migration"] = {
        **build_reingest_selection(12, MODEL, [path]),
        "paths": [path],
        "errors": {},
    }
    store, hashes, chunks = Mock(), Mock(), Mock()
    store.get_ids_by_sources.return_value = ["old"]
    store.get_documents_by_ids.return_value = {
        "ids": ["old"],
        "metadatas": [{"ds_ids": [12, 15] if owner_layer == "payload" else [12]}],
    }
    chunks.get_other_datasources.return_value = {15} if owner_layer == "postgres" else set()
    with pytest.raises(ValueError, match="anderen Datenquellen"):
        _cleanup_stale_chunks.__wrapped__(config, store, hashes, chunks, 12, MODEL, [path])
    store.delete_by_sources.assert_not_called()
    chunks.delete_by_chunk_ids.assert_not_called()
    hashes.remove_files.assert_not_called()


def test_changed_preflight_source_is_rejected_before_cleanup(tmp_path):
    from kiara.connectors.local_connector import LocalConnector
    from kiara.ingestion.html_reingest import build_request, prepare_reingestion
    from kiara.ingestion.pipeline import _cleanup_stale_chunks

    path = tmp_path / "a.html"
    path.write_text("<p>Gepruefter Inhalt</p>")
    config = {
        "_html_reingest": build_request("all"),
        "ingestion": {"chunk_size": 500, "chunk_overlap": 50},
    }
    result = prepare_reingestion(
        config,
        LocalConnector,
        {"path": str(tmp_path)},
        {"id": 12, "type": "local"},
        MODEL,
        config["_html_reingest"],
        [str(path)],
        {},
        None,
        None,
    )
    config["_html_migration"] = {
        **result["selection"],
        "paths": result["paths"],
        "errors": result["errors"],
    }
    config["_html_reingest_prepared_hashes"] = result["prepared_hashes"]
    path.write_text("<table><tr><th>Jetzt ungepruefter Inhalt</th></tr></table>")
    store, hashes, chunks = Mock(), Mock(), Mock()
    with pytest.raises(ValueError, match="seit Vorpruefung geaendert"):
        _cleanup_stale_chunks.__wrapped__(config, store, hashes, chunks, 12, MODEL, [str(path)])
    store.get_ids_by_sources.assert_not_called()
    store.delete_by_sources.assert_not_called()


@pytest.fixture
def html_job(test_db, s4_guard_passthrough, tmp_path, monkeypatch):
    import kiara.web.ingestion_worker as module

    config = tmp_path / "test.yaml"
    config.write_text("{}")
    monkeypatch.setenv("KIARA_CONFIG", str(config))
    monkeypatch.setattr(module, "safe_refresh", lambda *a, **kw: None)
    ds = test_db.execute(
        "INSERT INTO datasources(name,type,path,is_active) VALUES ('html selection','local',%s,true) RETURNING id",
        (str(tmp_path),),
    ).fetchone()["id"]
    test_db.execute(
        "INSERT INTO file_hashes(datasource_id,embedding_model,file_path,file_hash) VALUES(%s,%s,%s,%s)",
        (ds, MODEL, str(tmp_path / "a.html"), "old-hash"),
    )
    test_db.commit()
    job = module.enqueue_job(ds, embedding_model=MODEL, html_reingest="all")
    return module, ds, job


@pytest.mark.parametrize("manual", [False, True])
def test_real_queue_retry_preserves_original_request_scope_and_contract(
    html_job, test_db, tmp_path, manual
):
    from kiara.ingestion.html_migration import build_reingest_selection
    from kiara.ingestion.html_reingest import build_request

    module, ds, job = html_job
    paths = [str(tmp_path / "a.html")]
    stats = {
        "html_reingest": build_request("all"),
        "html_reingest_candidates": paths,
        "html_migration": build_reingest_selection(ds, MODEL, paths),
    }
    test_db.execute(
        "UPDATE ingestion_jobs SET status='error', stats_json=%s, requested_action=%s WHERE id=%s",
        (json.dumps(stats), "cancel" if manual else None, job),
    )
    test_db.commit()
    old = dict(test_db.execute("SELECT * FROM ingestion_jobs WHERE id=%s", (job,)).fetchone())
    if manual:
        new_id = module.retry_html_job(job)
    else:
        worker = module.IngestionWorker.__new__(module.IngestionWorker)
        assert worker._recover_single_job(None, old) is True
        new_id = test_db.execute(
            "SELECT id FROM ingestion_jobs WHERE resumed_from_job=%s", (job,)
        ).fetchone()["id"]
    new = dict(test_db.execute("SELECT * FROM ingestion_jobs WHERE id=%s", (new_id,)).fetchone())
    new_stats = (
        json.loads(new["stats_json"]) if isinstance(new["stats_json"], str) else new["stats_json"]
    )
    assert new_stats == {**stats, "resumed_from_job": job}
    assert new["resumed_from_job"] == job
    assert new["embedding_model"] == MODEL
    assert new["job_type"] == "incremental"
    assert new["triggered_by"] == ("manual" if manual else "auto-resume")
    assert {k: v for k, v in old.items() if k.startswith("contract_")} == {
        k: v for k, v in new.items() if k.startswith("contract_")
    }
    with pytest.raises(ValueError, match="aktiven Job"):
        module.retry_html_job(job)


def test_retry_after_parser_change_requires_new_request(html_job, test_db):
    from kiara.ingestion.html_reingest import build_request

    module, _ds, job = html_job
    request = build_request("all")
    request["preparation_revision"][1] = "historical-profile"
    test_db.execute(
        "UPDATE ingestion_jobs SET status='error', stats_json=%s WHERE id=%s",
        (json.dumps({"html_reingest": request}), job),
    )
    test_db.commit()
    with pytest.raises(ValueError, match="Revision"):
        module.retry_html_job(job)
    assert (
        test_db.execute(
            "SELECT count(*) AS n FROM ingestion_jobs WHERE resumed_from_job=%s", (job,)
        ).fetchone()["n"]
        == 0
    )


def test_candidate_freeze_is_committed_before_first_preparation(
    html_job, test_db, tmp_path, monkeypatch
):
    from kiara.connectors.local_connector import LocalConnector
    from kiara.ingestion import html_reingest
    from kiara.retrieval import vectorstore

    module, ds, job = html_job
    path = str(tmp_path / "a.html")
    worker = module.IngestionWorker.__new__(module.IngestionWorker)
    worker._set_job_phase = Mock()
    config = {"_html_reingest": html_reingest.build_request("all")}
    monkeypatch.setattr(html_reingest, "collect_candidates", lambda *a, **kw: [path])
    monkeypatch.setattr(html_reingest, "assert_exclusive_sources", lambda *a, **kw: None)
    monkeypatch.setattr(vectorstore, "VectorStore", Mock())

    class Crash(BaseException):
        pass

    def fail_after_committed_freeze(*args, **kwargs):
        stored = test_db.execute(
            "SELECT stats_json FROM ingestion_jobs WHERE id=%s", (job,)
        ).fetchone()["stats_json"]
        if isinstance(stored, str):
            stored = json.loads(stored)
        assert stored["html_reingest_candidates"] == [path]
        raise Crash()

    monkeypatch.setattr(html_reingest, "prepare_reingestion", fail_after_committed_freeze)
    with pytest.raises(Crash):
        worker._run_html_reingest_phase(
            job,
            ds,
            config,
            LocalConnector,
            {"path": str(tmp_path)},
            {"id": ds, "type": "local"},
            Mock(),
            lambda: None,
            MODEL,
        )


def test_html_job_keeps_mail_scan_marker_and_does_not_enable_other_model(monkeypatch, tmp_path):
    import kiara.config as config_module
    import kiara.connectors.registry as registry
    import kiara.web.ingestion_worker as module
    from kiara.ingestion.html_reingest import build_request

    marker = tmp_path / ".scan_done"
    marker.write_text("Originalmarker")
    connector = Mock(SCAN_CACHE_MAX_AGE=60)
    connector.effective_path.return_value = str(tmp_path)
    row = {"id": 12, "type": "imap", "ds_colbert_enabled": True}
    config = {"retrieval": {"colbert": {"enabled": True}}}
    monkeypatch.setattr(module, "_ds_repo", Mock(get_by_id=Mock(return_value=row)))
    monkeypatch.setattr(module, "load_connector_config", lambda *a: {})
    monkeypatch.setattr(config_module, "load_config", lambda: config)
    monkeypatch.setattr(registry, "ensure_loaded", lambda: None)
    monkeypatch.setattr(registry, "get", lambda _type: connector)
    worker = module.IngestionWorker.__new__(module.IngestionWorker)
    context = worker._load_job_context(
        {
            "embedding_model": MODEL,
            "triggered_by": "manual",
            "job_type": "incremental",
            "stats_json": {"html_reingest": build_request("all")},
        },
        12,
        99,
    )
    assert marker.read_text() == "Originalmarker"
    assert context["ds_row"]["ds_colbert_enabled"] is False
    assert context["config"]["retrieval"]["colbert"]["enabled"] is False
    assert row["ds_colbert_enabled"] is True
    assert config["retrieval"]["colbert"]["enabled"] is True


@pytest.mark.parametrize("signal", ["cancel", "shutdown"])
def test_inventory_cancellation_cannot_prepare_or_enter_pipeline(
    html_job, tmp_path, monkeypatch, signal
):
    from kiara.connectors.local_connector import LocalConnector
    from kiara.ingestion import html_reingest
    from kiara.ingestion._file_utils import _CancelledError

    module, ds, job = html_job
    worker = module.IngestionWorker.__new__(module.IngestionWorker)
    worker._set_job_phase = Mock()
    worker._finalize_cancelled = Mock()
    config = {"_html_reingest": html_reingest.build_request("all")}
    monkeypatch.setattr(html_reingest, "collect_candidates", Mock(side_effect=_CancelledError()))
    prepare = Mock(side_effect=AssertionError("Keine Aufbereitung nach Abbruch"))
    monkeypatch.setattr(html_reingest, "prepare_reingestion", prepare)
    control = Mock(side_effect=[None, signal])
    assert (
        worker._run_html_reingest_phase(
            job,
            ds,
            config,
            LocalConnector,
            {"path": str(tmp_path)},
            {"id": ds, "type": "local"},
            Mock(),
            control,
            MODEL,
        )
        is None
    )
    assert worker._finalize_cancelled.called == (signal == "cancel")
    assert "_html_migration" not in config
    prepare.assert_not_called()


def test_empty_selection_recounts_whole_datasource_instead_of_zeroing_stock(monkeypatch):
    import kiara.web.ingestion_worker as module
    from kiara.ingestion.html_migration import build_reingest_selection
    from kiara.ingestion.html_reingest import build_request

    stored = {
        "html_reingest": build_request("outdated"),
        "html_reingest_candidates": [],
        "html_migration": build_reingest_selection(12, MODEL, []),
    }
    db, jobs, emb, ds, recount = Mock(), Mock(), Mock(), Mock(), Mock()
    db.execute.return_value.fetchone.return_value = {"stats_json": stored}
    monkeypatch.setattr(module, "_job_repo", jobs)
    monkeypatch.setattr(module, "_emb_repo", emb)
    monkeypatch.setattr(module, "_ds_repo", ds)
    monkeypatch.setattr("kiara.ingestion.vectordb_ops._recount_datasource_embeddings", recount)
    worker = module.IngestionWorker.__new__(module.IngestionWorker)
    worker._update_datasource_metadata(
        db, 99, 12, {"processed_files": 0, "total_files": 0}, {}, MODEL, 25, "2026-09-30"
    )
    recount.assert_called_once_with(db, 12, MODEL)
    emb.upsert_stats.assert_not_called()
    assert json.loads(jobs.update.call_args.kwargs["stats_json"])["html_reingest_candidates"] == []


def test_cleanup_failure_invalidates_current_hash_before_mutations(tmp_path, monkeypatch, test_db):
    from kiara.ingestion._file_utils import HashStore
    from kiara.ingestion.document_publication import pending_source_cleanups
    from kiara.ingestion.html_migration import build_reingest_selection
    from kiara.ingestion.html_reingest import build_request, prepared_source_hash
    from kiara.ingestion.pipeline import _cleanup_stale_chunks

    path = str(tmp_path / "a.html")
    (tmp_path / "a.html").write_text("<p>Inhalt</p>")
    ds = test_db.execute(
        "INSERT INTO datasources(name,type) VALUES('HTML Hashanker','local') RETURNING id"
    ).fetchone()["id"]
    test_db.commit()
    config = {"_html_reingest": build_request("all")}
    config["_html_migration"] = {
        **build_reingest_selection(ds, MODEL, [path]),
        "paths": [path],
        "errors": {},
    }
    config["_html_reingest_prepared_hashes"] = {path: prepared_source_hash(config, path)}
    store, hashes, chunks = Mock(), HashStore(), Mock()
    store.get_collection_name.return_value = "html-precleanup-test"
    store.get_ids_by_sources.return_value = []
    chunks.get_other_datasources.return_value = []
    hashes.upsert_many(ds, MODEL, {path: ("current-hash", 13)})

    def interrupted_inventory(*a, **kw):
        assert not hashes.load(ds, MODEL), (
            "Vor Indexmutation darf kein gueltiger Abschlussmarker bleiben"
        )
        assert pending_source_cleanups("html-precleanup-test") == {path: {MODEL: []}}
        raise OSError("Cleanup-Inventar nicht erreichbar")

    monkeypatch.setattr("kiara.ingestion.pipeline._cleanup_source_batch", interrupted_inventory)
    with pytest.raises(OSError, match="Cleanup-Inventar"):
        _cleanup_stale_chunks.__wrapped__(config, store, hashes, chunks, ds, MODEL, [path])
    assert not hashes.load(ds, MODEL)
    store.delete_by_sources.assert_not_called()


@pytest.fixture
def cleanup_stock(html_job, test_db, tmp_path):
    from kiara.ingestion._file_utils import ChunkDatasources, HashStore
    from kiara.ingestion.html_migration import build_reingest_selection
    from kiara.ingestion.html_reingest import build_request, prepared_source_hash

    module, ds, job = html_job
    path = str(tmp_path / "a.html")
    (tmp_path / "a.html").write_text("<p>Review: bestehender HTML-Bestand</p>")
    config = {"_html_reingest": build_request("all")}
    config["_html_migration"] = {
        **build_reingest_selection(ds, MODEL, [path]),
        "paths": [path],
        "errors": {},
    }
    config["_html_reingest_prepared_hashes"] = {path: prepared_source_hash(config, path)}
    test_db.execute(
        "INSERT INTO chunk_datasources(chunk_id,datasource_id,embedding_model) VALUES('html-review-chunk',%s,%s)",
        (ds, MODEL),
    )
    test_db.execute(
        "INSERT INTO datasource_embeddings(datasource_id,embedding_model,chunk_count,file_count) VALUES(%s,%s,1,1)",
        (ds, MODEL),
    )
    test_db.commit()
    store = Mock()
    store.get_ids_by_sources.return_value = ["html-review-chunk"]
    store.get_documents_by_ids.return_value = {
        "ids": ["html-review-chunk"],
        "metadatas": [{"source": path, "ds_ids": [ds]}],
    }
    return module, ds, job, config, path, store, HashStore(), ChunkDatasources()


def _assert_stock_totals(db, ds, expected_files, expected_chunks):
    actual = db.execute(
        "SELECT (SELECT count(*) FROM file_hashes WHERE datasource_id=%s AND embedding_model=%s) AS files, (SELECT count(*) FROM chunk_datasources WHERE datasource_id=%s AND embedding_model=%s) AS chunks",
        (ds, MODEL, ds, MODEL),
    ).fetchone()
    stored = db.execute(
        "SELECT file_count,chunk_count FROM datasource_embeddings WHERE datasource_id=%s AND embedding_model=%s",
        (ds, MODEL),
    ).fetchone()
    assert (actual["files"], actual["chunks"]) == (expected_files, expected_chunks)
    assert ((stored["file_count"], stored["chunk_count"]) if stored else (0, 0)) == (
        expected_files,
        expected_chunks,
    )


@pytest.mark.parametrize("failure", ["elasticsearch", "qdrant"])
@pytest.mark.parametrize("unselected", [False, True])
def test_cleanup_failure_keeps_real_statistics_consistent(
    cleanup_stock, test_db, monkeypatch, failure, unselected
):
    from kiara.ingestion.pipeline import _cleanup_stale_chunks

    module, ds, job, config, path, store, hashes, chunks = cleanup_stock
    if unselected:
        test_db.execute(
            "INSERT INTO file_hashes(datasource_id,embedding_model,file_path,file_hash) VALUES(%s,%s,'/retained.txt','retained')",
            (ds, MODEL),
        )
        test_db.execute(
            "INSERT INTO chunk_datasources(chunk_id,datasource_id,embedding_model) VALUES('retained',%s,%s)",
            (ds, MODEL),
        )
        test_db.execute(
            "UPDATE datasource_embeddings SET file_count=2,chunk_count=2 WHERE datasource_id=%s AND embedding_model=%s",
            (ds, MODEL),
        )
        test_db.commit()
    expected = int(unselected)
    failure_totals = []

    def fail(*args, **kwargs):
        # Capture committed counters before worker finalization can repair them.
        row = test_db.execute(
            "SELECT file_count,chunk_count FROM datasource_embeddings WHERE datasource_id=%s AND embedding_model=%s",
            (ds, MODEL),
        ).fetchone()
        failure_totals.append((row["file_count"], row["chunk_count"]) if row else (0, 0))
        raise OSError(failure)

    from kiara.retrieval import elasticsearch_service as es

    monkeypatch.setattr(es, "model_uses_elasticsearch", lambda *a: True)
    monkeypatch.setattr(es, "targets_for_datasources", lambda *a: ())
    monkeypatch.setattr(es, "get_ids_by_source", lambda *a, **kw: [])
    monkeypatch.setattr(es, "delete_by_ids", fail if failure == "elasticsearch" else Mock())
    if failure == "qdrant":
        store.delete_by_sources.side_effect = fail
    with pytest.raises(OSError, match=failure):
        _cleanup_stale_chunks.__wrapped__(config, store, hashes, chunks, ds, MODEL, [path])
    worker = module.IngestionWorker.__new__(module.IngestionWorker)
    worker._progress_cache = None
    worker._finalize_error(job, ds, failure, config=config, embedding_model=MODEL)
    _assert_stock_totals(test_db, ds, expected, expected)
    assert failure_totals == [(expected, expected)]
    assert (
        test_db.execute("SELECT status FROM ingestion_jobs WHERE id=%s", (job,)).fetchone()[
            "status"
        ]
        == "error"
    )
    if unselected:
        assert hashes.load(ds, MODEL) == {"/retained.txt": "retained"}


def test_counter_failure_rolls_back_hash_invalidation(cleanup_stock, test_db, monkeypatch):
    from kiara.ingestion.pipeline import _cleanup_stale_chunks

    _module, ds, _job, config, path, store, hashes, chunks = cleanup_stock
    recount = Mock(side_effect=RuntimeError("counter transaction failed"))
    monkeypatch.setattr("kiara.ingestion.vectordb_ops._recount_datasource_embeddings", recount)
    with pytest.raises(RuntimeError, match="counter transaction failed"):
        _cleanup_stale_chunks.__wrapped__(config, store, hashes, chunks, ds, MODEL, [path])
    _assert_stock_totals(test_db, ds, 1, 1)
    assert hashes.load(ds, MODEL) == {path: "old-hash"}
    store.delete_by_sources.assert_not_called()
