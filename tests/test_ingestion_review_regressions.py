"""P50 review regressions: real PG/scan/hash/worker with controlled backends."""

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from kiara_ingestion_core.index_state import ChunkDatasources, HashStore
from test_ingestion_membership_visibility import MemoryES, MemoryStore, context

from kiara.ingestion import pipeline, runtime
from kiara.retrieval import elasticsearch_service as es
from kiara.retrieval.index_compatibility import IndexCompatibilityError, dense_index_target
from kiara.shared.db import db_query, db_transaction


class VersionedES(MemoryES):
    """Realtime reads and conditional mutations use actual evolving versions."""

    def __init__(self):
        super().__init__()
        self.versions = {}
        self.before_bulk = None
        self.after_bulk = None
        self.conflicts = 0
        self.actions = []

    def publish(self, cid, memberships):
        self.docs[cid] = {"metadata": {"ds_ids": list(memberships)}}
        self.versions[cid] = self.versions.get(cid, 0) + 1

    def mget(self, *, ids, **kwargs):
        response = super().mget(ids=ids, **kwargs)
        for doc in response["docs"]:
            doc["_seq_no"] = self.versions.get(doc["_id"], 0)
        return response

    def bulk(self, client, actions, **kwargs):
        actions = list(actions)
        self.actions.extend(copy.deepcopy(actions))
        if self.before_bulk:
            self.before_bulk()
        success, errors = 0, []
        for action in actions:
            cid = action["_id"]
            if "if_seq_no" in action and (
                action["if_seq_no"] != self.versions.get(cid, 0) or action["if_primary_term"] != 1
            ):
                self.conflicts += 1
                errors.append({action["_op_type"]: {"status": 409}})
                continue
            if action.get("_op_type") == "update":
                self.docs[cid]["metadata"].update(copy.deepcopy(action["doc"]["metadata"]))
                self.updates += 1
            elif action.get("_op_type") == "delete":
                self.docs.pop(cid, None)
            else:
                self.docs[cid] = copy.deepcopy(action["_source"])
            self.versions[cid] = self.versions.get(cid, 0) + 1
            success += 1
        if self.after_bulk:
            self.after_bulk()
        return success, errors


@pytest.fixture
def backend(monkeypatch):
    backend = VersionedES()
    monkeypatch.setattr(es, "_get_client", lambda config: backend)
    monkeypatch.setattr(es, "create_index", lambda config: True)
    monkeypatch.setattr("elasticsearch.helpers.bulk", backend.bulk)
    return backend


@pytest.fixture
def rollback_context(test_db, backend):
    for ds_id in (1, 2, 3, 4):
        test_db.execute(
            "INSERT INTO datasources(id,name,type) VALUES (%s,%s,'local')",
            (ds_id, f"P50-review-{ds_id}"),
        )
    test_db.commit()
    sot = ChunkDatasources()
    for ds_id in (1, 2, 3):
        sot.add_associations(["shared"], ds_id, "p50")
    ictx = context(backend, {}, {"shared": [1, 2, 3]})
    ictx.chunk_ds = sot
    return ictx


@pytest.mark.parametrize("es_memberships", [[1], [1, 2, 3], [2, 3], []])
def test_shared_rollback_preserves_document_when_payload_lags_pg(
    rollback_context, backend, es_memberships
):
    ictx = rollback_context
    # PG has committed foreign memberships; backend publication can lag.
    ictx.store.mapping = {"shared": [1], "orphan": [1]}
    ictx.chunk_ds.add_associations(["orphan"], 1, "p50")
    original = {
        "text": "Shared document text must survive every rollback stage.",
        "metadata": {"source": "/original.txt", "ds_ids": es_memberships, "chunk_index": 7},
        "extra": {"language": "de"},
    }
    backend.docs["shared"] = copy.deepcopy(original)
    backend.publish("orphan", [1])
    backend.docs["orphan"]["text"] = "This document really is an orphan."
    with es.ingestion_visibility(ictx.config):
        result = runtime._rollback_embedding_persistence(
            ictx,
            "/failed.txt",
            ["shared", "orphan"],
            rollback_elasticsearch=True,
            rollback_db_associations=True,
            context="lagging_shared_payload",
        )
    assert "shared" in backend.docs, "PG-shared document must never be physically deleted"
    assert not any(
        action["_id"] == "shared" and action["_op_type"] == "delete" for action in backend.actions
    )
    assert result is True
    expected = copy.deepcopy(original)
    expected["metadata"]["ds_ids"] = [2, 3]
    assert backend.docs == backend.visible == {"shared": expected}
    assert ictx.store.mapping == {"shared": [2, 3]}
    assert ictx.chunk_ds.get_datasource_ids_for_chunks(["shared", "orphan"], "p50") == {
        "shared": [2, 3]
    }
    assert any(
        action["_id"] == "orphan" and action["_op_type"] == "delete" for action in backend.actions
    )


@pytest.mark.parametrize("failure", ["cas_conflict", "before_write", "after_write"])
def test_shared_rollback_failure_preserves_lagging_document(rollback_context, backend, failure):
    ictx = rollback_context
    ictx.store.mapping["shared"] = [1]
    backend.publish("shared", [1])
    backend.docs["shared"]["text"] = "Keep the existing text even if reconciliation fails."
    backend.docs["shared"]["metadata"]["source"] = "/shared.txt"

    def conflict():
        # A competing ES update changes its version without losing its content.
        backend.docs["shared"]["external_revision"] = 1
        backend.versions["shared"] += 1

    def outage():
        raise es.ElasticsearchPartialFailureError("injected membership write outage")

    if failure == "cas_conflict":
        backend.before_bulk = conflict
    elif failure == "before_write":
        backend.before_bulk = outage
    else:
        backend.after_bulk = outage
    with es.ingestion_visibility(ictx.config):
        result = runtime._rollback_embedding_persistence(
            ictx,
            "/failed.txt",
            ["shared"],
            rollback_elasticsearch=True,
            rollback_db_associations=True,
            context="lagging_payload_failure",
        )
    assert result is False
    assert (
        backend.visible["shared"]["text"] == "Keep the existing text even if reconciliation fails."
    )
    assert backend.visible["shared"]["metadata"]["source"] == "/shared.txt"
    assert not any(action["_op_type"] == "delete" for action in backend.actions)
    assert ictx.chunk_ds.get_datasource_ids_for_chunks(["shared"], "p50") == {"shared": [1, 2, 3]}
    assert ictx.stats["rollback_errors"] == 1
    if failure == "cas_conflict":
        assert backend.conflicts == 1


def test_rollback_retains_es_if_orphan_gains_pg_owner_after_vector_delete(
    rollback_context,
    backend,
    monkeypatch,
):
    ictx = rollback_context
    ictx.store.mapping = {"orphan": [1]}
    ictx.chunk_ds.add_associations(["orphan"], 1, "p50")
    backend.publish("orphan", [1])
    backend.docs["orphan"]["text"] = "Keep the remaining document and PG retry anchors."
    original = copy.deepcopy(backend.docs)
    delete = ictx.store.delete_by_ids

    def delete_then_commit_foreign_owner(ids):
        delete(ids)
        # A foreign writer has committed PG but has not published its payloads.
        ictx.chunk_ds.add_associations(ids, 2, "p50")

    monkeypatch.setattr(ictx.store, "delete_by_ids", delete_then_commit_foreign_owner)
    with es.ingestion_visibility(ictx.config):
        result = runtime._rollback_embedding_persistence(
            ictx,
            "/failed.txt",
            ["orphan"],
            rollback_elasticsearch=True,
            rollback_db_associations=True,
            context="orphan_becomes_shared",
        )
    assert backend.docs == backend.visible == original
    assert not any(action["_op_type"] == "delete" for action in backend.actions)
    assert result is False  # The missing vector prevents a consistency claim.
    assert ictx.store.mapping == {}
    assert ictx.chunk_ds.get_datasource_ids_for_chunks(["orphan"], "p50") == {"orphan": [1, 2]}
    assert ictx.stats["rollback_errors"] == 1


@pytest.mark.parametrize("change", ["add", "remove"])
@pytest.mark.parametrize("boundary", ["pg_read", "qdrant_write", "es_write", "pg_delete"])
def test_rollback_reconciles_concurrent_membership(
    rollback_context, backend, monkeypatch, change, boundary
):
    ictx = rollback_context
    ready, published = threading.Event(), threading.Event()
    fired = False

    def writer():
        assert ready.wait(5)
        with db_transaction() as conn:
            if change == "add":
                ictx.chunk_ds.add_associations(["shared"], 4, "p50", conn=conn)
            else:
                conn.execute(
                    "DELETE FROM chunk_datasources WHERE chunk_id='shared' AND datasource_id=3"
                )
        memberships = ictx.chunk_ds.get_datasource_ids_for_chunks(["shared"], "p50")["shared"]
        ictx.store.mapping["shared"] = list(memberships)
        backend.publish("shared", memberships)
        published.set()

    def interleave():
        nonlocal fired
        if not fired:
            fired = True
            ready.set()
            assert published.wait(5), "Concurrent PG writer must not be fenced by backend work"

    if boundary == "pg_read":
        read = ictx.chunk_ds.get_datasource_ids_for_chunks

        def read_then_publish(*args, **kwargs):
            snapshot = read(*args, **kwargs)
            if threading.current_thread() is threading.main_thread():
                interleave()
            return snapshot

        monkeypatch.setattr(ictx.chunk_ds, "get_datasource_ids_for_chunks", read_then_publish)
    elif boundary == "qdrant_write":
        ictx.store.after_update = interleave
    elif boundary == "es_write":
        backend.after_bulk = interleave
    else:
        delete = runtime._rollback_db_associations

        def delete_then_publish(*args, **kwargs):
            result = delete(*args, **kwargs)
            interleave()
            return result

        monkeypatch.setattr(runtime, "_rollback_db_associations", delete_then_publish)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(writer)
        try:
            with es.ingestion_visibility(ictx.config):
                result = runtime._rollback_embedding_persistence(
                    ictx,
                    "/review.md",
                    ["shared"],
                    rollback_elasticsearch=True,
                    rollback_db_associations=True,
                    context="concurrent_rollback",
                )
        finally:
            ready.set()
        future.result(timeout=5)
    expected = [2, 3, 4] if change == "add" else [2]
    assert fired
    assert result is True
    assert ictx.chunk_ds.get_datasource_ids_for_chunks(["shared"], "p50") == {"shared": expected}
    assert ictx.store.mapping == {"shared": expected}
    assert backend.visible["shared"]["metadata"]["ds_ids"] == expected


def test_rollback_cas_conflict_preserves_retry_anchor(rollback_context, backend):
    ictx = rollback_context

    def publish():
        backend.before_bulk = None
        ictx.chunk_ds.add_associations(["shared"], 4, "p50")
        ictx.store.mapping["shared"] = [1, 2, 3, 4]
        backend.publish("shared", [1, 2, 3, 4])

    backend.before_bulk = publish
    result = runtime._rollback_embedding_persistence(
        ictx,
        "/review.md",
        ["shared"],
        rollback_elasticsearch=True,
        rollback_db_associations=True,
        context="cas_conflict",
    )
    assert result is False
    assert backend.conflicts == 1
    assert ictx.chunk_ds.get_datasource_ids_for_chunks(["shared"], "p50")["shared"] == [1, 2, 3, 4]
    assert backend.docs["shared"]["metadata"]["ds_ids"] == [1, 2, 3, 4]


def test_rollback_persistent_payload_drift_is_bounded(rollback_context, backend):
    ictx = rollback_context
    ictx.config["ingestion"] = {"ds_ids_sync_attempts": 2}
    ictx.store.after_update = lambda: ictx.store.mapping.update(shared=[1, 2, 3])
    result = runtime._rollback_embedding_persistence(
        ictx,
        "/review.md",
        ["shared"],
        rollback_elasticsearch=True,
        rollback_db_associations=True,
        context="persistent_drift",
    )
    assert result is False
    assert ictx.stats["rollback_errors"] == 1
    assert len(ictx.store.writes) == 2
    assert ictx.chunk_ds.get_datasource_ids_for_chunks(["shared"], "p50") == {"shared": [1, 2, 3]}


@pytest.mark.parametrize("hash_mode", ["mtime", "content"])
@pytest.mark.parametrize("failure", ["refresh", "missing_index", "empty_replacement"])
def test_unchanged_pipeline_retry_completes_failed_visibility(
    test_db,
    backend,
    tmp_path,
    monkeypatch,
    hash_mode,
    failure,
    s4_guard_passthrough,
):
    source = tmp_path / "source"
    source.mkdir()
    files = [source / "one.txt", source / "two.txt"]
    for file in files:
        file.write_text(f"A representative document: {file.name}")
    test_db.execute("INSERT INTO datasources(id,name,type) VALUES (1,'P50-retry','local')")
    test_db.commit()
    config = {
        "datasource": {"path": str(source), "file_types": [".txt"]},
        "vectordb": {"path": str(tmp_path / "vectors")},
        "ingestion": {"batch_size": 10, "chunk_size": 500, "chunk_overlap": 50},
        "retrieval": {"hybrid_search": "elasticsearch"},
        "graph": {"enabled": False},
    }
    store = MemoryStore({})
    store._index_contract_guard = lambda **kwargs: nullcontext()

    def add(texts, metadatas, ids, **kwargs):
        store.mapping.update(
            {cid: list(meta["ds_ids"]) for cid, meta in zip(ids, metadatas, strict=True)}
        )

    store.add = Mock(side_effect=add)
    monkeypatch.setattr(pipeline, "VectorStore", lambda *a, **k: store)
    load = Mock(
        side_effect=lambda path, *a, **kw: (
            ["document text"],
            [{"source": path}],
            [path],
            "document text",
            [(0, 13)],
        )
    )
    monkeypatch.setattr(runtime, "_process_single_file", load)
    failed_refresh = Mock(side_effect=es.ElasticsearchPartialFailureError("refresh unavailable"))
    if failure == "refresh":
        backend.indices.refresh = failed_refresh
    else:

        def lose_index_after_hash_commit(*args, **kwargs):
            # Real pipeline checkpoint and PG hashes are already durable.
            assert len(HashStore().load(1, "p50")) == 2
            assert len(backend.docs) == 2
            backend.docs.clear()
            backend.visible.clear()
            backend.indices.exists = Mock(return_value=failure == "empty_replacement")
            return True

        monkeypatch.setattr(
            pipeline, "_run_colbert_model_switch_cleanup", lose_index_after_hash_commit
        )
    message = {
        "refresh": "refresh unavailable",
        "missing_index": "Index .* fehlt",
        "empty_replacement": "ohne nachgewiesene Suchdaten",
    }[failure]

    def run():
        return pipeline.run_ingestion(
            config, incremental=True, datasource_id=1, embedding_model="p50", hash_mode=hash_mode
        )

    with pytest.raises(es.ElasticsearchPartialFailureError, match=message):
        run()
    hashes = HashStore().load(1, "p50")
    assert set(hashes) == {str(file) for file in files}
    assert load.call_count == 2
    if failure == "refresh":
        assert backend.docs and not backend.visible
        assert failed_refresh.call_count == 1  # A single barrier for two files.
    else:
        assert backend.docs == backend.visible == {}
        monkeypatch.setattr(pipeline, "_run_colbert_model_switch_cleanup", lambda *a, **k: True)

    # Even another failed retry must perform the barrier and remain an error.
    with pytest.raises(es.ElasticsearchPartialFailureError, match=message):
        run()
    assert HashStore().load(1, "p50") == hashes
    assert load.call_count == store.add.call_count == 2
    if failure != "refresh":
        assert backend.docs == backend.visible == {}
        assert backend.refreshes == (0 if failure == "missing_index" else 2)
        return
    assert failed_refresh.call_count == 2
    backend.indices.refresh = backend.refresh
    stats = run()
    assert stats["processed_files"] == 0
    assert stats["skipped_files"] == 2
    assert stats["errors"] == 0
    assert HashStore().load(1, "p50") == hashes
    assert backend.refreshes == 1
    assert backend.visible == backend.docs


@pytest.mark.parametrize("failure_mode", ["exception", "stats"])
@pytest.mark.parametrize("secondary", ["refresh", "missing_index"])
def test_worker_persists_primary_and_secondary_visibility_diagnostics(
    test_db, backend, monkeypatch, failure_mode, secondary
):
    from kiara.web.ingestion_worker import IngestionWorker

    test_db.execute("INSERT INTO datasources(id,name,type) VALUES (1,'P50-worker','local')")
    row = test_db.execute(
        "INSERT INTO ingestion_jobs(datasource_id,status,job_type,embedding_model) "
        "VALUES (1,'running','incremental','p50') RETURNING *"
    ).fetchone()
    test_db.commit()
    config = {"retrieval": {"hybrid_search": "elasticsearch"}}
    primary = IndexCompatibilityError(
        "index_contract_missing", "review_write", dense_index_target({}, "p50"), "primary failure"
    )
    backend.indices.refresh = Mock(
        side_effect=es.ElasticsearchPartialFailureError("secondary refresh outage")
    )

    @pipeline._refresh_search_on_exit
    def ingest(config):
        es.bulk_index_chunks(config, ["earlier file"], [{"ds_ids": [1]}], ["c"])
        if secondary == "missing_index":
            backend.docs.clear()
            backend.indices.exists = Mock(return_value=False)
        if failure_mode == "exception":
            raise primary
        return {
            "errors": 1,
            "abort_reason": f"Index-Kompatibilitaetsguard: {primary.reason_code}",
            "index_compatibility_error": primary.to_dict(),
            "error_details": [{"phase": "index_compatibility", "reason_code": primary.reason_code}],
        }

    def run(*args, **kwargs):
        return ingest(config), lambda: None

    worker = IngestionWorker()
    monkeypatch.setattr(
        worker,
        "_load_job_context",
        lambda *a: {
            "config": config,
            "connector_cls": SimpleNamespace(HASH_MODE="mtime"),
            "conn_config": {},
            "ds_row": {"id": 1, "type": "local"},
        },
    )
    monkeypatch.setattr(worker, "_mark_job_running", lambda *a: True)
    monkeypatch.setattr(worker, "_build_callbacks", lambda *a: {"on_progress": None})
    monkeypatch.setattr(worker, "_run_warmup", lambda *a: {})
    monkeypatch.setattr(worker, "_run_scan_phase", lambda *a: {})
    monkeypatch.setattr(worker, "_load_resume_skip_files", lambda *a: set())
    monkeypatch.setattr(worker, "_run_ingestion", run)
    monkeypatch.setattr("kiara.web.cache_refresh.IngestionCacheRefresh", Mock())
    monkeypatch.setattr("kiara.web.ingestion_worker.safe_refresh", Mock())
    worker._execute_job(row)
    with db_query() as conn:
        job = conn.execute("SELECT * FROM ingestion_jobs WHERE id=%s", (row["id"],)).fetchone()
    assert job["status"] == "error"
    stats = job["stats_json"]
    assert stats["index_compatibility_error"]["reason_code"] == "index_contract_missing"
    assert "index_contract_missing" in stats["abort_reason"]
    assert any(
        detail.get("reason_code") == "index_contract_missing" for detail in stats["error_details"]
    )
    assert any(
        detail.get("phase") == "elasticsearch_refresh"
        and ("secondary refresh outage" if secondary == "refresh" else "fehlt") in detail["error"]
        for detail in stats["error_details"]
    )
    assert "index_contract_missing" in job["error_message"]


def test_pipeline_preflight_rejection_does_not_refresh_old_writes(backend, monkeypatch, tmp_path):
    config = {
        "vectordb": {"path": str(tmp_path)},
        "ingestion": {},
        "retrieval": {"hybrid_search": "elasticsearch"},
    }
    primary = IndexCompatibilityError(
        "index_contract_missing", "review_preflight", dense_index_target({}, "p50"), "blocked"
    )
    backend.publish("old", [1])
    monkeypatch.setattr(pipeline, "VectorStore", Mock(side_effect=primary))
    with pytest.raises(IndexCompatibilityError) as caught:
        pipeline.run_ingestion(config, incremental=True, datasource_id=1, embedding_model="p50")
    assert caught.value is primary
    assert backend.refreshes == 0
    assert not backend.visible


def test_acknowledged_writes_require_index_at_barrier(backend):
    config = {"retrieval": {"hybrid_search": "elasticsearch"}}
    with pytest.raises(es.ElasticsearchPartialFailureError, match="Index .* fehlt"):
        with es.ingestion_visibility(config, refresh_on_success=False) as state:
            assert es.bulk_index_chunks(config, ["preserve me"], [{"ds_ids": [1]}], ["c"]) == 1
            assert state["dirty"]
            assert backend.docs["c"]["text"] == "preserve me"
            backend.docs.clear()
            backend.indices.exists = Mock(return_value=False)
    assert backend.refreshes == 0
    assert backend.docs == backend.visible == {}


@pytest.mark.parametrize("hybrid", ["elasticsearch", "none"])
def test_empty_first_pipeline_run_does_not_require_index(
    test_db, backend, monkeypatch, tmp_path, hybrid, s4_guard_passthrough
):
    test_db.execute("INSERT INTO datasources(id,name,type) VALUES (1,'P50-empty','local')")
    test_db.commit()
    backend.indices.exists = Mock(return_value=False)
    config = {
        "datasource": {"path": str(tmp_path), "file_types": [".txt"]},
        "vectordb": {"path": str(tmp_path / "vectors")},
        "ingestion": {"batch_size": 10},
        "retrieval": {"hybrid_search": hybrid},
        "graph": {"enabled": False},
    }
    monkeypatch.setattr(pipeline, "VectorStore", lambda *a, **k: MemoryStore({}))
    stats = pipeline.run_ingestion(config, incremental=True, datasource_id=1, embedding_model="p50")
    assert stats["errors"] == stats["processed_files"] == stats["skipped_files"] == 0
    assert HashStore().load(1, "p50") == {}
    assert backend.refreshes == 0
    backend.indices.exists.assert_not_called()


def test_disabled_es_skips_required_visibility(monkeypatch):
    client = Mock(side_effect=AssertionError("Disabled ES must not contact a backend"))
    monkeypatch.setattr(es, "_get_client", client)
    es.refresh_index({}, require_index=True, require_documents=True)
    client.assert_not_called()
