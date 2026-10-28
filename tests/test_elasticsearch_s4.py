"""S4: real isolated PG memberships/QdrantLocal, controlled ES failures."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from kiara.retrieval import elasticsearch_inventory as inv
from kiara.retrieval import elasticsearch_maintenance as maintenance
from kiara.retrieval import elasticsearch_service as es
from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore


@pytest.fixture
def corpus(test_db, tmp_path, monkeypatch, s4_guard_passthrough, request):
    model = "nomic-embed-text"
    monkeypatch.setattr(inv, "get_embedding_model", lambda config: model)
    test_db.execute(
        "INSERT INTO datasources (id,name,type) VALUES (0,'CLI','local') ON CONFLICT DO NOTHING"
    )
    ids = [0]
    for name in ["S4-B", "S4-empty"]:
        ids.append(
            test_db.execute(
                "INSERT INTO datasources (name,type) VALUES (%s,'local') RETURNING id", (name,)
            ).fetchone()["id"]
        )
    test_db.commit()
    config = {
        "ollama": {"embedding_model": model},
        "retrieval": {
            "hybrid_search": "elasticsearch",
            "elasticsearch_index": "s4_test",
            "elasticsearch_maintenance_file": str(tmp_path / "maintenance.json"),
        },
    }
    client = QdrantClient(location=":memory:")
    collection = "kb_nomic_embed_text"
    client.create_collection(
        collection, vectors_config=VectorParams(size=2, distance=Distance.COSINE)
    )
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store._client, store._collection, store._embed_model = client, collection, model
    store._config = config
    store._index_contract_guard = lambda **kwargs: nullcontext()
    points = []
    for number in range(getattr(request, "param", 12)):
        cid = f"{number + 1:032x}"
        owners = [0, ids[1]] if number < 4 else [0]
        points.append(
            PointStruct(
                id=cid,
                vector=[1.0, 0.0],
                payload={
                    "_text": f"Kanonischer Text {number} ä",
                    "ds_ids": owners,
                    "source": f"s4/{number}.txt",
                    "chunk_index": number,
                },
            )
        )
        for owner in owners:
            test_db.execute(
                "INSERT INTO chunk_datasources (chunk_id,datasource_id,embedding_model) VALUES (%s,%s,%s)",
                (cid, owner, model),
            )
    # Another model's memberships must never produce ES copies/counts.
    test_db.execute(
        "INSERT INTO chunk_datasources (chunk_id,datasource_id,embedding_model) VALUES ('foreign',%s,'other-model')",
        (ids[1],),
    )
    test_db.commit()
    client.upsert(collection, points)
    monkeypatch.setattr(inv, "_store", lambda config: store)
    monkeypatch.setattr(maintenance, "assert_stopped", lambda: None)
    yield SimpleNamespace(
        config=config, store=store, client=client, ids=ids, model=model, db=test_db, points=points
    )
    client.close()


class Backend:
    def __init__(self):
        self.data = {
            "s4_test": {"old": {"text": "old corpus"}},
            "unrelated_ds_9": {"foreign": {"text": "foreign"}},
        }
        self.indices = SimpleNamespace(
            exists=self.exists,
            create=self.create,
            delete=self.delete,
            get_settings=self.settings,
            stats=self.stats,
            refresh=self.refresh,
        )
        self.fail_refresh = False
        self.fail_bulk = False
        self.corrupt = False
        self.bulk_sizes = []

    def info(self):
        return {"version": {"number": "controlled"}}

    def exists(self, index):
        return index in self.data

    def create(self, index, **kwargs):
        self.data[index] = {}
        return {"acknowledged": True, "shards_acknowledged": True}

    def delete(self, index, **kwargs):
        del self.data[index]
        return {"acknowledged": True}

    def settings(self, index, **kwargs):
        return {
            name: {"settings": {"index": {"uuid": name + "-uuid", "max_result_window": "10000"}}}
            for name in index
            if name in self.data
        }

    def stats(self, index, **kwargs):
        return {
            "_shards": {"total": len(index), "successful": len(index), "failed": 0},
            "indices": {
                name: {
                    "uuid": name + "-uuid",
                    "primaries": {"docs": {"count": len(self.data[name])}},
                }
                for name in index
            },
        }

    def refresh(self, index, **kwargs):
        if self.fail_refresh:
            raise RuntimeError("injected refresh failure")
        return {"_shards": {"total": len(index), "successful": len(index), "failed": 0}}

    def mget(self, docs, **kwargs):
        result = []
        for doc in docs:
            source = self.data[doc["_index"]].get(doc["_id"])
            result.append({**doc, "found": source is not None, "_source": deepcopy(source)})
        if self.corrupt and result:
            result[0]["_source"]["text"] = "wrong text"
        return {"docs": result}

    def stream(self, client, actions, chunk_size, **kwargs):
        count = 0
        for action in actions:
            count += 1
            ok = not self.fail_bulk or count != 2
            if ok:
                self.data[action["_index"]][action["_id"]] = deepcopy(action["_source"])
            yield (
                ok,
                {
                    "index": {
                        "_index": action["_index"],
                        "_id": action["_id"],
                        "status": 201 if ok else 500,
                    }
                },
            )
        self.bulk_sizes.append(chunk_size)


@pytest.fixture
def backend(monkeypatch):
    backend = Backend()
    monkeypatch.setattr(es, "_get_client", lambda config: backend)
    monkeypatch.setattr("elasticsearch.helpers.streaming_bulk", backend.stream)
    return backend


def run(corpus, **kwargs):
    if not maintenance.marker_path(corpus.config).exists():
        maintenance.begin(corpus.config)
    with maintenance.offline_session(corpus.config):
        return es.rebuild_from_vectorstore(corpus.config, batch_size=3, **kwargs)


def test_preview_exact_copies_and_empty_without_writes(corpus, backend):
    before = deepcopy(backend.data)
    result = es.rebuild_from_vectorstore(corpus.config, dry_run=True, batch_size=3)
    assert result["qdrant_count"] == 12
    assert result["expected_physical_documents"] == 16
    assert {r["datasource_id"]: r["expected"] for r in result["datasources"]} == dict(
        zip(corpus.ids, [12, 4, 0], strict=True)
    )
    assert [r["current_state"] for r in result["datasources"]] == [
        "missing_expected",
        "missing_expected",
        "absent_empty",
    ]
    assert result["index_binding"]["contract_index_compatibility_id"]
    assert backend.data == before
    assert not maintenance.marker_path(corpus.config).exists()


def test_rebuild_full_text_metadata_and_physical_counts(corpus, backend):
    result = run(corpus)
    assert result["status"] == "success", result
    assert result["attempted"] == result["succeeded"] == 16
    assert result["failed"] == result["unconfirmed"] == 0
    assert result["verification_result"]["sampled"] == 16
    assert result["release_ready"]
    assert maintenance.marker_path(corpus.config).exists()  # rebuild never releases
    assert backend.data["s4_test"] == {"old": {"text": "old corpus"}}
    assert backend.data["unrelated_ds_9"] == {"foreign": {"text": "foreign"}}
    with maintenance.offline_session(corpus.config):
        status = es.get_status(corpus.config)
    assert status["counts_match"] and status["doc_count"] == 16


@pytest.mark.parametrize("fault", ["fail_bulk", "fail_refresh", "corrupt"])
def test_failure_never_releases_and_full_restart_succeeds(corpus, backend, fault):
    setattr(backend, fault, True)
    failed = run(corpus)
    assert failed["status"] == "error" and not failed["release_ready"]
    assert maintenance.marker_path(corpus.config).exists()
    assert backend.data["s4_test"] == {"old": {"text": "old corpus"}}
    setattr(backend, fault, False)
    assert run(corpus)["release_ready"]
    assert backend.bulk_sizes[-1] == 3


def test_controlled_cancel_and_retry(corpus, backend):
    calls = []

    def cancel():
        calls.append(1)
        return len(calls) == 3

    result = run(corpus, cancel=cancel)
    assert result["status"] == "error" and "abgebrochen" in result["error"]
    assert not result["release_ready"]
    assert run(corpus)["release_ready"]


@pytest.mark.parametrize("fault", ["orphan", "mismatch", "pending", "missing_pg_vector"])
def test_inconsistent_canonical_data_fails_before_es_mutation(corpus, backend, fault):
    point = corpus.points[0]
    if fault == "orphan":
        corpus.db.execute(
            "DELETE FROM chunk_datasources WHERE chunk_id=%s AND embedding_model=%s",
            (str(point.id), corpus.model),
        )
        corpus.db.commit()
    elif fault == "missing_pg_vector":
        corpus.client.delete(corpus.store._collection, points_selector=[point.id])
    else:
        payload = {"ds_ids": [0]} if fault == "mismatch" else {"processing_pending": True}
        corpus.client.set_payload(corpus.store._collection, payload, points=[point.id])
    before = deepcopy(backend.data)
    expected = {
        "orphan": "ohne bestaetigte",
        "mismatch": "widersprechen",
        "pending": "nicht abgeschlossen",
        "missing_pg_vector": "Mengen weichen",
    }
    with pytest.raises(Exception, match=expected[fault]):
        run(corpus)
    assert before == backend.data
    assert maintenance.marker_path(corpus.config).exists()


def test_sample_cannot_release_and_missing_expected_index_not_zero(corpus, backend):
    assert run(corpus)["release_ready"]
    with maintenance.offline_session(corpus.config):
        result = es.verify_integrity(corpus.config, sample_size=2)
        assert result["sampled"] == 2 and not result["release_ready"]
        del backend.data["s4_test_ds_0"]
        status = es.get_status(corpus.config)
    assert status["error"] and status["doc_count"] is None


def test_marker_survives_exception_and_rejects_new_session_config(corpus):
    maintenance.begin(corpus.config)
    with pytest.raises(RuntimeError, match="gesperrt"):
        maintenance.assert_access(corpus.config)
    with maintenance.offline_session(corpus.config):
        maintenance.assert_access(corpus.config)
        with pytest.raises(BlockingIOError), maintenance.offline_session(corpus.config):
            pass
    changed = deepcopy(corpus.config)
    changed["retrieval"]["elasticsearch_index"] = "different"
    with pytest.raises(RuntimeError, match="Config/Modell"), maintenance.offline_session(changed):
        pass
    with pytest.raises(RuntimeError):
        es.rebuild_from_vectorstore(corpus.config)


def test_admin_does_not_promote_normal_error_result(monkeypatch):
    from kiara.admin.services import elasticsearch_service as admin

    monkeypatch.setattr("kiara.config.load_config", lambda: {})
    monkeypatch.setattr(
        es, "rebuild_from_vectorstore", lambda *a, **kw: {"status": "error", "failed": 2}
    )
    admin._run_rebuild()
    assert admin.get_task_status()["result"]["status"] == "error"
    monkeypatch.setattr(
        es, "verify_integrity", lambda *a, **kw: {"status": "error", "missing_in_es": 1}
    )
    admin._run_verify()
    assert admin.get_task_status()["result"]["status"] == "error"


def test_admin_online_rebuild_denied(monkeypatch):
    from kiara.admin.services import elasticsearch_service as admin

    monkeypatch.setattr(
        "kiara.config.load_config", lambda: {"retrieval": {"hybrid_search": "elasticsearch"}}
    )
    with pytest.raises(RuntimeError, match="Offline"):
        admin.start_rebuild()
    assert not admin.is_running()


@pytest.mark.parametrize("corrupt", [False, True])
def test_release_rechecks_full_content_and_persists_gate(
    corpus, backend, monkeypatch, tmp_path, corrupt
):
    import json
    import sys

    assert run(corpus)["release_ready"]
    backend.corrupt = corrupt
    report = tmp_path / "release.json"
    monkeypatch.setattr("kiara.config.load_config", lambda path: corpus.config)
    monkeypatch.setattr("kiara.shared.db.init_db_backend", lambda config: None)
    monkeypatch.setattr(
        sys, "argv", ["maintenance", "release", "--config", "fixture.yaml", "--report", str(report)]
    )
    if corrupt:
        with pytest.raises(SystemExit):
            maintenance.main()
        assert maintenance.marker_path(corpus.config).exists()
        assert not json.loads(report.read_text())["release_ready"]
    else:
        maintenance.main()
        assert not maintenance.marker_path(corpus.config).exists()
        assert json.loads(report.read_text())["release_ready"]


def test_repair_uses_complete_addresses_and_rejects_unconfirmed_owner(corpus, monkeypatch):
    from kiara.retrieval.elasticsearch_indices import document_address

    cid = str(corpus.points[0].id)
    confirmed = [0, corpus.ids[1]]
    update = MagicMock(return_value=2)
    monkeypatch.setattr(es, "update_ds_ids", update)
    assert es.repair_memberships(corpus.config, {cid: confirmed}, corpus.model) == 2
    assert update.call_args.args[1] == {
        document_address(corpus.config, ds, cid): confirmed for ds in confirmed
    }
    update.reset_mock()
    with pytest.raises(RuntimeError, match="bestaetigte PG"):
        es.repair_memberships(corpus.config, {cid: [0]}, corpus.model)
    update.assert_not_called()
    assert es.repair_memberships(corpus.config, {cid: confirmed}, "other-model") == 0
    update.assert_not_called()


def test_publication_record_blocks_rebuild_even_with_clear_payload(corpus, backend):
    import json

    from kiara.retrieval.embedding import collection_name_for_model

    source = corpus.points[0].payload["source"]
    corpus.db.execute(
        "INSERT INTO document_publications (collection_name,source,versions,generation) VALUES (%s,%s,%s::jsonb,1)",
        (collection_name_for_model(corpus.model), source, json.dumps({"standard": "pending"})),
    )
    corpus.db.commit()
    result = run(corpus)
    assert result["status"] == "error" and not result["release_ready"]
    assert result["reason_code"] == "open_document_publications"
    assert result["open_publications"] == [source]
    assert set(backend.data) == {"s4_test", "unrelated_ds_9"}


def test_stopped_services_required(monkeypatch):
    monkeypatch.setattr(
        maintenance.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="active\n")
    )
    with pytest.raises(RuntimeError, match="gestoppten Dienst"):
        maintenance.assert_stopped()


def test_maintenance_blocks_real_client_construction(corpus, monkeypatch):
    maintenance.begin(corpus.config)
    constructor = MagicMock()
    monkeypatch.setattr("elasticsearch.Elasticsearch", constructor)
    with pytest.raises(RuntimeError, match="gesperrt"):
        es._get_client(corpus.config)
    constructor.assert_not_called()


def test_new_process_remains_blocked(corpus):
    import subprocess
    import sys

    maintenance.begin(corpus.config)
    code = """
import sys
from kiara.retrieval.elasticsearch_maintenance import assert_access
try:
    assert_access({'retrieval': {'elasticsearch_maintenance_file': sys.argv[1]}})
except RuntimeError as exc:
    assert 'gesperrt' in str(exc)
    raise SystemExit(23)
raise SystemExit(0)
"""
    process = subprocess.run(
        [sys.executable, "-c", code, str(maintenance.marker_path(corpus.config))],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 23, process.stderr
    assert maintenance.marker_path(corpus.config).exists()


def test_same_global_count_does_not_hide_per_source_loss(corpus, backend):
    assert run(corpus)["release_ready"]
    source = backend.data["s4_test_ds_0"]
    cid, doc = source.popitem()
    backend.data[f"s4_test_ds_{corpus.ids[1]}"][cid] = doc
    with maintenance.offline_session(corpus.config):
        status = es.get_status(corpus.config)
    assert status["doc_count"] == 16 and not status["counts_match"]
    assert status["error"]


def test_new_cli_ingestion_is_blocked_before_owner_or_model_lookup(corpus, monkeypatch):
    from kiara.ingestion import pipeline

    maintenance.begin(corpus.config)
    owner = MagicMock()
    model = MagicMock()
    monkeypatch.setattr(pipeline, "_ensure_cli_datasource", owner)
    monkeypatch.setattr(pipeline, "get_embedding_model", model)
    with pytest.raises(RuntimeError, match="gesperrt"):
        pipeline.run_ingestion(corpus.config)
    owner.assert_not_called()
    model.assert_not_called()


def test_existing_large_file_diagnostic_never_adds_copies_across_sources(monkeypatch):
    from kiara.integrity.checks import check_es_duplicates
    from kiara.retrieval.elasticsearch_indices import DatasourceIndexTarget

    targets = (
        DatasourceIndexTarget(1, "kiara_ds_1", 3500),
        DatasourceIndexTarget(2, "kiara_ds_2", 3500),
    )
    monkeypatch.setattr(es, "search_scope", lambda *a: nullcontext(targets))
    monkeypatch.setattr(es, "_get_client", lambda *a: object())
    seen = []

    def search(client, targets, body):
        seen.append(targets)
        return {
            "hits": {"total": {"value": 3500}},
            "aggregations": {
                "sources": {"buckets": [{"key": "same/file.txt", "doc_count": 3500}]},
                "unique_sources": {"value": 1},
            },
        }

    monkeypatch.setattr(es, "search_explicit_targets", search)
    result = check_es_duplicates({"retrieval": {"hybrid_search": "elasticsearch"}})
    assert result["total_docs"] == 7000 and result["suspected_duplicates"] == 0
    assert result["status"] == "ok"
    assert seen == [(targets[0],), (targets[1],)]


def test_administrative_delete_requires_offline_session_and_keeps_old_index(corpus, backend):
    with pytest.raises(RuntimeError, match="Offline-Wartung"):
        es.delete_index(corpus.config)
    assert run(corpus)["release_ready"]
    with maintenance.offline_session(corpus.config):
        assert es.delete_index(corpus.config)
        assert es.get_status(corpus.config)["error"]
    assert set(backend.data) == {"s4_test", "unrelated_ds_9"}
    assert maintenance.marker_path(corpus.config).exists()
