"""S2: physical DS addresses, CAS and real PG/Qdrant canonical publication."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_document_version_completion import document_run  # noqa: F401

from kiara.ingestion import pipeline, runtime
from kiara.retrieval import elasticsearch_service as es
from kiara.retrieval.elasticsearch_indices import document_address, resolve_datasource_targets

CONFIG = {"retrieval": {"hybrid_search": "elasticsearch", "elasticsearch_index": "s2"}}


def targets(config, *ids, expected=0):
    return resolve_datasource_targets(
        config, ids, allowed_ds_ids=ids, expected_document_counts={ds: expected for ds in ids}
    )


class Backend:
    """Index-aware realtime/visible stores with evolving CAS and fault injection."""

    def __init__(self):
        self.docs = {}
        self.visible = {}
        self.versions = {}
        self.refreshes = []
        self.actions = []
        self.fail_after = None
        self.before_bulk = None
        self.indices = SimpleNamespace(
            exists=lambda index: index in self.docs,
            create=self.create,
            get_settings=self.settings,
            stats=self.stats,
            refresh=self.refresh,
            delete=self.delete_index,
        )

    def create(self, *, index, **kw):
        self.docs.setdefault(index, {})
        self.visible.setdefault(index, {})
        return {"acknowledged": True}

    def delete_index(self, *, index):
        self.docs.pop(index, None)
        self.visible.pop(index, None)
        return {"acknowledged": True}

    def settings(self, *, index, **kw):
        return {i: {"settings": {"index": {"uuid": "uuid-" + i}}} for i in index if i in self.docs}

    def stats(self, *, index, **kw):
        return {
            "_shards": {"total": len(index), "successful": len(index), "failed": 0},
            "indices": {
                i: {"uuid": "uuid-" + i, "primaries": {"docs": {"count": len(self.visible[i])}}}
                for i in index
            },
        }

    def refresh(self, *, index, **kw):
        self.refreshes.append(list(index))
        for i in index:
            self.visible[i] = deepcopy(self.docs[i])
        return {"_shards": {"total": len(index), "successful": len(index), "failed": 0}}

    def mget(self, *, docs, **kw):
        result = []
        for a in docs:
            i, cid = a["_index"], a["_id"]
            if i not in self.docs:
                result.append({**a, "error": {"type": "index_not_found_exception"}})
            else:
                result.append(
                    {
                        **a,
                        "found": cid in self.docs[i],
                        "_source": deepcopy(self.docs[i].get(cid, {})),
                        "_seq_no": self.versions.get((i, cid), 0),
                        "_primary_term": 1,
                    }
                )
        return {"docs": result}

    def bulk(self, client, actions, **kw):
        actions = list(actions)
        if self.before_bulk:
            self.before_bulk()
        success = 0
        self.actions.extend(deepcopy(actions))
        for a in actions:
            if self.fail_after is not None and success >= self.fail_after:
                return success, [{"index": {"status": 500}}]
            i, cid = a["_index"], a["_id"]
            key = i, cid
            if "if_seq_no" in a and a["if_seq_no"] != self.versions.get(key, 0):
                return success, [{"delete": {"status": 409}}]
            if a.get("_op_type") == "delete":
                self.docs[i].pop(cid, None)
            elif a.get("_op_type") == "update":
                self.docs[i][cid]["metadata"].update(deepcopy(a["doc"]["metadata"]))
            else:
                self.docs[i][cid] = deepcopy(a["_source"])
            self.versions[key] = self.versions.get(key, 0) + 1
            success += 1
        return success, []

    def search(self, *, index, query, **kw):
        source = query["term"]["metadata.source.raw"]
        hits = [
            {"_index": i, "_id": cid}
            for i in index
            for cid, d in self.visible[i].items()
            if d["metadata"].get("source") == source
        ]
        return {
            "hits": {"total": {"relation": "eq", "value": len(hits)}, "hits": hits},
            "_shards": {"failed": 0},
        }

    def delete_by_query(self, *, index, query, **kw):
        sources = query["terms"]["metadata.source.raw"]
        count = 0
        for i in index:
            for cid, d in list(self.docs[i].items()):
                if d["metadata"].get("source") in sources:
                    del self.docs[i][cid]
                    count += 1
        self.refresh(index=index)
        return {"deleted": count}


@pytest.fixture
def backend(monkeypatch):
    backend = Backend()
    monkeypatch.setattr(es, "_get_client", lambda cfg: backend)
    monkeypatch.setattr("elasticsearch.helpers.bulk", backend.bulk)
    return backend


def write(ds, ids=("a", "b"), *, config=CONFIG, owners=None):
    return es.bulk_index_chunks(
        config,
        ["canonical " + cid for cid in ids],
        [{"source": "/file", "ds_ids": owners or [ds]} for _ in ids],
        list(ids),
        datasource_id=ds,
    )


@pytest.mark.parametrize("outcome", ["ok", "cancel", "error", "partial"])
def test_visibility_and_retry_only_touch_attempted_indices(backend, outcome):
    write(9)
    foreign = deepcopy(backend.docs["s2_ds_9"])
    backend.fail_after = 1 if outcome == "partial" else None
    try:
        with es.ingestion_visibility(CONFIG):
            write(1)
            if outcome == "error":
                raise ValueError("after write")
            if outcome == "cancel":
                raise KeyboardInterrupt()
    except (ValueError, KeyboardInterrupt, es.ElasticsearchPartialFailureError):
        assert outcome != "ok"
    assert backend.visible["s2_ds_1"] == backend.docs["s2_ds_1"]
    assert backend.refreshes == [["s2_ds_1"]]
    assert backend.docs["s2_ds_9"] == foreign
    backend.fail_after = None
    with es.ingestion_visibility(CONFIG):
        assert write(1) == 2
    assert len(backend.docs["s2_ds_1"]) == 2


def test_realtime_rollback_addresses_preserve_identical_foreign_ids(backend):
    write(2, owners=[1, 2])
    other = deepcopy(backend.docs["s2_ds_2"])
    with es.ingestion_visibility(CONFIG):
        write(1, owners=[1, 2])
        assert (
            es.delete_by_ids(CONFIG, [document_address(CONFIG, 1, cid) for cid in ("a", "b")]) == 2
        )
    assert backend.docs["s2_ds_1"] == {}
    assert backend.docs["s2_ds_2"] == other


def test_cas_does_not_delete_new_document_version(backend):
    write(1)
    backend.before_bulk = lambda: backend.versions.update({("s2_ds_1", "a"): 100})
    with pytest.raises(es.ElasticsearchPartialFailureError):
        es.delete_by_ids(CONFIG, [document_address(CONFIG, 1, "a")])
    assert "a" in backend.docs["s2_ds_1"]


def test_memberships_use_index_and_id_and_missing_index_is_fatal(backend):
    write(1, owners=[1, 2])
    write(2, owners=[1, 2])
    addresses = [document_address(CONFIG, ds, "a") for ds in (1, 2)]
    assert len(es.get_ds_ids(CONFIG, addresses)) == 2
    es.update_ds_ids(CONFIG, {addresses[0]: [1, 2, 3]}, refresh=False, skip_unchanged=True)
    assert backend.docs["s2_ds_2"]["a"]["metadata"]["ds_ids"] == [1, 2]
    with pytest.raises(es.ElasticsearchPartialFailureError):
        es.get_ds_ids(CONFIG, [document_address(CONFIG, 3, "a")])


def test_source_inventory_and_delete_are_explicit(backend):
    write(1)
    write(2)
    found = es.get_ids_by_source(CONFIG, "/file", targets=targets(CONFIG, 1, 2))
    assert len(found) == 4
    assert es.delete_by_sources(CONFIG, ["/file"], targets=targets(CONFIG, 1)) == 2
    assert len(backend.docs["s2_ds_2"]) == 2
    assert es.get_ids_by_source(CONFIG, "/file", targets=()) == []


@pytest.mark.parametrize(
    "ds,metas,ids",
    [
        (None, [{"ds_ids": [1]}], ["a"]),
        (True, [{"ds_ids": [1]}], ["a"]),
        (1, [{}], ["a"]),
        (1, [{"ds_ids": [2]}], ["a"]),
        (1, [], ["a"]),
    ],
)
def test_invalid_write_has_no_side_effects(backend, ds, metas, ids):
    with pytest.raises(ValueError):
        es.bulk_index_chunks(CONFIG, ["text"], metas, ids, datasource_id=ds)
    assert backend.docs == {}


def test_missing_ack_fails(backend, monkeypatch):
    monkeypatch.setattr("elasticsearch.helpers.bulk", lambda *a, **kw: (0, []))
    with pytest.raises(es.ElasticsearchPartialFailureError):
        write(0)


@pytest.fixture
def canonical(document_run, backend, monkeypatch):
    case = document_run
    case.store._config["retrieval"] = deepcopy(CONFIG["retrieval"])
    monkeypatch.setattr("kiara.config.get_embedding_model", lambda cfg: "test-model")
    monkeypatch.setattr(runtime, "_write_to_elasticsearch", _real_write)
    return case


_real_write = runtime._write_to_elasticsearch


def test_pg_qdrant_canonical_publication_and_new_owner(canonical, backend, test_db):
    case = canonical
    assert case.run()
    assert case.context.stats["processed_files"] == 1
    before = deepcopy(backend.docs[f"s2_ds_{case.ds}"])
    ds2 = test_db.execute(
        "INSERT INTO datasources(name,type) VALUES ('second','local') RETURNING id"
    ).fetchone()["id"]
    test_db.commit()
    case.context.datasource_id = ds2
    from kiara.ingestion._file_utils import ChunkDatasources

    ids = list(before)
    ChunkDatasources().add_associations(ids, ds2, "test-model")
    runtime._update_external_ds_ids(case.context, ids)
    case.store.finalize_document_versions(
        ids, publish_search=lambda t, m, i: runtime._publish_search_snapshot(case.context, t, m, i)
    )
    assert set(backend.docs[f"s2_ds_{ds2}"]) == set(before)
    for cid, doc in backend.docs[f"s2_ds_{ds2}"].items():
        assert doc["text"] == before[cid]["text"]
        assert doc["metadata"]["processing_pending"] is False
        assert doc["metadata"]["processing_version"]


def test_failed_canonical_write_has_no_hash_and_retry_recovers(canonical, backend):
    case = canonical
    original = backend.bulk

    def partial_final(client, actions, **kw):
        actions = list(actions)
        if any(
            a.get("_source", {}).get("metadata", {}).get("processing_pending") is False
            for a in actions
        ):
            backend.fail_after = 1
        return original(client, actions, **kw)

    from unittest.mock import patch

    with patch("elasticsearch.helpers.bulk", partial_final):
        with pytest.raises(es.ElasticsearchPartialFailureError):
            case.run()
    assert case.path not in case.hashes.load(case.ds, "test-model")
    backend.fail_after = None
    assert case.run()
    assert case.context.stats["processed_files"] == 1
    assert case.path in case.hashes.load(case.ds, "test-model")


def test_other_model_cleanup_has_no_es_side_effects(backend, monkeypatch):
    from kiara.ingestion.vectordb_ops import delete_from_elasticsearch

    monkeypatch.setattr("kiara.config.get_embedding_model", lambda cfg: "bound")
    write(1)
    before = deepcopy(backend.docs)
    delete_from_elasticsearch(CONFIG, ["/file"], datasource_id=1, embedding_model="other")
    assert backend.docs == before
    assert backend.refreshes == []


@pytest.mark.parametrize("failure", ["pg", "es", "dense"])
def test_source_cleanup_persists_addresses_until_all_layers_finish(
    canonical, backend, monkeypatch, failure
):
    from kiara.graph.store import GraphStore
    from kiara.ingestion import document_publication as publication
    from kiara.ingestion._file_utils import ChunkDatasources

    case = canonical
    assert case.run()
    ids = [c["_doc_id"] for c in case.snapshot()]
    original = deepcopy(backend.docs)
    monkeypatch.setattr(ChunkDatasources, "distinct_models", lambda self: ["test-model"])

    def cleanup():
        pipeline._cleanup_stale_chunks.__wrapped__(
            case.store._config,
            case.store,
            case.hashes,
            ChunkDatasources(),
            case.ds,
            "test-model",
            [case.path],
        )

    def fail(*args, **kwargs):
        raise RuntimeError("injected cleanup boundary")

    with monkeypatch.context() as fault:
        if failure == "pg":
            fault.setattr(GraphStore, "delete_by_chunk_ids", fail)
        elif failure == "es":
            fault.setattr(es, "delete_by_ids", fail)
        else:
            fault.setattr(case.store, "delete_by_sources", fail)
        with pytest.raises(RuntimeError, match="cleanup boundary"):
            cleanup()
    saved = publication.pending_source_cleanups("kb_test")[case.path]
    import json

    addresses = [json.loads(value) for value in saved["es::test-model"]]
    assert {a[2] for a in addresses} == set(ids)
    assert {a[1] for a in addresses} == {f"s2_ds_{case.ds}"}
    assert case.path not in case.hashes.load(case.ds, "test-model")
    if failure != "dense":
        assert backend.docs == original
    cleanup()
    assert publication.pending_source_cleanups("kb_test") == {}
    assert not case.snapshot()
    assert not backend.docs[f"s2_ds_{case.ds}"]
    assert not ChunkDatasources().get_datasource_ids_for_chunks(ids, "test-model")


def test_refresh_failure_preserves_primary_exception_and_tracks_all_targets(backend):
    def fail(**kwargs):
        raise RuntimeError("refresh down")

    backend.indices.refresh = fail
    with pytest.raises(ValueError, match="primary") as caught:
        with es.ingestion_visibility(CONFIG):
            write(1)
            write(2)
            raise ValueError("primary")
    assert caught.value.ingestion_secondary_errors[0]["error"] == "refresh down"


def test_skipped_hash_retry_rejects_missing_or_empty_corpus(backend):
    write(1)
    for missing in (False, True):
        backend.docs["s2_ds_1"] = {}
        if missing:
            del backend.docs["s2_ds_1"]
        with pytest.raises(es.ElasticsearchPartialFailureError):
            with es.ingestion_visibility(CONFIG):
                es.mark_search_data_adopted(CONFIG, [1])


def test_datasource_delete_does_not_need_file_hashes(backend):
    write(1)
    write(2)
    foreign = deepcopy(backend.docs["s2_ds_2"])
    es.delete_datasource_index(CONFIG, datasource_id=1)
    es.delete_datasource_index(CONFIG, datasource_id=1)
    assert "s2_ds_1" not in backend.docs
    assert backend.docs["s2_ds_2"] == foreign


@pytest.fixture
def whole_topology(whole_pipeline, monkeypatch, request):
    from test_document_publication_memberships import _use_es_sdk

    case = whole_pipeline
    _use_es_sdk(case, case.ds, monkeypatch, request)
    return case


from test_document_publication_pipeline import whole_pipeline  # noqa: F401


@pytest.mark.parametrize("ds", [None, "configured"])
def test_full_pipeline_initial_unchanged_changed_and_deleted(whole_topology, ds):
    from pathlib import Path

    from kiara.ingestion._file_utils import ChunkDatasources

    case = whole_topology
    owner = case.ds if ds else 0
    result = case.run_whole(case.ds if ds else None)
    assert result["processed_files"] == 1 and result["errors"] == 0
    name = f"kiara_ds_{owner}"
    assert len(case.es_backend.docs[name]) == 3
    before = case.embeddings
    result = case.run_whole(case.ds if ds else None)
    assert result["skipped_files"] == 1 and result["processed_files"] == 0
    assert case.embeddings == before
    Path(case.path).write_text("D" * 100)
    result = case.run_whole(case.ds if ds else None)
    assert result["processed_files"] == 1 and result["errors"] == 0
    assert len(case.es_backend.docs[name]) == 1
    Path(case.path).unlink()
    result = case.run_whole(case.ds if ds else None)
    assert result["deleted_files"] == 1 and result["errors"] == 0
    assert case.es_backend.docs[name] == {}
    assert case.hashes.load(owner, "test-model") == {}
    assert ChunkDatasources().count_by_datasource(owner, "test-model") == 0


def test_file_delete_recovers_pg_when_dense_points_are_missing(canonical, backend, monkeypatch):
    from kiara.ingestion import vectordb_ops
    from kiara.ingestion._file_utils import ChunkDatasources

    case = canonical
    assert case.run()
    ids = [c["_doc_id"] for c in case.snapshot()]
    case.store.delete_by_ids(ids)
    monkeypatch.setattr(vectordb_ops, "VectorStore", lambda *a, **kw: case.store)
    monkeypatch.setattr(vectordb_ops, "_resolve_colbert_models_for_ds", lambda *a: [])
    result = vectordb_ops.delete_files.__wrapped__(
        case.store._config,
        case.ds,
        "test-model",
        [case.path],
        chunk_ds=ChunkDatasources(),
        hash_store=case.hashes,
    )
    assert result["deleted_chunks"] == len(ids)
    assert not ChunkDatasources().get_datasource_ids_for_chunks(ids, "test-model")
    assert not case.hashes.load(case.ds, "test-model")
    assert backend.docs[f"s2_ds_{case.ds}"] == {}


@pytest.mark.parametrize("missing", [False, True])
def test_full_pipeline_rejects_lost_index_on_unchanged_retry(whole_topology, missing):
    case = whole_topology
    assert case.run_whole(case.ds)["processed_files"] == 1
    name = f"kiara_ds_{case.ds}"
    before = case.embeddings
    case.es_backend.docs[name].clear()
    if missing:
        case.es_backend.docs.pop(name)
    with pytest.raises(es.ElasticsearchPartialFailureError):
        case.run_whole(case.ds)
    assert case.embeddings == before


def test_empty_targets_do_not_create_client(monkeypatch):
    from unittest.mock import Mock

    client = Mock(side_effect=AssertionError("unbounded ES request"))
    monkeypatch.setattr(es, "_get_client", client)
    es.refresh_index(CONFIG, targets=())
    assert es.get_ids_by_source(CONFIG, "/file", targets=()) == []
    assert es.delete_by_ids(CONFIG, []) == 0
    client.assert_not_called()


def test_disabled_es_does_not_validate_unused_index_prefix():
    from unittest.mock import Mock

    context = SimpleNamespace(
        config={"retrieval": {"hybrid_search": "none", "elasticsearch_index": "Unused PREFIX"}},
        datasource_id=0,
        stats={},
        chunk_ds=Mock(side_effect=AssertionError("unused membership read")),
    )
    assert runtime._publish_search_snapshot(context, ["text"], [{}], ["a"]) is None
    context.chunk_ds.get_datasource_ids_for_chunks.assert_not_called()
