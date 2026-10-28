"""F3/F4: Persistenz mit echtem PostgreSQL und lokalem Qdrant."""

from pathlib import Path

import pytest
from test_document_publication_colbert import colbert_pipeline  # noqa: F401
from test_document_publication_memberships import _second_datasource
from test_document_publication_pipeline import whole_pipeline  # noqa: F401
from test_document_version_completion import document_run  # noqa: F401
from test_index_compatibility_s4 import _persist_contract, _resolved

from kiara.graph.store import GraphStore
from kiara.ingestion import document_publication as publication
from kiara.ingestion import pipeline
from kiara.ingestion._file_utils import ChunkDatasources
from kiara.retrieval import colbert
from kiara.retrieval import index_compatibility as ic


def _change_enrichment(state, monkeypatch):
    case = state.case
    ids = [c["_doc_id"] for c in case.snapshot()]
    case.store.delete_by_ids(ids[1:])
    publication.invalidate_documents(
        "kb_test", [case.path], "test-model", hash_datasource_id=case.ds
    )
    case.prefix = "Neu: "
    state.calls.clear()

    def embed(_config, _model, texts, **kwargs):
        state.calls.extend(texts)
        return [[[1.0, 0.0]] if t.startswith("Neu: ") else [[0.0, 1.0]] for t in texts]

    monkeypatch.setattr(colbert, "embed_colbert", embed)
    return ids


@pytest.mark.parametrize("shared", [False, True])
def test_dense_text_repair_refreshes_colbert_score_and_preserves_owners(
    colbert_pipeline, test_db, monkeypatch, shared
):
    state = colbert_pipeline
    case = state.case
    case.prefix = "Alt: "
    assert state.run()["processed_files"] == 1
    owners = [case.ds]
    if shared:
        owners.append(_second_datasource(case, test_db))
        assert state.run(owners[-1])["processed_files"] == 1
    ids = _change_enrichment(state, monkeypatch)
    result = state.run()
    assert result["processed_files"] == 1 and result["errors"] == 0
    assert result["colbert_chunks_written"] == 2
    assert sorted(state.calls) == ["Neu: " + "B" * 100, "Neu: " + "C" * 100]
    dense = {c["_doc_id"]: c["text"] for c in case.snapshot()}
    points = case.client.retrieve(state.store._collection, ids, with_vectors=True)
    for point in points:
        cid = point.id.replace("-", "")
        assert point.payload["text"] == dense[cid]
        assert sorted(point.payload["ds_ids"]) == owners
        if cid in ids[1:]:
            assert point.vector == [[1.0, 0.0]]
    hits = state.store.search([[1.0, 0.0]], k=10)
    assert all(h["score"] == pytest.approx(1.0) for h in hits if h["_doc_id"] in ids[1:])
    assert ChunkDatasources().get_datasource_ids_for_chunks(ids, state.key) == {
        cid: owners for cid in ids
    }
    assert case.path in case.hashes.load(case.ds, "test-model")
    case.reader()


@pytest.mark.parametrize("failure", ["partial_upsert", "verify"])
def test_colbert_repair_contract_rollback_keeps_previous_owner(
    colbert_pipeline, test_db, monkeypatch, failure
):
    state = colbert_pipeline
    case = state.case
    case.prefix = "Alt: "
    assert state.run()["processed_files"] == 1
    ds2 = _second_datasource(case, test_db)
    ids = _change_enrichment(state, monkeypatch)

    def fail():
        raise ic.IndexCompatibilityError(
            "index_contract_mismatch",
            "repair-test",
            state.store._index_target,
            "Injected repair contract failure",
            current_state="mismatch",
        )

    with monkeypatch.context() as fault:
        calls = 0
        if failure == "partial_upsert":
            fault.setattr("kiara.shared.constants.QDRANT_UPSERT_BATCH_SIZE", 1)
            upsert = case.client.upsert

            def partial_upsert(*args, **kwargs):
                nonlocal calls
                if kwargs.get("collection_name") == state.store._collection:
                    calls += 1
                    if calls == 2:
                        fail()
                return upsert(*args, **kwargs)

            fault.setattr(case.client, "upsert", partial_upsert)
        else:
            existing = state.store.existing_ids

            def verification(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    fail()
                return existing(*args, **kwargs)

            fault.setattr(state.store, "existing_ids", verification)
        result = state.run(ds2)
    assert result["errors"] > 0 and result["processed_files"] == 0
    assert case.path not in case.hashes.load(ds2, "test-model")
    assert ChunkDatasources().get_datasource_ids_for_chunks(ids, state.key) == {
        cid: [case.ds] for cid in ids
    }
    points = case.client.retrieve(state.store._collection, ids, with_vectors=True)
    assert len(points) == 3
    assert all(p.vector and p.payload["ds_ids"] == [case.ds] for p in points)
    result = state.run(ds2)
    assert result["processed_files"] == 1 and result["errors"] == 0
    dense = {c["_doc_id"]: c["text"] for c in case.snapshot()}
    points = case.client.retrieve(state.store._collection, ids, with_vectors=True)
    assert all(p.payload["text"] == dense[p.id.replace("-", "")] for p in points)
    assert all(sorted(p.payload["ds_ids"]) == [case.ds, ds2] for p in points)


def test_colbert_wrong_text_after_upsert_cannot_complete(colbert_pipeline, test_db, monkeypatch):
    state = colbert_pipeline
    case = state.case
    assert state.run()["processed_files"] == 1
    ids = _change_enrichment(state, monkeypatch)
    add = state.store.add
    with monkeypatch.context() as fault:

        def wrong_text(texts, metas, chunk_ids, vectors):
            add(["Veraltet"] * len(texts), metas, chunk_ids, vectors)

        fault.setattr(state.store, "add", wrong_text)
        result = state.run()
    assert result["errors"] > 0 and result["processed_files"] == 0
    assert case.path not in case.hashes.load(case.ds, "test-model")
    assert state.run()["processed_files"] == 1
    dense = {c["_doc_id"]: c["text"] for c in case.snapshot()}
    assert all(
        p.payload["text"] == dense[p.id.replace("-", "")]
        for p in case.client.retrieve(state.store._collection, ids)
    )


@pytest.mark.parametrize("replacement", ["", "X" * 100 + "B" * 100], ids=["empty", "shorter"])
def test_cli_replacement_cleans_all_previous_owners_and_statistics(
    colbert_pipeline, test_db, replacement
):
    state = colbert_pipeline
    case = state.case
    assert state.run()["processed_files"] == 1
    ds2 = _second_datasource(case, test_db)
    assert state.run(ds2)["processed_files"] == 1
    old_ids = [c["_doc_id"] for c in case.snapshot()]
    # Dasselbe Chunk-ID-Schema eines anderen Dense-Modells bleibt unangetastet.
    chunks = ChunkDatasources()
    chunks.add_associations(old_ids, ds2, "unrelated-model")
    case.hashes.upsert_many(ds2, "unrelated-model", {case.path: ("untouched", 300)})
    Path(case.path).write_text(replacement)
    result = case.run_whole(None)
    assert result["errors"] == 0
    assert len(case.snapshot()) == (2 if replacement else 0)
    for model in ("test-model", state.key):
        expected = (
            {c["_doc_id"]: [0] for c in case.snapshot()}
            if model == "test-model" and replacement
            else {}
        )
        assert chunks.get_datasource_ids_for_chunks(old_ids, model) == expected
        assert not test_db.execute(
            "SELECT * FROM datasource_embeddings WHERE embedding_model=%s "
            "AND datasource_id IN (%s,%s)",
            (model, case.ds, ds2),
        ).fetchall()
    if replacement:
        row = test_db.execute(
            "SELECT chunk_count,file_count FROM datasource_embeddings "
            "WHERE datasource_id=0 AND embedding_model='test-model'"
        ).fetchone()
        assert (row["chunk_count"], row["file_count"]) == (2, 1)
    assert not case.client.retrieve(state.store._collection, old_ids, with_vectors=True)
    for ds in (case.ds, ds2):
        assert case.path not in case.hashes.load(ds, "test-model")
    assert chunks.get_datasource_ids_for_chunks(old_ids, "unrelated-model") == {
        cid: [ds2] for cid in old_ids
    }
    assert case.hashes.load(ds2, "unrelated-model") == {case.path: "untouched"}


def test_cli_cleanup_includes_historical_colbert_only_owner_and_graph(
    colbert_pipeline, test_db, monkeypatch
):
    from qdrant_client.models import (
        Distance,
        MultiVectorComparator,
        MultiVectorConfig,
        VectorParams,
    )

    state = colbert_pipeline
    case = state.case
    assert state.run()["processed_files"] == 1
    old_ids = [c["_doc_id"] for c in case.snapshot()]
    ds2 = _second_datasource(case, test_db)
    chunks = ChunkDatasources()
    config = case.store._config
    target = ic.colbert_index_target(config, "historical-colbert")
    case.client.create_collection(
        target.collection_name,
        vectors_config=VectorParams(
            size=2,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
        ),
    )
    _persist_contract(test_db, target, _resolved(target))
    test_db.commit()
    historical = type(state.store).__new__(type(state.store))
    historical._config, historical._model = config, "historical-colbert"
    historical._collection, historical._index_target = target.collection_name, target
    historical._intent = "write"
    historical._client = ic.GuardedQdrantClient(case.client, config, target)
    key = colbert.colbert_ds_model_key(historical._model)
    chunks.add_associations(old_ids, ds2, key)
    historical.add(["Alt"] * 3, [{"ds_ids": [ds2]}] * 3, old_ids, [[[1.0, 0.0]]] * 3)
    test_db.execute(
        "INSERT INTO datasource_embeddings(datasource_id,embedding_model,chunk_count,file_count) "
        "VALUES(%s,%s,3,0)",
        (ds2, key),
    )
    test_db.commit()
    stores = {state.store._model: state.store, historical._model: historical}
    monkeypatch.setattr(colbert, "ColBERTStore", lambda cfg, model, **kw: stores[model])
    # Ein anderer Dense-Bestand besitzt nur Chunk 0; dessen Graph bleibt erhalten.
    chunks.add_associations(old_ids[:1], ds2, "unrelated-model")
    graph = GraphStore()
    for index, cid in enumerate(old_ids):
        node = f"repair:{index}"
        graph.add_node(node, "topic", node, source="extraction")
        graph.add_edge(node, node, "self", chunk_id=cid)
    # CLI muss historische Modelle auch bei global deaktiviertem ColBERT finden.
    config["retrieval"]["colbert"]["enabled"] = False
    Path(case.path).write_text("")
    assert case.run_whole(None)["errors"] == 0
    assert not historical.existing_ids(old_ids)
    assert not chunks.get_datasource_ids_for_chunks(old_ids, key)
    assert not test_db.execute(
        "SELECT * FROM datasource_embeddings WHERE datasource_id=%s AND embedding_model=%s",
        (ds2, key),
    ).fetchall()
    edges = test_db.execute("SELECT chunk_id FROM graph_edge_chunks ORDER BY chunk_id").fetchall()
    assert [row["chunk_id"] for row in edges] == old_ids[:1]


@pytest.mark.parametrize("failure", ["colbert", "graph", "elasticsearch", "dense"])
def test_cli_cleanup_failure_retains_retry_anchor_and_recounts(
    colbert_pipeline, test_db, monkeypatch, failure
):
    state = colbert_pipeline
    case = state.case
    assert state.run()["processed_files"] == 1
    ids = [c["_doc_id"] for c in case.snapshot()]
    chunks = ChunkDatasources()

    def cleanup():
        pipeline._cleanup_stale_chunks(
            case.store._config, case.store, case.hashes, chunks, None, "test-model", [case.path]
        )

    def fail(*args, **kwargs):
        raise RuntimeError("Injected cleanup failure")

    with monkeypatch.context() as fault:
        if failure == "colbert":
            fault.setattr(state.store, "delete_by_ids", fail)
        elif failure == "graph":
            fault.setattr(GraphStore, "delete_by_chunk_ids", fail)
        elif failure == "elasticsearch":
            fault.setattr("kiara.ingestion.vectordb_ops.delete_from_elasticsearch", fail)
        else:
            fault.setattr(case.store, "delete_by_sources", fail)
        with pytest.raises(RuntimeError, match="Injected cleanup failure"):
            cleanup()
    assert len(case.snapshot()) == 3
    assert case.path not in case.hashes.load(case.ds, "test-model")
    # Vor dem PG-Commit bleibt die Zuordnung bestehen; danach sind die Zaehler
    # bereits konsistent, obwohl Dense/ES noch auf den Wiederanlauf warten.
    for model in ("test-model", state.key):
        remaining = chunks.get_datasource_ids_for_chunks(ids, model)
        expected = 3 if failure in {"colbert", "graph"} else 0
        assert len(remaining) == expected
        row = test_db.execute(
            "SELECT chunk_count,file_count FROM datasource_embeddings "
            "WHERE datasource_id=%s AND embedding_model=%s",
            (case.ds, model),
        ).fetchone()
        assert ((row["chunk_count"], row["file_count"]) if row else (0, 0)) == (expected, 0)
    cleanup()
    assert not case.snapshot()
    assert not state.store.existing_ids(ids)
    assert not chunks.get_datasource_ids_for_chunks(ids, "test-model")
    assert not chunks.get_datasource_ids_for_chunks(ids, state.key)


def test_cli_cleanup_without_membership_helper_retains_other_source(colbert_pipeline, test_db):
    state = colbert_pipeline
    case = state.case
    retained = Path(case.path).with_name("retained.txt")
    retained.write_text("Erhaltener Inhalt")
    assert state.run()["processed_files"] == 2
    retained_ids = case.store.get_ids_by_sources([str(retained)])
    assert len(retained_ids) == 1
    retained_hashes = case.hashes.load(case.ds, "test-model")
    pipeline._cleanup_stale_chunks(
        case.store._config, case.store, case.hashes, None, None, "test-model", [case.path]
    )
    assert not case.snapshot()
    assert case.store.get_ids_by_sources([str(retained)]) == retained_ids
    assert state.store.existing_ids(retained_ids, require_vectors=True) == set(retained_ids)
    assert case.hashes.load(case.ds, "test-model") == {
        str(retained): retained_hashes[str(retained)]
    }
    for model in ("test-model", state.key):
        row = test_db.execute(
            "SELECT chunk_count,file_count FROM datasource_embeddings "
            "WHERE datasource_id=%s AND embedding_model=%s",
            (case.ds, model),
        ).fetchone()
        assert (row["chunk_count"], row["file_count"]) == (1, int(model == "test-model"))
