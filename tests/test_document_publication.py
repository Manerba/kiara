"""Echte Ingestion/PG/Qdrant-Grenzen: Cache, Teilpublikation und Prozessabbruch."""

import asyncio
import json
import subprocess
import sys
import threading
from contextlib import closing
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from test_document_version_completion import document_run, read  # noqa: F401

from kiara.exceptions import RetrievalError
from kiara.ingestion import document_publication as publication
from kiara.ingestion import runtime
from kiara.ingestion._file_utils import ChunkDatasources
from kiara.retrieval.retriever import HybridRetriever
from kiara.tools.builtin_tools import SearchDocumentsTool
from kiara.tools.document_tools import DocumentReferences

_write_to_elasticsearch = runtime._write_to_elasticsearch


def _hybrid(case):
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever._store, retriever._config = case.store, case.store._config
    retriever._chunk_ds, retriever._embed_model_name = ChunkDatasources(), "test-model"
    retriever._colbert_diagnostics_local = threading.local()
    retriever._retrieval_diagnostics_local = threading.local()
    retriever._colbert_enabled, retriever._graph_enabled = False, False
    retriever._setup_hybrid_mode({"hybrid_search": "bm25"})
    return retriever


def test_bm25_cache_built_before_completion_uses_current_evidence(document_run, monkeypatch):
    case = document_run
    case.client.upsert(
        "kb_test",
        [
            PointStruct(
                id=f"{i:032x}",
                vector=[0.0, 1.0],
                payload={
                    "_text": f"Andere Quelle Nummer {i}",
                    "source": f"other-{i}.txt",
                    "parser": "standard",
                    "chunk_index": 0,
                    "chunk_total": 1,
                },
            )
            for i in range(100)
        ],
    )
    monkeypatch.setattr("kiara.retrieval.vectorstore_qdrant.embed_query", lambda *_: [0.0, 1.0])
    retriever = _hybrid(case)
    HybridRetriever.invalidate_bm25_cache("kb_test")

    def activity(event, **details):
        if event == "graph_phase_complete":
            retriever.build_bm25_index()

    try:
        assert case.run(activity_callback=activity)
        cache = retriever._get_bm25_cached()
        assert all(m["processing_pending"] for m in cache[3] if m.get("source") == case.path)
        search = SearchDocumentsTool(
            retriever,
            None,
            {"retrieval": {"agent_dedup": False}},
            document_references=DocumentReferences("kb_test"),
        )
        result = asyncio.run(search.execute({"query": "B" * 100}))
        hits = [r for r in result.data["results"] if r["metadata"]["source"] == case.path]
        assert result.success and hits
        assert all(r["metadata"]["processing_pending"] is False for r in hits)
        assert all(r["metadata"]["document_ref"] for r in hits)
        assert retriever._get_bm25_cached() is cache
        case.store.delete_by_ids([case.snapshot()[1]["_doc_id"]])
        case.prefix = "Aktueller Kontext: "
        assert case.run()
        updated = asyncio.run(
            SearchDocumentsTool(
                retriever, None, {}, document_references=DocumentReferences("kb_test")
            ).execute({"query": "B" * 100})
        )
        assert any(r["text"].startswith(case.prefix) for r in updated.data["results"])
        assert retriever._get_bm25_cached() is cache
        # A deleted cache entry must never supply a stale text/metadata copy.
        case.store.delete_by_ids([c["_doc_id"] for c in case.snapshot()])
        assert not retriever._bm25_search("B" * 100, 5)
    finally:
        HybridRetriever.invalidate_bm25_cache("kb_test")


def test_failed_second_default_batch_blocks_every_hit_and_old_read(document_run, monkeypatch):
    case = document_run
    Path(case.path).write_text("A" * 100100)
    case.prefix = "Alter Kontext: "
    assert case.run(batch_size=128)
    before = case.snapshot()
    assert len(before) == 1001
    old_reader, old_ref = case.reader([before[0]])
    assert read(old_reader, old_ref, max_chunks=1).success
    case.store.delete_by_ids([before[500]["_doc_id"]])
    case.prefix = "Neuer Kontext: "
    original = case.client.set_payload
    batches = []

    def fail_second(*args, **kwargs):
        if "processing_version" in kwargs.get("payload", {}):
            batches.append(len(kwargs["points"]))
            if len(batches) == 2:
                raise RuntimeError("publication interrupted")
        return original(*args, **kwargs)

    monkeypatch.setattr(case.client, "set_payload", fail_second)
    with pytest.raises(RuntimeError, match="publication interrupted"):
        case.run(batch_size=128)
    assert batches == [500, 500]
    persisted = case.snapshot()
    assert len({c["metadata"]["processing_version"] for c in persisted}) == 3
    assert case.context.stats["processed_files"] == 0
    assert case.path not in case.hashes.load(case.ds, "test-model")
    for hit in (persisted[0], persisted[500], persisted[-1]):
        with pytest.raises(RetrievalError, match="nicht abgeschlossen"):
            case.reader([hit])
    with pytest.raises(RetrievalError, match="nicht abgeschlossen"):
        read(old_reader, old_ref, max_chunks=1)
    monkeypatch.setattr(case.client, "set_payload", original)
    case.enrich.reset_mock()
    assert case.run(batch_size=128)
    case.enrich.assert_not_called()
    fresh, ref = case.reader()
    assert read(fresh, ref, start_chunk=499, max_chunks=3).success
    assert [c["text"] for c in case.snapshot()] == [c["text"] for c in persisted]


def test_real_elasticsearch_partial_bulk_cannot_release_document(document_run, monkeypatch):
    from elastic_transport import ApiResponseMeta, NodeConfig
    from elastic_transport._response import ObjectApiResponse
    from elasticsearch import Elasticsearch

    from kiara.retrieval import elasticsearch_service as es_service

    case = document_run
    indexed = {}
    fail = False
    with closing(Elasticsearch("http://127.0.0.1:1")) as es:
        monkeypatch.setattr(es, "options", lambda **kwargs: es)

        def transport(*args, **kwargs):
            body = kwargs["body"]
            if isinstance(body, (bytes, str)):
                body = body.splitlines()
            items = [json.loads(v) if isinstance(v, (bytes, str)) else v for v in body]
            results = []
            for i in range(0, len(items), 2):
                action, document = items[i : i + 2]
                cid = action["index"]["_id"]
                rejected = (
                    fail and document["metadata"].get("processing_pending") is False and i > 0
                )
                if not rejected:
                    indexed[cid] = deepcopy(document)
                result = {"_id": cid, "_index": "test", "status": 429 if rejected else 201}
                if rejected:
                    result["error"] = {"type": "es_rejected_execution_exception", "reason": "test"}
                results.append({"index": result})
            return ObjectApiResponse(
                {"errors": any(r["index"]["status"] >= 300 for r in results), "items": results},
                ApiResponseMeta(200, "1.1", {}, 0.0, NodeConfig("http", "127.0.0.1", 1)),
            )

        monkeypatch.setattr(es, "perform_request", transport)
        monkeypatch.setattr(es_service, "_get_client", lambda _: es)
        monkeypatch.setattr(es_service, "create_index", lambda _, **kw: True)
        monkeypatch.setattr("kiara.config.get_embedding_model", lambda cfg: "test-model")
        monkeypatch.setattr(es.indices, "get_settings", lambda **kw: {i: {} for i in kw["index"]})
        monkeypatch.setattr(es_service, "is_configured", lambda _: True)
        monkeypatch.setattr(es_service, "update_ds_ids", lambda *a, **kw: None)
        monkeypatch.setattr(
            es_service,
            "get_ds_ids",
            lambda _, ids: {
                a: indexed[a.chunk_id]["metadata"].get("ds_ids", [])
                for a in ids
                if a.chunk_id in indexed
            },
        )
        monkeypatch.setattr(runtime, "_write_to_elasticsearch", _write_to_elasticsearch)
        assert case.run()
        first = case.snapshot()
        case.store.delete_by_ids([first[1]["_doc_id"]])
        case.prefix = "Neuer Kontext: "
        fail = True
        with pytest.raises(es_service.ElasticsearchPartialFailureError):
            case.run()
        hit = indexed[first[0]["_doc_id"]]
        assert hit["metadata"]["processing_pending"] is False
        with pytest.raises(RetrievalError, match="nicht abgeschlossen"):
            case.reader([hit])
        assert case.path not in case.hashes.load(case.ds, "test-model")
        fail = False
        case.enrich.reset_mock()
        assert case.run()
        case.enrich.assert_not_called()
        reader, ref = case.reader()
        assert read(reader, ref).success


@pytest.mark.parametrize("failure", ["transaction", "generation"])
def test_completion_rejection_keeps_hash_and_publication_open(
    document_run, monkeypatch, test_db, failure
):
    case = document_run
    assert case.run()
    complete = publication.complete_document_publication

    def reject(conn, source, model, *, datasource_id):
        complete(conn, source, model, datasource_id=datasource_id)
        raise RuntimeError("transaction interrupted")

    # Generation changes must happen before hash upsert acquires row locks.
    if failure == "generation":
        original = publication.stage_document_versions

        def stage(versions):
            publication.invalidate_documents("kb_test", [case.path], "test-model")
            original(versions)

        monkeypatch.setattr(publication, "stage_document_versions", stage)
    else:
        monkeypatch.setattr(publication, "complete_document_publication", reject)
    with pytest.raises(RuntimeError, match="entwertet|transaction interrupted"):
        case.run()
    assert case.path not in case.hashes.load(case.ds, "test-model")
    row = test_db.execute(
        "SELECT completed_at FROM document_publications WHERE source=%s", (case.path,)
    ).fetchone()
    assert row["completed_at"] is None
    with pytest.raises(RetrievalError, match="nicht abgeschlossen"):
        case.reader([case.snapshot()[0]])


@pytest.mark.parametrize("phase", ["after_begin", "before_commit"])
def test_process_death_leaves_retryable_gate_and_releases_source_lock(document_run, test_db, phase):
    case = document_run
    assert case.run()
    versions = {case.path: {"standard": case.snapshot()[0]["metadata"]["processing_version"]}}
    # A real process exits without Python cleanup, optionally inside _complete_file.
    program = """
import json, os, sys
import kiara.shared.db as db
db._dsn = "postgresql://kiara:kiara@localhost:5432/kiara_test"
from kiara.ingestion.document_publication import document_publication, stage_document_versions
from kiara.ingestion._file_utils import HashStore
from kiara.ingestion.pipeline import _complete_file
from kiara.ingestion import vectordb_ops
source, ds, versions, phase = json.loads(sys.argv[1])
with db.db_query() as conn:
    assert conn.execute("SELECT current_database() AS name").fetchone()["name"] == "kiara_test"
with document_publication("kb_test", source, "test-model", datasource_id=ds) as operation:
    operation.begin()
    if phase == "after_begin":
        os._exit(73)
    stage_document_versions(versions)
    original = vectordb_ops._recount_datasource_embeddings
    def crash(conn, ds, model):
        original(conn, ds, model)
        os._exit(73)
    vectordb_ops._recount_datasource_embeddings = crash
    _complete_file(HashStore(), ds, "test-model", source, ("source-hash", 300))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, json.dumps([case.path, case.ds, versions, phase])],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        timeout=20,
    )
    assert completed.returncode == 73, completed.stderr.decode()
    assert case.path not in case.hashes.load(case.ds, "test-model")
    with pytest.raises(RetrievalError, match="nicht abgeschlossen"):
        case.reader([case.snapshot()[0]])
    counts = test_db.execute(
        "SELECT file_count FROM datasource_embeddings WHERE datasource_id=%s AND embedding_model=%s",
        (case.ds, "test-model"),
    ).fetchone()
    assert counts["file_count"] == 0
    case.enrich.reset_mock()
    assert case.run()
    case.enrich.assert_not_called()
    reader, ref = case.reader()
    assert read(reader, ref).success


def test_es_publication_holds_collection_write_fence_until_completion(test_db):
    from test_index_compatibility_s4 import _config, _persist_contract, _resolved, _target

    from kiara.retrieval.index_compatibility import GuardedQdrantClient
    from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

    target = _target()
    _persist_contract(test_db, target, _resolved(target))
    test_db.commit()
    with closing(QdrantClient(location=":memory:")) as raw:
        raw.create_collection(
            target.collection_name, vectors_config=VectorParams(size=2, distance=Distance.COSINE)
        )
        for cid, source in [("a" * 32, "writing.txt"), ("b" * 32, "unrelated.txt")]:
            raw.upsert(
                target.collection_name,
                [
                    PointStruct(
                        id=cid,
                        vector=[1.0, 0.0],
                        payload={
                            "_text": "Evidence",
                            "source": source,
                            "source_key": source,
                            "parser": "standard",
                            "chunk_index": 0,
                            "chunk_total": 1,
                            "processing_pending": True,
                        },
                    )
                ],
            )
        store = QdrantVectorStore.__new__(QdrantVectorStore)
        store._client = GuardedQdrantClient(raw, _config(), target)
        store._config, store._index_target = _config(), target
        store._collection, store._use_source_key = target.collection_name, True
        store._require_configured_default = False
        done = threading.Event()
        errors = []

        def read_other():
            try:
                assert store.get_documents_by_ids(["b" * 32])["ids"] == ["b" * 32]
            except BaseException as exc:
                errors.append(exc)
            finally:
                done.set()

        reader = threading.Thread(target=read_other)

        def publish(*args):
            reader.start()
            assert not done.wait(0.2), "reader entered during ES publication"
            raise RuntimeError("ES interrupted")

        try:
            with patch(
                "kiara.retrieval.index_compatibility.resolve_current_index_contract",
                side_effect=AssertionError("live backend"),
            ):
                with pytest.raises(RuntimeError, match="ES interrupted"):
                    store.finalize_document_versions(["a" * 32], publish_search=publish)
                assert done.wait(5), "reader blocked after publication failed"
                store.finalize_document_versions(["a" * 32])
        finally:
            if reader.ident is not None:
                reader.join(timeout=5)
        assert not errors

        writer_started, writer_done = threading.Event(), threading.Event()

        def write_other():
            writer_started.set()
            try:
                store.delete_by_ids(["b" * 32])
            except BaseException as exc:
                errors.append(exc)
            finally:
                writer_done.set()

        writer = threading.Thread(target=write_other)

        def commit(versions):
            assert versions["writing.txt"]["standard"]
            writer.start()
            assert writer_started.wait(3)
            assert not writer_done.wait(0.2), "index writer entered before the completion callback"

        try:
            with patch(
                "kiara.retrieval.index_compatibility.resolve_current_index_contract",
                side_effect=AssertionError("live backend"),
            ):
                store.finalize_document_versions(["a" * 32], on_complete=commit)
                assert writer_done.wait(5)
        finally:
            if writer.ident is not None:
                writer.join(timeout=5)
        assert not errors


def test_publication_db_failure_does_not_register_references(document_run, monkeypatch):
    case = document_run
    assert case.run()
    hit = case.snapshot()[0]
    references = DocumentReferences("kb_test")
    with monkeypatch.context() as context:
        context.setattr(
            publication, "db_query", lambda: (_ for _ in ()).throw(RuntimeError("PG unavailable"))
        )
        with pytest.raises(RuntimeError, match="PG unavailable"):
            references.annotate([hit])
    assert references._references == references._versions == {}
    assert references.annotate([hit])[0]["metadata"]["document_ref"]
