"""Ingestion bis Suche/Lesen: Teilreparatur, Publikation und Wiederanlauf."""

import asyncio
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from kiara.exceptions import RetrievalError, VectorDBConnectionError
from kiara.ingestion import runtime
from kiara.ingestion._file_utils import ChunkDatasources, HashStore
from kiara.ingestion.pipeline import _complete_file
from kiara.retrieval.retriever import Retriever
from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore
from kiara.tools.document_tools import DocumentReferences, ReadDocumentTool


@pytest.fixture
def document_run(tmp_path, test_db, monkeypatch):
    path = tmp_path / "document.txt"
    path.write_text("A" * 100 + "B" * 100 + "C" * 100)
    ds = test_db.execute(
        "INSERT INTO datasources (name, type, path) VALUES ('Versionstest', 'local', %s) RETURNING id",
        (str(tmp_path),),
    ).fetchone()["id"]
    test_db.commit()
    client = QdrantClient(location=":memory:")
    client.create_collection(
        "kb_test", vectors_config=VectorParams(size=2, distance=Distance.COSINE)
    )
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store._client, store._collection, store._embed_model = client, "kb_test", "test-model"
    store._use_source_key = True
    config = {
        "ingestion": {"chunk_size": 100, "chunk_overlap": 0},
        "enrichment": {"enabled": True},
        "graph": {"enabled": False},
    }
    store._config = config
    store._index_contract_guard = lambda **kwargs: nullcontext()
    store.add = lambda texts, metas, ids: store.add_with_embeddings(
        texts, metas, ids, [[1.0, 0.0]] * len(ids)
    )
    hashes = HashStore()
    es = {}
    state = SimpleNamespace(
        store=store,
        client=client,
        path=str(path),
        ds=ds,
        es=es,
        pending={},
        hashes=hashes,
        prefix="",
        fail_es=False,
        embeddings=0,
    )
    monkeypatch.setattr(runtime, "_get_enrichment_prompt", lambda _: ("test", "prompt-hash"))
    monkeypatch.setattr(runtime, "_get_enrichment_llm", lambda *_: (None, None, False))
    from kiara.ingestion import enrichment

    def enrich(**kwargs):
        return [state.prefix + text for text in kwargs["chunk_texts"]], {
            "enriched": len(kwargs["chunk_texts"])
        }

    state.enrich = Mock(side_effect=enrich)
    monkeypatch.setattr(enrichment, "enrich_chunks_batch", state.enrich)

    def write_es(_config, texts, metas, ids, _stats, *, datasource_id):
        if state.fail_es and metas[0].get("processing_pending") is False:
            raise RuntimeError("ES publication failed")
        for cid, text, meta in zip(ids, texts, metas, strict=True):
            es[cid] = (text, deepcopy(meta))
            if meta.get("processing_pending"):
                state.pending[cid] = (text, deepcopy(meta))

    monkeypatch.setattr(runtime, "_write_to_elasticsearch", write_es)

    def run(**options):
        file_hash = ("source-hash", path.stat().st_size)
        ictx = runtime._IngestionContext(
            config=config,
            ctx=runtime.ProcessingContext(),
            store=store,
            chunk_ds=ChunkDatasources(),
            hash_store=hashes,
            batch_size=options.pop("batch_size", 1),
            datasource_id=ds,
            embedding_model="test-model",
            hash_mode="mtime",
            ds_overrides={"chunk_size": 100, "chunk_overlap": 0, "contextual_retrieval": True},
            file_hashes={str(path): file_hash},
            stats={"processed_files": 0, "total_chunks": 0, "errors": 0},
            pending_hashes={},
            file_completion_callback=lambda name: _complete_file(
                hashes, ds, "test-model", name, file_hash
            ),
            **options,
        )
        state.context = ictx
        return runtime._process_file_in_pipeline(ictx, str(path))

    def snapshot():
        data = store.get_documents_by_source(str(path))
        return sorted(
            [
                {"_doc_id": cid, "text": text, "metadata": meta}
                for cid, text, meta in zip(
                    data["ids"], data["documents"], data["metadatas"], strict=True
                )
            ],
            key=lambda c: c["metadata"]["chunk_index"],
        )

    def reader(chunks=None):
        chunks = chunks or snapshot()
        refs = DocumentReferences("kb_test")
        ref = refs.annotate(chunks)[0]["metadata"]["document_ref"]
        retriever = Retriever.__new__(Retriever)
        retriever._store = store
        return ReadDocumentTool(retriever, {}, refs, [ds]), ref

    state.run, state.snapshot, state.reader = run, snapshot, reader
    yield state
    client.close()


def read(tool, ref, **params):
    return asyncio.run(tool.execute({"document_id": ref, **params}))


@pytest.mark.parametrize(
    "prefix,repair_prefix",
    [("", ""), ("Alter Kontext: ", "Alter Kontext: "), ("Alter Kontext: ", "Neuer Kontext: ")],
)
def test_partial_repair_uses_complete_stored_representation(document_run, prefix, repair_prefix):
    case = document_run
    case.prefix = prefix
    assert case.run()
    before = case.snapshot()
    old_reader, old_ref = case.reader(before)
    assert read(old_reader, old_ref, max_chunks=1).success
    case.store.delete_by_ids([before[1]["_doc_id"]])
    case.prefix = repair_prefix
    case.enrich.reset_mock()
    assert case.run()
    after = case.snapshot()
    assert case.enrich.call_args.kwargs["chunk_texts"] == ["B" * 100]
    assert after[0]["text"] == before[0]["text"]
    assert after[2]["text"] == before[2]["text"]
    versions = {c["metadata"]["processing_version"] for c in after}
    assert len(versions) == 1
    assert (next(iter(versions)) == before[0]["metadata"]["processing_version"]) == (
        prefix == repair_prefix
    )
    fresh, ref = case.reader(after)
    assert read(fresh, ref).success
    if prefix != repair_prefix:
        with pytest.raises(RetrievalError, match="Dokumentversion"):
            read(old_reader, old_ref, start_chunk=1)
    for chunk in after:
        text, meta = case.es[chunk["_doc_id"]]
        assert text == chunk["text"]
        assert meta["processing_version"] == chunk["metadata"]["processing_version"]
    assert case.context.stats["processed_files"] == 1


@pytest.mark.parametrize("failure", ["qdrant", "elasticsearch"])
def test_failed_publication_recovers_with_full_dedup(document_run, monkeypatch, failure):
    case = document_run
    assert case.run()
    before = case.snapshot()
    case.store.delete_by_ids([before[1]["_doc_id"]])
    case.prefix = "Neuer Kontext: "
    original = case.client.set_payload
    monkeypatch.setattr("kiara.retrieval.vectorstore_qdrant.QDRANT_DELETE_BATCH_SIZE", 1)
    calls = 0

    def fail_update(*args, **kwargs):
        nonlocal calls
        if "processing_version" in kwargs.get("payload", {}):
            calls += 1
            if calls == 2:
                raise RuntimeError("Qdrant publication failed")
        return original(*args, **kwargs)

    if failure == "qdrant":
        monkeypatch.setattr(case.client, "set_payload", fail_update)
    else:
        case.fail_es = True
    with pytest.raises(RuntimeError, match="publication failed"):
        case.run()
    assert case.context.stats["processed_files"] == 0
    assert case.path not in case.hashes.load(case.ds, "test-model")
    with pytest.raises(RetrievalError):
        case.reader()
    persisted = case.snapshot()
    assert len(persisted) == 3
    case.fail_es = False
    monkeypatch.setattr(case.client, "set_payload", original)
    case.enrich.reset_mock()
    assert case.run()
    case.enrich.assert_not_called()
    fresh, ref = case.reader()
    assert read(fresh, ref).success
    assert [c["text"] for c in case.snapshot()] == [c["text"] for c in persisted]
    assert case.path in case.hashes.load(case.ds, "test-model")


def test_partial_embedding_failure_preserves_existing_chunks_then_retries(
    document_run, monkeypatch
):
    case = document_run
    assert case.run()
    before = case.snapshot()
    case.store.delete_by_ids([c["_doc_id"] for c in before[1:]])
    original = case.store.add
    calls = 0

    def fail_add(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise VectorDBConnectionError("embedding write failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(case.store, "add", fail_add)
    case.run()
    assert case.context.stats["processed_files"] == 0
    assert case.snapshot() == before[:1]
    monkeypatch.setattr(case.store, "add", original)
    assert case.run()
    fresh, ref = case.reader()
    assert read(fresh, ref).success


def test_completed_repair_rejects_missing_and_extra_stored_positions(document_run):
    from qdrant_client.models import PointStruct

    case = document_run
    assert case.run()
    before = case.snapshot()
    ids = [c["_doc_id"] for c in before]
    case.store.delete_by_ids([ids[-1]])
    with pytest.raises(VectorDBConnectionError, match="fehlen"):
        case.store.finalize_document_versions(ids)
    assert case.run()
    extra = {**before[0]["metadata"], "chunk_index": 3, "_text": "Veralteter Rest"}
    case.client.upsert("kb_test", [PointStruct(id="f" * 32, vector=[1.0, 0.0], payload=extra)])
    with pytest.raises(ValueError, match="zusaetzliche"):
        case.store.finalize_document_versions(ids)


def test_completed_version_is_order_independent_and_bound_to_stored_text(document_run):
    case = document_run
    case.prefix = "Gespeicherter Kontext: "
    assert case.run()
    before = case.snapshot()
    case.store.finalize_document_versions([c["_doc_id"] for c in reversed(before)])
    assert case.snapshot() == before


def test_pending_search_batch_fails_before_any_reference_is_registered():
    refs = DocumentReferences()
    with pytest.raises(RetrievalError, match="nicht abgeschlossen"):
        refs.annotate(
            [
                {
                    "text": "Sichtbarer Text",
                    "metadata": {"source": "file.txt", "processing_pending": True},
                }
            ]
        )
    assert refs._references == refs._versions == {}


@pytest.mark.parametrize(
    "metadata",
    [
        {"chunk_index": 0},
        {"chunk_total": 4},
        {"chunk_size_used": 110},
        {"modified": "2026-10-01T19:00:00"},
    ],
)
def test_completion_rejects_inconsistent_stored_representation(document_run, metadata):
    case = document_run
    assert case.run()
    chunks = case.snapshot()
    case.client.set_payload("kb_test", metadata, [chunks[1]["_doc_id"]])
    with pytest.raises(ValueError, match="Uneinheitliche"):
        case.store.finalize_document_versions([c["_doc_id"] for c in chunks])


def test_completion_keeps_parser_representations_independent(document_run):
    from qdrant_client.models import PointStruct

    case = document_run
    assert case.run()
    chunks = case.snapshot()
    original_version = chunks[0]["metadata"]["processing_version"]
    vlm_id = "e" * 32
    vlm_meta = {
        **chunks[0]["metadata"],
        "parser": "vlm",
        "chunk_total": 1,
        "processing_pending": True,
        "_text": "Eine Bildbeschreibung.",
    }
    case.client.upsert("kb_test", [PointStruct(id=vlm_id, vector=[1.0, 0.0], payload=vlm_meta)])
    case.store.finalize_document_versions([vlm_id, *[c["_doc_id"] for c in chunks]])
    stored = case.store.get_documents_by_ids([vlm_id])
    assert stored["metadatas"][0]["processing_version"] != original_version
    assert stored["metadatas"][0]["processing_pending"] is False
    stored = case.store.get_documents_by_ids([c["_doc_id"] for c in chunks])
    assert {m["processing_version"] for m in stored["metadatas"]} == {original_version}


@pytest.mark.parametrize("failure", [None, "missing", "membership", "pending"])
def test_colbert_evidence_uses_completed_primary_document(document_run, monkeypatch, failure):
    from qdrant_client.models import MultiVectorComparator, MultiVectorConfig

    from kiara.retrieval.colbert import ColBERTStore
    from kiara.retrieval.retriever import HybridRetriever

    case = document_run
    case.prefix = "Angereichert: "
    assert case.run()
    chunks = case.snapshot()
    ids = [c["_doc_id"] for c in chunks]
    case.client.create_collection(
        "colbert_test",
        vectors_config=VectorParams(
            size=2,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
        ),
    )
    cstore = ColBERTStore.__new__(ColBERTStore)
    cstore._client, cstore._collection = case.client, "colbert_test"
    cstore._index_contract_guard = lambda **kwargs: nullcontext()
    # Tatsaechliche Metadaten vor dem Abschluss, wie sie ColBERT erhaelt.
    cstore.add(
        [case.pending[cid][0] for cid in ids],
        [case.pending[cid][1] for cid in ids],
        ids,
        [[[1.0, 0.0]]] * len(ids),
    )
    cstore.update_ds_ids({cid: [case.ds] for cid in ids}, wait=True)
    assert all(c["metadata"]["processing_pending"] for c in cstore.search([[1.0, 0.0]]))
    if failure == "missing":
        case.store.delete_by_ids(ids[:1])
    elif failure == "membership":
        case.client.set_payload("kb_test", {"ds_ids": [case.ds + 1]}, ids[:1])
    elif failure == "pending":
        case.client.set_payload("kb_test", {"processing_pending": True}, ids[:1])
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever._store, retriever._colbert_store = case.store, cstore
    retriever._colbert_enabled, retriever._colbert_model = True, "test-colbert"
    retriever._config = case.store._config
    monkeypatch.setattr("kiara.retrieval.colbert.embed_colbert_query", lambda *_: [[1.0, 0.0]])
    result = retriever._colbert_retrieve_result("Text", 3, {"ds_ids": {"$in": [case.ds]}})
    if failure in {"missing", "membership"}:
        assert result.failed and not result.results
        return
    assert not result.failed
    if failure == "pending":
        with pytest.raises(RetrievalError, match="nicht abgeschlossen"):
            case.reader(result.results)
        return
    fresh, ref = case.reader(result.results)
    assert read(fresh, ref).success
    assert {c["metadata"]["processing_version"] for c in result.results} == {
        chunks[0]["metadata"]["processing_version"]
    }
    assert {c["text"] for c in result.results} == {c["text"] for c in chunks}
