"""F1/F2: PG-Zuordnungen und Publikation bis zum tatsaechlichen Modellaufruf."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_document_publication_pipeline import whole_pipeline  # noqa: F401
from test_document_version_completion import document_run  # noqa: F401

from kiara.exceptions import RetrievalError
from kiara.generation.chain import RAGChain
from kiara.ingestion import document_publication as publication
from kiara.ingestion import runtime
from kiara.ingestion._file_utils import ChunkDatasources
from kiara.retrieval.retriever import HybridRetriever, Retriever, _ColbertRetrieveResult
from kiara.tools.builtin_tools import SearchDocumentsTool


def _retriever(case, monkeypatch, cls=HybridRetriever):
    monkeypatch.setattr("kiara.retrieval.retriever.VectorStore", lambda *a, **kw: case.store)
    monkeypatch.setattr("kiara.retrieval.vectorstore_qdrant.embed_query", lambda *a: [1.0, 0.0])
    return cls(case.store._config, embedding_model="test-model")


def _chain(case, monkeypatch):
    retriever = _retriever(case, monkeypatch)
    backend = SimpleNamespace(
        provider="ollama",
        generate=Mock(return_value="Antwort"),
        generate_stream=Mock(return_value=iter(["Antwort"])),
    )
    chain = RAGChain(case.store._config, backend, embedding_model="test-model")
    chain._retriever = retriever
    return chain, backend


def _remove_memberships(case, db):
    db.execute("DELETE FROM chunk_datasources WHERE embedding_model = 'test-model'")
    db.commit()


def _invalidate(case):
    publication.invalidate_documents("kb_test", [case.path], "test-model")


@pytest.mark.parametrize("cls", [Retriever, HybridRetriever])
@pytest.mark.parametrize("filtered", [True, False])
def test_search_and_siblings_reject_removed_or_wrong_model_memberships(
    document_run, test_db, monkeypatch, cls, filtered
):
    case = document_run
    assert case.run()
    retriever = _retriever(case, monkeypatch, cls)
    allowed = [case.ds] if filtered else None
    assert len(retriever.retrieve("B", filter_datasource_ids=allowed)) == 3
    _remove_memberships(case, test_db)
    ChunkDatasources().add_associations(
        [hit["_doc_id"] for hit in case.snapshot()], case.ds, "other-model"
    )
    assert retriever.retrieve("B", filter_datasource_ids=allowed) == []
    assert retriever.load_chunks_by_source(case.path, allowed) == []


@pytest.mark.parametrize("path", ["semantic", "keyword", "graph", "colbert"])
@pytest.mark.parametrize("filtered", [True, False])
def test_every_fusion_path_checks_authority_after_fusion(
    document_run, test_db, monkeypatch, path, filtered
):
    case = document_run
    assert case.run()
    retriever = _retriever(case, monkeypatch)
    hits = case.snapshot()
    for hit in hits:
        hit["score"] = 1.0
    retriever._es_enabled = True
    retriever._graph_enabled = True
    retriever._graph_retriever = object()
    for name in ("semantic", "keyword", "graph"):
        monkeypatch.setattr(
            retriever, f"_{name}_retrieve", Mock(return_value=hits if name == path else [])
        )
    monkeypatch.setattr(
        retriever,
        "_colbert_retrieve_result",
        lambda *a: _ColbertRetrieveResult(True, False, hits if path == "colbert" else []),
    )
    allowed = [case.ds] if filtered else None
    healthy = retriever.retrieve("B", k=3, filter_datasource_ids=allowed)
    assert len(healthy) == 3
    assert {h["chunk_id"] for h in healthy} == {h["_doc_id"] for h in hits}
    assert all("_doc_id" not in h for h in healthy)  # Fusion transportiert die oeffentliche ID.

    # Ein ungueltiger Top-Treffer darf den naechsten gueltigen nicht verdraengen.
    test_db.execute(
        "DELETE FROM chunk_datasources WHERE chunk_id = %s AND embedding_model='test-model'",
        (hits[0]["_doc_id"],),
    )
    test_db.commit()
    result = retriever.retrieve("B", k=1, filter_datasource_ids=allowed)
    assert [hit["chunk_id"] for hit in result] == [hits[1]["_doc_id"]]
    _invalidate(case)
    with pytest.raises(RetrievalError, match="nicht abgeschlossen"):
        retriever.retrieve("B", filter_datasource_ids=allowed)
    _remove_memberships(case, test_db)
    assert retriever.retrieve("B", filter_datasource_ids=allowed) == []


def test_shared_owner_is_current_postgres_owner_even_without_payload_membership(
    document_run, test_db, monkeypatch
):
    case = document_run
    assert case.run()
    other = test_db.execute(
        "INSERT INTO datasources (name, type, path) VALUES ('Other', 'local', '/other') RETURNING id"
    ).fetchone()["id"]
    test_db.commit()
    ids = [h["_doc_id"] for h in case.snapshot()]
    ChunkDatasources().add_associations(ids, other, "test-model")
    test_db.execute(
        "DELETE FROM chunk_datasources WHERE datasource_id=%s AND embedding_model='test-model'",
        (case.ds,),
    )
    test_db.commit()
    retriever = _retriever(case, monkeypatch)
    assert retriever.retrieve("B", filter_datasource_ids=[case.ds]) == []
    results = retriever.load_chunks_by_source(case.path, [other])
    assert len(results) == 3
    assert all(hit["metadata"]["ds_ids"] == [other] for hit in results)
    assert len(retriever.retrieve("B")) == 3


@pytest.mark.parametrize("failure", ["membership", "publication"])
def test_proof_lookup_failure_is_closed(document_run, monkeypatch, failure):
    case = document_run
    assert case.run()
    retriever = _retriever(case, monkeypatch)
    if failure == "membership":
        monkeypatch.setattr(
            ChunkDatasources, "get_datasource_ids_for_chunks", Mock(side_effect=RuntimeError("PG"))
        )
    else:
        monkeypatch.setattr(publication, "db_query", Mock(side_effect=RuntimeError("PG")))
    with pytest.raises(RetrievalError, match="nicht geprueft"):
        retriever.retrieve("B")
    with pytest.raises(RetrievalError, match="nicht geprueft"):
        retriever.load_chunks_by_source(case.path)


@pytest.mark.parametrize(
    "broken", ["missing_id", "conflicting_id", "missing_source", "version", "pending"]
)
def test_incomplete_or_conflicting_evidence_cannot_be_model_context(
    document_run, monkeypatch, broken
):
    case = document_run
    assert case.run()
    retriever = _retriever(case, monkeypatch)
    hit = deepcopy(case.snapshot()[0])
    if broken == "missing_id":
        hit.pop("_doc_id")
    elif broken == "conflicting_id":
        hit["chunk_id"] = "another-chunk"
    elif broken == "missing_source":
        hit["metadata"].pop("source")
    elif broken == "version":
        hit["metadata"].pop("processing_version")
    else:
        hit["metadata"]["processing_pending"] = True
    with pytest.raises(RetrievalError):
        retriever.validate_context([hit])


@pytest.mark.parametrize("stream", [False, True])
def test_failed_ingestion_never_reaches_rag_backend(whole_pipeline, monkeypatch, stream):
    case = whole_pipeline
    assert case.run_whole(case.ds)["processed_files"] == 1
    case.store.delete_by_ids([case.snapshot()[1]["_doc_id"]])
    case.hashes.remove_files(case.ds, "test-model", [case.path])
    case.prefix = "Unveroeffentlichter neuer Text: "
    with monkeypatch.context() as fault:
        fault.setattr(runtime, "_mark_file_processed", Mock(side_effect=RuntimeError("Abbruch")))
        with pytest.raises(RuntimeError, match="Abbruch"):
            case.run_whole(case.ds)
    assert case.path not in case.hashes.load(case.ds, "test-model")
    assert any(h["text"].startswith(case.prefix) for h in case.snapshot())
    chain, backend = _chain(case, monkeypatch)
    with pytest.raises(RetrievalError, match="nicht abgeschlossen"):
        if stream:
            generator, _ = chain.query_stream("B", filter_datasource_ids=[case.ds])
            list(generator)
        else:
            chain.query("B", filter_datasource_ids=[case.ds])
    backend.generate.assert_not_called()
    backend.generate_stream.assert_not_called()


@pytest.mark.parametrize("change", ["pending", "membership", "completed_new_version"])
@pytest.mark.parametrize("stream", [False, True])
def test_final_rag_context_rechecked_after_reranking(
    document_run, test_db, monkeypatch, change, stream
):
    case = document_run
    assert case.run()
    chain, backend = _chain(case, monkeypatch)

    def rerank(_query, results, **kwargs):
        if change == "membership":
            _remove_memberships(case, test_db)
        elif change == "pending":
            _invalidate(case)
        else:
            case.store.delete_by_ids([case.snapshot()[1]["_doc_id"]])
            case.prefix = "Neue Version: "
            assert case.run()
        return results

    chain._reranker = SimpleNamespace(enabled=False, rerank=rerank)
    with pytest.raises(RetrievalError):
        if stream:
            generator, _ = chain.query_stream("B", filter_datasource_ids=[case.ds])
            list(generator)
        else:
            chain.query("B", filter_datasource_ids=[case.ds])
    backend.generate.assert_not_called()
    backend.generate_stream.assert_not_called()


def test_sibling_publication_error_cannot_degrade_to_selected_chunk(document_run, monkeypatch):
    case = document_run
    assert case.run()
    chain, backend = _chain(case, monkeypatch)
    load = chain._retriever.load_chunks_by_source

    def load_after_invalidation(*args, **kwargs):
        _invalidate(case)
        return load(*args, **kwargs)

    monkeypatch.setattr(chain._retriever, "load_chunks_by_source", load_after_invalidation)
    # Nur die Mitte gefunden: beide Nachbarn muessen nachgeladen werden.
    monkeypatch.setattr(chain._retriever, "retrieve", lambda **kw: [case.snapshot()[1]])
    chain._expander = SimpleNamespace(expand=lambda question: [question])
    chain._reranker = SimpleNamespace(enabled=True, rerank=lambda q, hits, **kw: hits)
    with pytest.raises(RetrievalError, match="nicht abgeschlossen"):
        chain.query("B", filter_datasource_ids=[case.ds])
    backend.generate.assert_not_called()


def test_stream_rechecks_at_consumption(document_run, monkeypatch):
    case = document_run
    assert case.run()
    chain, backend = _chain(case, monkeypatch)
    generator, sources = chain.query_stream("B", filter_datasource_ids=[case.ds])
    assert sources
    _invalidate(case)
    with pytest.raises(RetrievalError, match="nicht abgeschlossen"):
        list(generator)
    backend.generate_stream.assert_not_called()


@pytest.mark.parametrize("state", ["published", "orphan_siblings", "pending"])
def test_agent_parent_expansion_uses_same_retrieval_contract(
    document_run, test_db, monkeypatch, state
):
    case = document_run
    assert case.run()
    retriever = _retriever(case, monkeypatch)
    selected = [case.snapshot()[1]]
    # Suchauswahl ist bereits abgeschlossen; Parent-Nachladen folgt erst danach.
    monkeypatch.setattr(retriever, "retrieve", lambda **kw: selected)
    if state == "orphan_siblings":
        test_db.execute(
            "DELETE FROM chunk_datasources WHERE embedding_model='test-model' AND chunk_id != %s",
            (selected[0]["_doc_id"],),
        )
        test_db.commit()
    elif state == "pending":
        _invalidate(case)
    reranker = SimpleNamespace(enabled=True, rerank=lambda q, hits, **kw: hits)
    tool = SearchDocumentsTool(retriever, reranker, {}, allowed_datasource_ids=[case.ds])
    if state == "pending":
        with pytest.raises(RetrievalError, match="nicht abgeschlossen"):
            asyncio.run(tool.execute({"query": "Bedingung"}))
        assert not tool._seen_keys
    else:
        result = asyncio.run(tool.execute({"query": "Bedingung"}))
        assert result.success
        hits = result.data["results"]
        assert len(hits) == (3 if state == "published" else 1)
        assert all(hit["_doc_id"] for hit in hits)


def test_query_variant_cannot_hide_publication_failure(document_run, monkeypatch):
    case = document_run
    assert case.run()
    chain, backend = _chain(case, monkeypatch)
    retrieve = chain._retriever.retrieve

    def variant(**kwargs):
        if kwargs["query"] == "Variante":
            _invalidate(case)
        return retrieve(**kwargs)

    monkeypatch.setattr(chain._retriever, "retrieve", variant)
    chain._expander = SimpleNamespace(expand=lambda question: [question, "Variante"])
    chain._reranker = SimpleNamespace(enabled=True, rerank=lambda q, hits, **kw: hits)
    with pytest.raises(RetrievalError, match="nicht abgeschlossen"):
        chain.query("B", filter_datasource_ids=[case.ds])
    backend.generate.assert_not_called()


def test_grounded_retry_rechecks_original_context(document_run, monkeypatch):
    case = document_run
    assert case.run()
    chain, backend = _chain(case, monkeypatch)
    monkeypatch.setattr(
        "kiara.generation.grounding.resolve_grounding_config", lambda config: {"enabled": True}
    )
    chain._is_abstain_answer = lambda answer: True

    def generate(*args, **kwargs):
        _invalidate(case)
        return "Keine Antwort"

    backend.generate.side_effect = generate
    with pytest.raises(RetrievalError, match="nicht abgeschlossen"):
        chain.query("B", filter_datasource_ids=[case.ds])
    assert backend.generate.call_count == 1


def test_search_tab_context_is_reloaded_and_checked(document_run, test_db, monkeypatch):
    from kiara.web.api_chat import _build_context_override

    case = document_run
    assert case.run()
    chain, backend = _chain(case, monkeypatch)
    hit = case.snapshot()[0]
    override = _build_context_override(
        [{"id": hit["_doc_id"], "source": case.path, "text": "Browser-Kopie"}]
    )
    generator, _ = chain.query_stream(
        "B", context_override=override, filter_datasource_ids=[case.ds]
    )
    assert list(generator) == ["Antwort"]
    prompt = backend.generate_stream.call_args.args[1]
    assert hit["text"] in prompt and "Browser-Kopie" not in prompt
    backend.generate_stream.reset_mock()
    _remove_memberships(case, test_db)
    with pytest.raises(RetrievalError, match="nicht vollstaendig freigegeben"):
        chain.query_stream("B", context_override=override, filter_datasource_ids=[case.ds])
    backend.generate_stream.assert_not_called()
