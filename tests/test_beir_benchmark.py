import importlib.util
import json
import sys
from pathlib import Path


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


beir_prepare = _load_script("beir_prepare_dataset")
beir_eval = _load_script("beir_eval_retrieval")


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def test_prepare_beir_dataset_materializes_documents_and_qrels(tmp_path):
    source = tmp_path / "raw" / "scifact"
    qrels_dir = source / "qrels"
    qrels_dir.mkdir(parents=True)
    _write_jsonl(
        source / "corpus.jsonl",
        [
            {"_id": "doc/1", "title": "First title", "text": "First evidence text."},
            {"_id": "doc/2", "title": "", "text": "Second evidence text."},
        ],
    )
    _write_jsonl(
        source / "queries.jsonl",
        [
            {"_id": "q1", "text": "first question"},
            {"_id": "q2", "text": "missing document question"},
        ],
    )
    (qrels_dir / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\tdoc/1\t1\nq2\tdoc/404\t1\n",
        encoding="utf-8",
    )

    stats = beir_prepare.prepare_beir_dataset(
        "scifact",
        output_root=tmp_path / "prepared",
        source_dir=source,
        download=False,
    )

    output_dir = Path(stats.output_dir)
    doc_map = [json.loads(line) for line in (output_dir / "doc_map.jsonl").read_text().splitlines()]
    qrels = [json.loads(line) for line in (output_dir / "qrels.jsonl").read_text().splitlines()]
    queries = [json.loads(line) for line in (output_dir / "queries.jsonl").read_text().splitlines()]

    assert stats.documents_written == 2
    assert stats.qrels_written == 1
    assert stats.qrels_dropped_missing_document == 1
    assert doc_map[0]["doc_id"] == "doc/1"
    assert qrels == [{"doc_id": "doc/1", "query_id": "q1", "score": 1}]
    assert queries == [{"query_id": "q1", "text": "first question"}]
    first_doc = output_dir / doc_map[0]["relative_path"]
    assert "# First title" in first_doc.read_text(encoding="utf-8")


def test_evaluate_beir_retrieval_computes_doc_level_metrics(tmp_path):
    dataset_dir = tmp_path / "beir" / "scifact"
    docs_dir = dataset_dir / "documents"
    docs_dir.mkdir(parents=True)
    doc_a = docs_dir / "doc-a.md"
    doc_b = docs_dir / "doc-b.md"
    doc_c = docs_dir / "doc-c.md"
    for path in (doc_a, doc_b, doc_c):
        path.write_text(path.stem, encoding="utf-8")
    _write_jsonl(
        dataset_dir / "doc_map.jsonl",
        [
            {"doc_id": "a", "relative_path": "documents/doc-a.md"},
            {"doc_id": "b", "relative_path": "documents/doc-b.md"},
            {"doc_id": "c", "relative_path": "documents/doc-c.md"},
        ],
    )
    _write_jsonl(
        dataset_dir / "queries.jsonl",
        [
            {"query_id": "q1", "text": "alpha"},
            {"query_id": "q2", "text": "beta"},
        ],
    )
    _write_jsonl(
        dataset_dir / "qrels.jsonl",
        [
            {"query_id": "q1", "doc_id": "a", "score": 1},
            {"query_id": "q2", "doc_id": "b", "score": 1},
        ],
    )
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "scifact",
                "split": "test",
                "documents_dir": str(docs_dir),
            }
        ),
        encoding="utf-8",
    )

    class FakeRetriever:
        def retrieve(self, query, **kwargs):
            assert kwargs["filter_dir"] == str(docs_dir.resolve())
            assert kwargs["filter_datasource_ids"] == [42]
            if query == "alpha":
                return [
                    {"metadata": {"source": str(doc_c)}, "score": 0.9},
                    {"metadata": {"source": str(doc_a)}, "score": 0.8},
                    {"metadata": {"source": str(doc_a)}, "score": 0.7},
                ]
            return [{"metadata": {"source": str(doc_b)}, "score": 0.9}]

    progress = []

    stats = beir_eval.evaluate_beir_retrieval(
        dataset_dir,
        datasource_id=42,
        retriever_name="hybrid",
        k_values=(1, 2),
        run_label="test-run",
        retriever=FakeRetriever(),
        progress_callback=progress.append,
    )
    summary = json.loads(Path(stats.summary_path).read_text(encoding="utf-8"))
    details = [json.loads(line) for line in Path(stats.details_path).read_text().splitlines()]

    assert summary["overall"]["queries_evaluated"] == 2
    assert summary["settings"]["datasource_id"] == 42
    assert summary["settings"]["k_values"] == [1, 2]
    assert summary["settings"]["use_filter_dir"] is True
    assert "duration_ms" in summary["run"]
    assert summary["settings"]["ranking"]["dedupe_strategy"] == "first_chunk_rank_per_document"
    assert summary["settings"]["diagnostic_fields"]["schema"] == "kiara-retrieval-path-diagnostics"
    assert summary["settings"]["retriever_runtime"]["class"].endswith("FakeRetriever")
    assert summary["ranking_diagnostics"]["queries_without_relevant_at_max_k"] == 0
    assert summary["ranking_diagnostics"]["first_relevant_rank_median"] == 1.5
    assert summary["overall"]["recall@1"] == 0.5
    assert summary["overall"]["hitrate@1"] == 0.5
    assert summary["overall"]["mrr@2"] == 0.75
    assert summary["overall"]["ndcg@2"] > summary["overall"]["ndcg@1"]
    assert details[0]["top_docs"][0]["doc_id"] == "c"
    assert details[0]["retrieved_chunks"][0]["final_context_rank"] == 1
    assert details[0]["retrieved_chunks"][0]["source"] == str(doc_c)
    assert details[0]["top_docs"][1]["doc_id"] == "a"
    assert details[0]["ranking"]["first_relevant_rank"] == 2
    assert details[0]["candidate_stats"]["duplicate_doc_chunks"] == 1
    assert progress[-1]["current"] == 2
    assert progress[-1]["total"] == 2


def test_evaluate_beir_retrieval_reranks_chunk_candidates(tmp_path):
    dataset_dir = tmp_path / "beir" / "scifact"
    docs_dir = dataset_dir / "documents"
    docs_dir.mkdir(parents=True)
    doc_a = docs_dir / "doc-a.md"
    doc_c = docs_dir / "doc-c.md"
    for path in (doc_a, doc_c):
        path.write_text(path.stem, encoding="utf-8")
    _write_jsonl(
        dataset_dir / "doc_map.jsonl",
        [
            {"doc_id": "a", "relative_path": "documents/doc-a.md"},
            {"doc_id": "c", "relative_path": "documents/doc-c.md"},
        ],
    )
    _write_jsonl(dataset_dir / "queries.jsonl", [{"query_id": "q1", "text": "alpha"}])
    _write_jsonl(dataset_dir / "qrels.jsonl", [{"query_id": "q1", "doc_id": "a", "score": 1}])
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"dataset": "scifact", "split": "test", "documents_dir": str(docs_dir)}),
        encoding="utf-8",
    )

    class FakeRetriever:
        def retrieve(self, query, **kwargs):
            return [
                {"text": "wrong", "metadata": {"source": str(doc_c)}, "score": 0.9},
                {"text": "right", "metadata": {"source": str(doc_a)}, "score": 0.8},
            ]

    class FakeReranker:
        enabled = True
        _model_name = "fake-reranker"
        _top_k = 10
        _batch_size = 2

        def rerank(self, query, results, top_k=None):
            scored = []
            for result in results:
                item = dict(result)
                source = item["metadata"]["source"]
                item["rerank_score"] = 10.0 if source == str(doc_a) else 1.0
                scored.append(item)
            return sorted(scored, key=lambda item: item["rerank_score"], reverse=True)[:top_k]

    stats = beir_eval.evaluate_beir_retrieval(
        dataset_dir,
        datasource_id=42,
        retriever_name="dense",
        rrf_k=30,
        elasticsearch_fuzziness="none",
        k_values=(1, 2),
        run_label="rerank-test",
        retriever=FakeRetriever(),
        reranking_enabled=True,
        rerank_model="fake-reranker",
        rerank_batch_size=2,
        rerank_candidate_limit=2,
        reranker=FakeReranker(),
    )
    summary = json.loads(Path(stats.summary_path).read_text(encoding="utf-8"))
    details = [json.loads(line) for line in Path(stats.details_path).read_text().splitlines()]

    assert summary["overall"]["recall@1"] == 1.0
    assert summary["overall"]["mrr@1"] == 1.0
    assert summary["run"]["reranking_enabled"] is True
    assert summary["settings"]["reranking"]["enabled_effective"] is True
    assert summary["settings"]["reranking"]["model_effective"] == "fake-reranker"
    assert summary["settings"]["reranking"]["candidate_limit"] == 2
    assert summary["run"]["rrf_k_override"] == 30
    assert summary["run"]["elasticsearch_fuzziness_override"] == "none"
    assert summary["settings"]["retrieval_overrides"] == {
        "elasticsearch_fuzziness": "none",
        "rrf_k": 30,
    }
    assert summary["ranking_diagnostics"]["avg_rerank_score_count"] == 2.0
    assert details[0]["top_docs"][0]["doc_id"] == "a"
    assert details[0]["top_docs"][0]["rerank_score"] == 10.0
    assert details[0]["reranking"]["reranked_chunks"] == 2


def test_evaluate_beir_retrieval_document_rank_rrf_aggregates_chunks(tmp_path):
    dataset_dir = tmp_path / "beir" / "scifact"
    docs_dir = dataset_dir / "documents"
    docs_dir.mkdir(parents=True)
    doc_a = docs_dir / "doc-a.md"
    doc_c = docs_dir / "doc-c.md"
    for path in (doc_a, doc_c):
        path.write_text(path.stem, encoding="utf-8")
    _write_jsonl(
        dataset_dir / "doc_map.jsonl",
        [
            {"doc_id": "a", "relative_path": "documents/doc-a.md"},
            {"doc_id": "c", "relative_path": "documents/doc-c.md"},
        ],
    )
    _write_jsonl(dataset_dir / "queries.jsonl", [{"query_id": "q1", "text": "alpha"}])
    _write_jsonl(dataset_dir / "qrels.jsonl", [{"query_id": "q1", "doc_id": "a", "score": 1}])
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"dataset": "scifact", "split": "test", "documents_dir": str(docs_dir)}),
        encoding="utf-8",
    )

    class FakeRetriever:
        def retrieve(self, query, **kwargs):
            return [
                {"metadata": {"source": str(doc_c)}, "score": 0.9},
                {"metadata": {"source": str(doc_a)}, "score": 0.8},
                {"metadata": {"source": str(doc_a)}, "score": 0.7},
            ]

    stats = beir_eval.evaluate_beir_retrieval(
        dataset_dir,
        datasource_id=42,
        retriever_name="hybrid",
        k_values=(1, 2),
        run_label="dla-test",
        retriever=FakeRetriever(),
        doc_aggregation="rank_rrf",
        doc_aggregation_rrf_k=60,
    )
    summary = json.loads(Path(stats.summary_path).read_text(encoding="utf-8"))
    details = [json.loads(line) for line in Path(stats.details_path).read_text().splitlines()]

    assert summary["overall"]["recall@1"] == 1.0
    assert summary["settings"]["doc_aggregation"] == {
        "bonus": 0.01,
        "max_bonus_chunks": 3,
        "rrf_k": 60,
        "strategy": "rank_rrf",
    }
    assert summary["settings"]["ranking"]["dedupe_strategy"] == "document_rank_rrf"
    assert details[0]["top_docs"][0]["doc_id"] == "a"
    assert details[0]["top_docs"][0]["chunk_hits"] == 2
    assert details[0]["candidate_stats"]["duplicate_doc_chunks"] == 1


def test_evaluate_beir_retrieval_best_plus_bonus_is_conservative(tmp_path):
    dataset_dir = tmp_path / "beir" / "scifact"
    docs_dir = dataset_dir / "documents"
    docs_dir.mkdir(parents=True)
    doc_a = docs_dir / "doc-a.md"
    doc_c = docs_dir / "doc-c.md"
    for path in (doc_a, doc_c):
        path.write_text(path.stem, encoding="utf-8")
    _write_jsonl(
        dataset_dir / "doc_map.jsonl",
        [
            {"doc_id": "a", "relative_path": "documents/doc-a.md"},
            {"doc_id": "c", "relative_path": "documents/doc-c.md"},
        ],
    )
    _write_jsonl(dataset_dir / "queries.jsonl", [{"query_id": "q1", "text": "alpha"}])
    _write_jsonl(dataset_dir / "qrels.jsonl", [{"query_id": "q1", "doc_id": "a", "score": 1}])
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"dataset": "scifact", "split": "test", "documents_dir": str(docs_dir)}),
        encoding="utf-8",
    )

    class FakeRetriever:
        def retrieve(self, query, **kwargs):
            return [
                {"metadata": {"source": str(doc_c)}, "score": 0.9},
                {"metadata": {"source": str(doc_a)}, "score": 0.8},
                {"metadata": {"source": str(doc_a)}, "score": 0.7},
            ]

    stats = beir_eval.evaluate_beir_retrieval(
        dataset_dir,
        datasource_id=42,
        retriever_name="hybrid",
        k_values=(1, 2),
        run_label="best-bonus-test",
        retriever=FakeRetriever(),
        doc_aggregation="best_plus_bonus",
        doc_aggregation_bonus=0.01,
        doc_aggregation_max_bonus_chunks=3,
    )
    details = [json.loads(line) for line in Path(stats.details_path).read_text().splitlines()]
    assert details[0]["top_docs"][0]["doc_id"] == "c"
    assert details[0]["top_docs"][1]["doc_id"] == "a"

    stats = beir_eval.evaluate_beir_retrieval(
        dataset_dir,
        datasource_id=42,
        retriever_name="hybrid",
        k_values=(1, 2),
        run_label="best-bonus-strong-test",
        retriever=FakeRetriever(),
        doc_aggregation="best_plus_bonus",
        doc_aggregation_bonus=0.03,
        doc_aggregation_max_bonus_chunks=3,
    )
    summary = json.loads(Path(stats.summary_path).read_text(encoding="utf-8"))
    details = [json.loads(line) for line in Path(stats.details_path).read_text().splitlines()]

    assert summary["overall"]["recall@1"] == 1.0
    assert summary["settings"]["doc_aggregation"] == {
        "bonus": 0.03,
        "max_bonus_chunks": 3,
        "rrf_k": 60,
        "strategy": "best_plus_bonus",
    }
    assert summary["settings"]["ranking"]["dedupe_strategy"] == "best_chunk_plus_duplicate_bonus"
    assert details[0]["top_docs"][0]["doc_id"] == "a"
    assert details[0]["top_docs"][0]["bonus_chunks"] == 1
    assert details[0]["top_docs"][0]["bonus_multiplier"] == 1.03


def test_evaluate_beir_retrieval_marks_colbert_failure_invalid(tmp_path):
    dataset_dir = tmp_path / "beir" / "scifact"
    docs_dir = dataset_dir / "documents"
    docs_dir.mkdir(parents=True)
    doc_a = docs_dir / "doc-a.md"
    doc_a.write_text("doc-a", encoding="utf-8")
    _write_jsonl(
        dataset_dir / "doc_map.jsonl",
        [{"doc_id": "a", "relative_path": "documents/doc-a.md"}],
    )
    _write_jsonl(dataset_dir / "queries.jsonl", [{"query_id": "q1", "text": "alpha"}])
    _write_jsonl(dataset_dir / "qrels.jsonl", [{"query_id": "q1", "doc_id": "a", "score": 1}])
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"dataset": "scifact", "split": "test", "documents_dir": str(docs_dir)}),
        encoding="utf-8",
    )

    class FakeRetriever:
        _colbert_enabled = True
        _colbert_model = "colbert-xm"

        def __init__(self):
            self._last_colbert = None

        def retrieve(self, query, **kwargs):
            self._last_colbert = {
                "attempted": True,
                "failed": True,
                "result_count": 0,
                "model": "colbert-xm",
                "latency_ms": 12,
                "error_type": "BackendUnavailableError",
                "error": "HTTP 409 Conflict",
            }
            raise RuntimeError("ColBERT-Retrieval fehlgeschlagen: HTTP 409 Conflict")

        def get_last_colbert_diagnostics(self):
            return self._last_colbert

    stats = beir_eval.evaluate_beir_retrieval(
        dataset_dir,
        datasource_id=42,
        retriever_name="hybrid",
        k_values=(1,),
        run_label="colbert-failure",
        retriever=FakeRetriever(),
    )
    summary = json.loads(Path(stats.summary_path).read_text(encoding="utf-8"))
    details = [json.loads(line) for line in Path(stats.details_path).read_text().splitlines()]

    assert summary["run"]["status"] == "invalid"
    assert summary["run"]["valid"] is False
    assert summary["retrieval_failures"]["failed_queries"] == 1
    assert summary["colbert_diagnostics"]["failed_queries"] == 1
    assert summary["colbert_diagnostics"]["error_types"] == {"BackendUnavailableError": 1}
    assert details[0]["retrieval_error"]["type"] == "RuntimeError"
    assert details[0]["colbert"]["failed"] is True
    assert details[0]["top_docs"] == []


def test_evaluate_beir_retrieval_writes_invalid_artifact_on_retriever_init_failure(
    tmp_path, monkeypatch
):
    dataset_dir = tmp_path / "beir" / "scifact"
    docs_dir = dataset_dir / "documents"
    docs_dir.mkdir(parents=True)
    doc_a = docs_dir / "doc-a.md"
    doc_a.write_text("doc-a", encoding="utf-8")
    _write_jsonl(
        dataset_dir / "doc_map.jsonl",
        [{"doc_id": "a", "relative_path": "documents/doc-a.md"}],
    )
    _write_jsonl(dataset_dir / "queries.jsonl", [{"query_id": "q1", "text": "alpha"}])
    _write_jsonl(dataset_dir / "qrels.jsonl", [{"query_id": "q1", "doc_id": "a", "score": 1}])
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"dataset": "scifact", "split": "test", "documents_dir": str(docs_dir)}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        beir_eval.beir_module,
        "_load_config_with_retrieval_overrides",
        lambda *args, **kwargs: {
            "retrieval": {"colbert": {"enabled": True, "model": "colbert-xm"}},
            "embedding": {"model": "nomic-embed-text"},
        },
    )

    from kiara.exceptions import ColBERTInitializationError

    def fail_retriever(*args, **kwargs):
        raise ColBERTInitializationError(
            "ColBERT konnte nicht initialisiert werden: HTTP 409 Conflict"
        )

    monkeypatch.setattr(beir_eval.beir_module, "_instantiate_retriever", fail_retriever)

    stats = beir_eval.evaluate_beir_retrieval(
        dataset_dir,
        datasource_id=42,
        retriever_name="hybrid",
        k_values=(1,),
        run_label="colbert-init-failure",
    )
    summary = json.loads(Path(stats.summary_path).read_text(encoding="utf-8"))
    details = [json.loads(line) for line in Path(stats.details_path).read_text().splitlines()]

    assert stats.queries_evaluated == 0
    assert summary["run"]["status"] == "invalid"
    assert summary["run"]["valid"] is False
    assert summary["retrieval_failures"]["startup_failed"] is True
    assert summary["retrieval_failures"]["failed_queries"] == 1
    assert summary["colbert_diagnostics"]["initialization_failed"] is True
    assert summary["colbert_diagnostics"]["blocked_by_retriever_startup_failure"] is False
    assert summary["colbert_diagnostics"]["status"] == "invalid"
    assert summary["colbert_diagnostics"]["failed_queries"] == 1
    assert summary["settings"]["retriever_runtime"]["init_failed"] is True
    assert details[0]["retrieval_error"]["phase"] == "retriever_init"
    assert details[0]["colbert"]["initialization_failed"] is True


def test_evaluate_beir_retrieval_keeps_generic_init_failure_out_of_colbert_diagnostics(
    tmp_path, monkeypatch
):
    dataset_dir = tmp_path / "beir" / "scifact"
    docs_dir = dataset_dir / "documents"
    docs_dir.mkdir(parents=True)
    doc_a = docs_dir / "doc-a.md"
    doc_a.write_text("doc-a", encoding="utf-8")
    _write_jsonl(
        dataset_dir / "doc_map.jsonl",
        [{"doc_id": "a", "relative_path": "documents/doc-a.md"}],
    )
    _write_jsonl(dataset_dir / "queries.jsonl", [{"query_id": "q1", "text": "alpha"}])
    _write_jsonl(dataset_dir / "qrels.jsonl", [{"query_id": "q1", "doc_id": "a", "score": 1}])
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"dataset": "scifact", "split": "test", "documents_dir": str(docs_dir)}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        beir_eval.beir_module,
        "_load_config_with_retrieval_overrides",
        lambda *args, **kwargs: {
            "retrieval": {"colbert": {"enabled": True, "model": "colbert-xm"}},
            "embedding": {"model": "nomic-embed-text"},
        },
    )

    def fail_retriever(*args, **kwargs):
        raise RuntimeError("Dense VectorStore konnte nicht initialisiert werden: Qdrant down")

    monkeypatch.setattr(beir_eval.beir_module, "_instantiate_retriever", fail_retriever)

    stats = beir_eval.evaluate_beir_retrieval(
        dataset_dir,
        datasource_id=42,
        retriever_name="hybrid",
        k_values=(1,),
        run_label="dense-init-failure",
    )
    summary = json.loads(Path(stats.summary_path).read_text(encoding="utf-8"))
    details = [json.loads(line) for line in Path(stats.details_path).read_text().splitlines()]

    assert stats.queries_evaluated == 0
    assert summary["run"]["status"] == "invalid"
    assert summary["retrieval_failures"]["startup_failed"] is True
    assert summary["retrieval_failures"]["failed_queries"] == 1
    assert summary["colbert_diagnostics"]["enabled_requested"] is True
    assert summary["colbert_diagnostics"]["initialization_failed"] is False
    assert summary["colbert_diagnostics"]["blocked_by_retriever_startup_failure"] is True
    assert summary["colbert_diagnostics"]["status"] == "unattempted"
    assert summary["colbert_diagnostics"]["failed_queries"] == 0
    assert summary["colbert_diagnostics"]["error_types"] == {}
    assert details[0]["retrieval_error"]["phase"] == "retriever_init"
    assert details[0]["colbert"]["initialization_failed"] is False


def test_evaluate_beir_retrieval_disabled_colbert_startup_failure_not_blocked(
    tmp_path, monkeypatch
):
    dataset_dir = tmp_path / "beir" / "scifact"
    docs_dir = dataset_dir / "documents"
    docs_dir.mkdir(parents=True)
    doc_a = docs_dir / "doc-a.md"
    doc_a.write_text("doc-a", encoding="utf-8")
    _write_jsonl(
        dataset_dir / "doc_map.jsonl",
        [{"doc_id": "a", "relative_path": "documents/doc-a.md"}],
    )
    _write_jsonl(dataset_dir / "queries.jsonl", [{"query_id": "q1", "text": "alpha"}])
    _write_jsonl(dataset_dir / "qrels.jsonl", [{"query_id": "q1", "doc_id": "a", "score": 1}])
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"dataset": "scifact", "split": "test", "documents_dir": str(docs_dir)}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        beir_eval.beir_module,
        "_load_config_with_retrieval_overrides",
        lambda *args, **kwargs: {
            "retrieval": {"colbert": {"enabled": False, "model": "colbert-xm"}},
            "embedding": {"model": "nomic-embed-text"},
        },
    )

    def fail_retriever(*args, **kwargs):
        raise RuntimeError("Dense VectorStore konnte nicht initialisiert werden: Qdrant down")

    monkeypatch.setattr(beir_eval.beir_module, "_instantiate_retriever", fail_retriever)

    stats = beir_eval.evaluate_beir_retrieval(
        dataset_dir,
        datasource_id=42,
        retriever_name="hybrid",
        k_values=(1,),
        run_label="dense-init-failure-colbert-disabled",
    )
    summary = json.loads(Path(stats.summary_path).read_text(encoding="utf-8"))

    assert summary["run"]["status"] == "invalid"
    assert summary["retrieval_failures"]["startup_failed"] is True
    assert summary["colbert_diagnostics"]["enabled_requested"] is False
    assert summary["colbert_diagnostics"]["status"] == "disabled"
    assert summary["colbert_diagnostics"]["initialization_failed"] is False
    assert summary["colbert_diagnostics"]["blocked_by_retriever_startup_failure"] is False
    assert summary["colbert_diagnostics"]["failed_queries"] == 0


def test_evaluate_beir_retrieval_records_colbert_store_failure(tmp_path):
    from unittest.mock import MagicMock, patch

    from kiara.exceptions import VectorDBConnectionError
    from kiara.retrieval.retriever import HybridRetriever

    dataset_dir = tmp_path / "beir" / "scifact"
    docs_dir = dataset_dir / "documents"
    docs_dir.mkdir(parents=True)
    doc_a = docs_dir / "doc-a.md"
    doc_a.write_text("doc-a", encoding="utf-8")
    _write_jsonl(
        dataset_dir / "doc_map.jsonl",
        [{"doc_id": "a", "relative_path": "documents/doc-a.md"}],
    )
    _write_jsonl(dataset_dir / "queries.jsonl", [{"query_id": "q1", "text": "alpha"}])
    _write_jsonl(dataset_dir / "qrels.jsonl", [{"query_id": "q1", "doc_id": "a", "score": 1}])
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"dataset": "scifact", "split": "test", "documents_dir": str(docs_dir)}),
        encoding="utf-8",
    )

    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever._store = MagicMock()
    retriever._store.get_collection_name.return_value = "kb_test"
    retriever._config = {}
    retriever._bm25_enabled = False
    retriever._es_enabled = False
    retriever._hybrid_mode = "none"
    retriever._graph_enabled = False
    retriever._graph_retriever = None
    retriever._colbert_enabled = True
    retriever._colbert_model = "colbert-xm"
    retriever._colbert_store = MagicMock()
    retriever._colbert_store.search.side_effect = VectorDBConnectionError("qdrant timeout")
    retriever._chunk_ds = MagicMock()
    retriever._embed_model_name = "nomic-embed-text"
    retriever._rrf_k = 60
    retriever._bm25_ttl = 3600
    retriever._check_bm25_ttl = MagicMock()
    retriever._semantic_retrieve = MagicMock(return_value=[])
    retriever._keyword_retrieve = MagicMock(return_value=[])
    retriever._graph_retrieve = MagicMock(return_value=[])

    with patch("kiara.retrieval.colbert.embed_colbert_query", return_value=[[0.1]]):
        stats = beir_eval.evaluate_beir_retrieval(
            dataset_dir,
            datasource_id=42,
            retriever_name="hybrid",
            k_values=(1,),
            run_label="colbert-store-failure",
            retriever=retriever,
        )

    summary = json.loads(Path(stats.summary_path).read_text(encoding="utf-8"))
    details = [json.loads(line) for line in Path(stats.details_path).read_text().splitlines()]

    assert summary["run"]["status"] == "invalid"
    assert summary["colbert_diagnostics"]["failed_queries"] == 1
    assert summary["colbert_diagnostics"]["error_types"] == {"VectorDBConnectionError": 1}
    assert details[0]["retrieval_error"]["type"] == "RuntimeError"
    assert details[0]["colbert"]["failed"] is True
    assert details[0]["colbert"]["error_type"] == "VectorDBConnectionError"


def test_evaluate_beir_retrieval_counts_colbert_diagnostics_failure(tmp_path):
    dataset_dir = tmp_path / "beir" / "scifact"
    docs_dir = dataset_dir / "documents"
    docs_dir.mkdir(parents=True)
    doc_a = docs_dir / "doc-a.md"
    doc_a.write_text("doc-a", encoding="utf-8")
    _write_jsonl(
        dataset_dir / "doc_map.jsonl",
        [{"doc_id": "a", "relative_path": "documents/doc-a.md"}],
    )
    _write_jsonl(dataset_dir / "queries.jsonl", [{"query_id": "q1", "text": "alpha"}])
    _write_jsonl(dataset_dir / "qrels.jsonl", [{"query_id": "q1", "doc_id": "a", "score": 1}])
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"dataset": "scifact", "split": "test", "documents_dir": str(docs_dir)}),
        encoding="utf-8",
    )

    class FakeRetriever:
        _colbert_enabled = True
        _colbert_model = "colbert-xm"

        def retrieve(self, query, **kwargs):
            return [{"metadata": {"source": str(doc_a)}, "score": 0.9}]

        def get_last_colbert_diagnostics(self):
            raise RuntimeError("diagnostics broken")

    stats = beir_eval.evaluate_beir_retrieval(
        dataset_dir,
        datasource_id=42,
        retriever_name="hybrid",
        k_values=(1,),
        run_label="colbert-diagnostics-failure",
        retriever=FakeRetriever(),
    )
    summary = json.loads(Path(stats.summary_path).read_text(encoding="utf-8"))
    details = [json.loads(line) for line in Path(stats.details_path).read_text().splitlines()]

    assert summary["run"]["status"] == "invalid"
    assert summary["colbert_diagnostics"]["failed_queries"] == 1
    assert summary["colbert_diagnostics"]["diagnostics_failed_queries"] == 1
    assert summary["colbert_diagnostics"]["error_types"] == {"DiagnosticsError": 1}
    assert details[0]["colbert"]["diagnostics_failed"] is True


def test_evaluate_beir_retrieval_records_successful_colbert_zero_results(tmp_path):
    dataset_dir = tmp_path / "beir" / "scifact"
    docs_dir = dataset_dir / "documents"
    docs_dir.mkdir(parents=True)
    doc_a = docs_dir / "doc-a.md"
    doc_a.write_text("doc-a", encoding="utf-8")
    _write_jsonl(
        dataset_dir / "doc_map.jsonl",
        [{"doc_id": "a", "relative_path": "documents/doc-a.md"}],
    )
    _write_jsonl(dataset_dir / "queries.jsonl", [{"query_id": "q1", "text": "alpha"}])
    _write_jsonl(dataset_dir / "qrels.jsonl", [{"query_id": "q1", "doc_id": "a", "score": 1}])
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"dataset": "scifact", "split": "test", "documents_dir": str(docs_dir)}),
        encoding="utf-8",
    )

    class FakeRetriever:
        _colbert_enabled = True
        _colbert_model = "colbert-xm"

        def retrieve(self, query, **kwargs):
            return []

        def get_last_colbert_diagnostics(self):
            return {
                "attempted": True,
                "failed": False,
                "result_count": 0,
                "model": "colbert-xm",
                "latency_ms": 3,
            }

    stats = beir_eval.evaluate_beir_retrieval(
        dataset_dir,
        datasource_id=42,
        retriever_name="hybrid",
        k_values=(1,),
        run_label="colbert-zero-results",
        retriever=FakeRetriever(),
    )
    summary = json.loads(Path(stats.summary_path).read_text(encoding="utf-8"))
    details = [json.loads(line) for line in Path(stats.details_path).read_text().splitlines()]

    assert summary["run"]["status"] == "success"
    assert summary["run"]["valid"] is True
    assert summary["colbert_diagnostics"]["zero_result_queries"] == 1
    assert summary["colbert_diagnostics"]["failed_queries"] == 0
    assert details[0]["colbert"]["failed"] is False
    assert details[0]["colbert"]["result_count"] == 0


def test_evaluate_beir_retrieval_marks_single_colbert_failure_partial(tmp_path):
    dataset_dir = tmp_path / "beir" / "scifact"
    docs_dir = dataset_dir / "documents"
    docs_dir.mkdir(parents=True)
    doc_a = docs_dir / "doc-a.md"
    doc_b = docs_dir / "doc-b.md"
    doc_a.write_text("doc-a", encoding="utf-8")
    doc_b.write_text("doc-b", encoding="utf-8")
    _write_jsonl(
        dataset_dir / "doc_map.jsonl",
        [
            {"doc_id": "a", "relative_path": "documents/doc-a.md"},
            {"doc_id": "b", "relative_path": "documents/doc-b.md"},
        ],
    )
    _write_jsonl(
        dataset_dir / "queries.jsonl",
        [
            {"query_id": "q1", "text": "alpha"},
            {"query_id": "q2", "text": "beta"},
        ],
    )
    _write_jsonl(
        dataset_dir / "qrels.jsonl",
        [
            {"query_id": "q1", "doc_id": "a", "score": 1},
            {"query_id": "q2", "doc_id": "b", "score": 1},
        ],
    )
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"dataset": "scifact", "split": "test", "documents_dir": str(docs_dir)}),
        encoding="utf-8",
    )

    class FakeRetriever:
        _colbert_enabled = True
        _colbert_model = "colbert-xm"

        def __init__(self):
            self._last_colbert = None

        def retrieve(self, query, **kwargs):
            if query == "alpha":
                self._last_colbert = {
                    "attempted": True,
                    "failed": False,
                    "result_count": 1,
                    "model": "colbert-xm",
                    "latency_ms": 2,
                }
                return [{"metadata": {"source": str(doc_a)}, "score": 0.9}]
            self._last_colbert = {
                "attempted": True,
                "failed": True,
                "result_count": 0,
                "model": "colbert-xm",
                "latency_ms": 12,
                "error_type": "BackendUnavailableError",
                "error": "HTTP 409 Conflict",
            }
            raise RuntimeError("ColBERT-Retrieval fehlgeschlagen: HTTP 409 Conflict")

        def get_last_colbert_diagnostics(self):
            return self._last_colbert

    stats = beir_eval.evaluate_beir_retrieval(
        dataset_dir,
        datasource_id=42,
        retriever_name="hybrid",
        k_values=(1,),
        run_label="colbert-partial",
        retriever=FakeRetriever(),
    )
    summary = json.loads(Path(stats.summary_path).read_text(encoding="utf-8"))
    details = [json.loads(line) for line in Path(stats.details_path).read_text().splitlines()]

    assert summary["run"]["status"] == "partial"
    assert summary["run"]["valid"] is False
    assert summary["retrieval_failures"]["failed_queries"] == 1
    assert summary["colbert_diagnostics"]["status"] == "partial"
    assert summary["colbert_diagnostics"]["failed_queries"] == 1
    assert summary["colbert_diagnostics"]["successful_queries"] == 1
    assert details[0]["colbert"]["failed"] is False
    assert details[1]["colbert"]["failed"] is True


def test_evaluate_beir_retrieval_zero_queries_is_invalid(tmp_path):
    dataset_dir = tmp_path / "beir" / "scifact"
    docs_dir = dataset_dir / "documents"
    docs_dir.mkdir(parents=True)
    doc_a = docs_dir / "doc-a.md"
    doc_a.write_text("doc-a", encoding="utf-8")
    _write_jsonl(
        dataset_dir / "doc_map.jsonl",
        [{"doc_id": "a", "relative_path": "documents/doc-a.md"}],
    )
    _write_jsonl(dataset_dir / "queries.jsonl", [{"query_id": "q1", "text": "alpha"}])
    _write_jsonl(dataset_dir / "qrels.jsonl", [{"query_id": "q1", "doc_id": "a", "score": 1}])
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"dataset": "scifact", "split": "test", "documents_dir": str(docs_dir)}),
        encoding="utf-8",
    )

    class FakeRetriever:
        def retrieve(self, query, **kwargs):
            raise AssertionError("retrieve must not be called")

    stats = beir_eval.evaluate_beir_retrieval(
        dataset_dir,
        datasource_id=42,
        retriever_name="hybrid",
        k_values=(1,),
        max_queries=0,
        run_label="zero-queries",
        retriever=FakeRetriever(),
    )
    summary = json.loads(Path(stats.summary_path).read_text(encoding="utf-8"))

    assert stats.queries_evaluated == 0
    assert summary["run"]["status"] == "invalid"
    assert summary["run"]["valid"] is False
    assert summary["settings"]["queries_evaluated"] == 0


def test_parse_k_values_rejects_empty_values():
    assert beir_eval.parse_k_values("10,1,10") == (1, 10)
    try:
        beir_eval.parse_k_values("")
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_collect_beir_artifacts_reads_history(tmp_path, monkeypatch):
    from kiara.web import api_benchmarks

    dataset_dir = tmp_path / "beir" / "scifact"
    results_dir = dataset_dir / "results"
    results_dir.mkdir(parents=True)
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "scifact",
                "split": "test",
                "documents_dir": str(dataset_dir / "documents"),
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(dataset_dir / "queries.jsonl", [{"query_id": "q1", "text": "question"}])
    _write_jsonl(dataset_dir / "qrels.jsonl", [{"query_id": "q1", "doc_id": "d1", "score": 1}])
    details_path = results_dir / "demo_details.jsonl"
    _write_jsonl(
        details_path,
        [
            {
                "query_id": "q1",
                "relevant_doc_ids": ["d1"],
                "retrieved_chunks": 2,
                "unmapped_chunks": 0,
                "top_docs": [{"doc_id": "d2"}, {"doc_id": "d1"}],
            },
            {
                "query_id": "q2",
                "relevant_doc_ids": ["d3"],
                "retrieved_chunks": 2,
                "unmapped_chunks": 1,
                "top_docs": [{"doc_id": "d4"}],
            },
        ],
    )
    (results_dir / "demo_summary.json").write_text(
        json.dumps(
            {
                "schema": "kiara-beir-retrieval-summary",
                "created_at": "2026-05-31T00:00:00+00:00",
                "dataset": "scifact",
                "split": "test",
                "run": {
                    "label": "demo",
                    "retriever": "hybrid",
                    "datasource_id": 7,
                    "status": "partial",
                    "valid": False,
                },
                "settings": {"datasource_id": 7, "retriever": "hybrid"},
                "overall": {
                    "queries_evaluated": 1,
                    "recall@10": 1.0,
                    "ndcg@10": 0.5,
                    "mrr@10": 0.5,
                    "hitrate@10": 1.0,
                },
                "retrieval_failures": {"failed_queries": 1, "error_types": {"RuntimeError": 1}},
                "colbert_diagnostics": {
                    "status": "partial",
                    "failed_queries": 1,
                    "error_types": {"BackendUnavailableError": 1},
                },
                "artifacts": {"details": str(details_path)},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        api_benchmarks,
        "_collect_beir_datasources",
        lambda: [{"id": 7, "name": "Benchmark BEIR SciFact"}],
    )

    data = api_benchmarks._collect_beir_artifacts(tmp_path)

    assert data["datasets"][0]["dataset"] == "scifact"
    assert data["datasources"][0]["id"] == 7
    assert data["summaries"][0]["label"] == "demo"
    assert data["summaries"][0]["status"] == "partial"
    assert data["summaries"][0]["valid"] is False
    assert data["summaries"][0]["retrieval_failures"]["failed_queries"] == 1
    assert data["summaries"][0]["colbert_diagnostics"]["failed_queries"] == 1
    assert data["summaries"][0]["timestamp"] == "2026-05-31T00:00:00+00:00"
    assert data["summaries"][0]["key_metrics"]["recall@10"] == 1.0
    diagnostics = data["summaries"][0]["ranking_diagnostics"]
    assert diagnostics["legacy_derived"] is True
    assert diagnostics["first_relevant_rank_median"] == 2.0
    assert diagnostics["queries_without_relevant_at_max_k"] == 1
    assert diagnostics["avg_retrieved_chunks"] == 2.0
    assert diagnostics["duplicate_doc_chunks_available"] is False
