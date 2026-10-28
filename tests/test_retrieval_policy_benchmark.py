import json
from pathlib import Path

from kiara.benchmarks import retrieval_policy


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _golden_case() -> dict:
    return {
        "id": "KGS-T-001",
        "question": "Wie hoch ist die Rechnung 123?",
        "expected_sources": [
            {
                "label": "invoice",
                "source_contains": "/mail/invoice-123.txt",
                "source_type": "email",
                "required": True,
            }
        ],
        "expected_answer_type": "amount",
        "allowed_answer_variants": ["42,00 EUR"],
        "evaluation_rule": "contains_any",
        "required_datasources": [12],
        "expected_failure_class": "short_or_numeric_answer",
        "expected_behavior": "rag",
        "sensitivity": "test",
        "short_or_numeric_answer": True,
        "tags": ["invoice"],
    }


class FakeRetriever:
    def retrieve(self, query, **kwargs):
        return [
            {
                "text": "Rechnungsbetrag 42,00 EUR",
                "metadata": {"source": "/mail/invoice-123.txt", "file_type": ".txt"},
                "score": 0.9,
                "chunk_id": "chunk-1",
                "semantic_rank": 1,
                "semantic_score": 0.9,
            }
        ]

    def get_last_retrieval_diagnostics(self):
        return {
            "schema": "kiara-retrieval-path-diagnostics",
            "paths": {
                "semantic": {"status": "success", "latency_ms": 1, "hits": []},
                "keyword": {"status": "disabled", "latency_ms": 0, "hits": []},
                "graph": {"status": "disabled", "latency_ms": 0, "hits": []},
                "colbert": {"status": "disabled", "latency_ms": 0, "hits": []},
            },
            "rrf": {"status": "disabled", "latency_ms": 0},
            "final_results": [],
        }


class FakeReranker:
    enabled = True
    _model_name = "fake-reranker"
    _top_k = 5
    _batch_size = 1

    def rerank(self, query, results, top_k=None):
        reranked = []
        for result in results:
            item = result.copy()
            item["rerank_score"] = 1.0
            reranked.append(item)
        return reranked[:top_k]


def test_deep_merge_does_not_mutate_inputs():
    base = {"retrieval": {"rrf_k": 60, "colbert": {"enabled": True}}}
    override = {"retrieval": {"colbert": {"enabled": False}}}

    merged = retrieval_policy.deep_merge(base, override)

    assert merged["retrieval"]["rrf_k"] == 60
    assert merged["retrieval"]["colbert"]["enabled"] is False
    assert base["retrieval"]["colbert"]["enabled"] is True


def test_run_golden_variant_writes_metrics_and_reranking(tmp_path):
    golden_path = tmp_path / "golden.jsonl"
    _write_jsonl(golden_path, [_golden_case()])
    variant = retrieval_policy.RetrievalPolicyVariant(
        label="unit",
        description="unit",
        overrides={},
        retrieve_k=3,
        reranking_enabled=True,
        rerank_candidate_limit=1,
    )

    summary = retrieval_policy.run_golden_variant(
        variant=variant,
        config={"retrieval": {}},
        golden_set_path=golden_path,
        output_dir=tmp_path / "out",
        k_values=(1, 3),
        retriever_factory=lambda config, embedding_model: FakeRetriever(),
        reranker_factory=lambda config, enabled: FakeReranker() if enabled else None,
    )

    details = [
        json.loads(line)
        for line in Path(summary["artifacts"]["details"]).read_text(encoding="utf-8").splitlines()
    ]
    assert summary["overall"]["recall@1"] == 1.0
    assert summary["overall"]["gold_ctx@1"] == 1.0
    assert summary["reranker"]["effective_enabled"] is True
    assert details[0]["reranking"]["rerank_score_count"] == 1


def test_run_crag_variant_writes_full136_style_metrics(tmp_path):
    manifest_path = tmp_path / "task1.jsonl"
    doc_dir = tmp_path / "docs" / "case-1"
    source = doc_dir / "web_001.html"
    _write_jsonl(
        manifest_path,
        [
            {
                "interaction_id": "case-1",
                "query": "What is the total?",
                "answers": ["42 EUR"],
                "doc_dir": str(doc_dir),
                "file_count": 1,
            }
        ],
    )

    class CragRetriever(FakeRetriever):
        def retrieve(self, query, **kwargs):
            return [
                {
                    "text": "The total is 42 EUR.",
                    "metadata": {"source": str(source), "file_type": ".html"},
                    "score": 0.8,
                    "chunk_id": "crag-1",
                    "semantic_rank": 1,
                }
            ]

    variant = retrieval_policy.RetrievalPolicyVariant(
        label="crag-unit",
        description="unit",
        overrides={},
        retrieve_k=3,
    )

    summary = retrieval_policy.run_crag_variant(
        variant=variant,
        config={"retrieval": {}},
        manifest_path=manifest_path,
        output_dir=tmp_path / "out",
        k_values=(1, 3),
        retriever_factory=lambda config, embedding_model: CragRetriever(),
        reranker_factory=lambda config, enabled: None,
    )

    detail = json.loads(
        Path(summary["artifacts"]["details"]).read_text(encoding="utf-8").splitlines()[0]
    )
    assert summary["overall"]["gold_answer_in_context@1"] == 1.0
    assert summary["overall"]["gold_ctx@1"] == 1.0
    assert detail["any_source_under_doc_dir"] is True
    assert detail["metrics"]["recall_proxy@1"] == 1.0


def test_decision_template_keeps_reranking_disabled_when_non_rerank_is_better():
    def variant(label, *, golden, crag, latency, reranking=False):
        return {
            "variant": {"label": label, "reranking_enabled": reranking},
            "golden": {
                "overall": {"gold_ctx@20": golden, "avg_latency_ms": latency},
                "reranker": {"effective_enabled": reranking},
            },
            "crag_full136": {"overall": {"gold_ctx@20": crag, "avg_latency_ms": latency}},
        }

    summary = {
        "variants": [
            variant("live_es_rrf60_colbert_on", golden=0.7143, crag=0.2426, latency=611.4),
            variant("es_rrf60_colbert_off", golden=0.7857, crag=0.2426, latency=80.9),
            variant(
                "es_rrf60_colbert_on_rerank20",
                golden=0.75,
                crag=0.2426,
                latency=1268.7,
                reranking=True,
            ),
            variant(
                "es_rrf60_colbert_on_rerank50",
                golden=0.7143,
                crag=0.2574,
                latency=2041.0,
                reranking=True,
            ),
        ]
    }

    decision = retrieval_policy._decision_template(
        summary,
        baseline_label="live_es_rrf60_colbert_on",
        k=20,
    )

    assert decision["reranking"]["decision"] == "keep_disabled"
    assert decision["reranking"]["candidate_limit"] is None
    assert decision["reranking"]["basis"]["best_non_rerank_variant"] == "es_rrf60_colbert_off"
    assert decision["reranking"]["basis"]["best_rerank_variant"] == "es_rrf60_colbert_on_rerank20"
