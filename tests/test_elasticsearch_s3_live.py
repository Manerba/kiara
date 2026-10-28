"""Opt-in live ES corpus proof. Only UUID-scoped test indices are mutated.

KIARA_S3_ES_URL=http://localhost:9200 python -m pytest -q -s <this file>
Optional KIARA_S3_REPORT writes machine-readable score/latency evidence.
No production PG, Qdrant, source files or embeddings are read/written.
"""

import json
import os
import statistics
import threading
import time
import uuid
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from elasticsearch import Elasticsearch, helpers

from kiara.retrieval import elasticsearch_service as svc
from kiara.retrieval.elasticsearch_indices import resolve_datasource_targets
from kiara.retrieval.retriever import HybridRetriever, _FilterParams
from kiara.web import api_search


@pytest.fixture
def live(monkeypatch):
    url = os.environ.get("KIARA_S3_ES_URL")
    if not url:
        pytest.skip("KIARA_S3_ES_URL enables isolated live ES proof")
    prefix = "kiara_p69_s3_test_" + uuid.uuid4().hex
    config = {
        "retrieval": {
            "elasticsearch_url": url,
            "elasticsearch_index": prefix,
            "hybrid_search": "elasticsearch",
            "keyword_fetch_multiplier": 4,
        }
    }
    es = Elasticsearch(url)
    version = es.info()["version"]["number"]
    targets = resolve_datasource_targets(
        config,
        None,
        allowed_ds_ids=range(1, 6),
        expected_document_counts=dict.fromkeys(range(1, 6), 0),
    )
    created = []
    try:
        for target in targets:
            assert target.index_name.startswith(prefix + "_ds_")
            settings = deepcopy(svc._INDEX_SETTINGS)
            settings["settings"]["number_of_shards"] = 1
            es.indices.create(index=target.index_name, body=settings)
            created.append(target.index_name)
        monkeypatch.setattr(svc, "_get_client", lambda c: es)
        yield config, es, targets, version
    finally:
        for name in created:
            assert name.startswith(prefix + "_ds_")
            es.indices.delete(index=name, ignore_unavailable=False)
        es.close()


def _docs():
    a = [
        (
            "a1",
            "Wartung der Hydraulikpumpe: Druck pruefen und Dichtung wechseln.",
            "wartung-pumpe.pdf",
        ),
        ("a2", "Drehmoment der Schrauben am Pumpengehaeuse betraegt 35 Nm.", "montage.pdf"),
        (
            "a3",
            "Hydraulikpumpe Druckabfall: Filter reinigen, Druck und Temperatur messen.",
            "stoerung.pdf",
        ),
        ("a4", "Artikel 4711 Lagerbestand und Liefertermin beim Einkauf anfragen.", "lager.pdf"),
        ("a5", "Rechnung 2026: Zahlungsziel dreissig Tage ab Lieferung.", "rechnung.pdf"),
        ("a6", "Wartungsplan: Lager schmieren und Schrauben nachziehen.", "wartungsplan.pdf"),
    ]
    b = [
        ("b1", "Hydraulikpumpe Wartung: Dichtung nach 2000 Stunden erneuern.", "service.pdf"),
        ("b2", "Drehmoment und Schrauben fuer die Montage dokumentieren.", "drehmoment.pdf"),
        (
            "b3",
            "Artikel 4711: Liefertermin Oktober, Lagerbestand zwanzig Stueck.",
            "artikel-4711.pdf",
        ),
        ("b4", "Rechnung pruefen: Zahlungsziel und Liefertermin abstimmen.", "buchhaltung.pdf"),
        ("b5", "Druckregler einstellen und Filter der Hydraulikpumpe reinigen.", "hydraulik.pdf"),
        ("b6", "Arbeitsschutz: Gehoerschutz und Sicherheitsschuhe tragen.", "sicherheit.pdf"),
    ]

    def convert(rows, owner):
        return [
            {
                "id": cid,
                "text": text,
                "metadata": {
                    "source": f"/test/{source}",
                    "directory": "/test",
                    "file_type": ".pdf",
                    "modified": "2026-10-01",
                    "ds_ids": [owner],
                },
            }
            for cid, text, source in rows
        ]

    return convert(a, 1), convert(b, 2)


def _write(es, target, docs):
    helpers.bulk(
        es,
        [
            {
                "_index": target.index_name,
                "_id": d["id"],
                "_source": {k: v for k, v in d.items() if k != "id"},
            }
            for d in docs
        ],
        refresh=True,
    )


def _scores(response):
    return {h["_id"]: h["_score"] for h in response["hits"]["hits"]}


def test_live_corpus_isolation_dfs_reference_and_bounded_comparison(live, monkeypatch):
    config, es, targets, version = live
    a, b = _docs()
    for target, docs in [
        (targets[0], a),
        (targets[1], b),
        (targets[3], a + b),
        (targets[4], a + b),
    ]:
        _write(es, target, docs)
    _write(
        es,
        targets[2],
        [{"id": "c0", "text": "Unabhaengige Datenquelle", "metadata": {"ds_ids": [3]}}],
    )
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever._config = config
    retriever._es_fuzziness = None
    retriever._embed_model_name = "test-model"
    retriever._es_client = es
    retriever._rrf_k = 60
    retriever._retrieval_diagnostics_local = threading.local()
    retriever._chunk_ds = MagicMock()
    retriever._chunk_ds.filter_allowed.side_effect = lambda ids, ds, model: set(ids)

    # Real inventory/search and both query consumers; isolate PG/model boundary,
    # which is covered separately by contract and publication regressions.
    @contextmanager
    def scope(c, ids, model=None):
        selected = resolve_datasource_targets(
            c, ids, allowed_ds_ids=[1, 2, 3], expected_document_counts={1: 1, 2: 1, 3: 1}
        )
        yield svc.inspect_datasource_indices(c, selected).present

    monkeypatch.setattr(svc, "search_scope", scope)
    monkeypatch.setattr(api_search, "_assert_text_hits_available", lambda *args: None)
    cases = [
        ("Hydraulikpumpe Wartung", {"a1", "b1"}),
        ("Drehmoment Schrauben", {"a2", "b2"}),
        ("Artikel 4711 Lagerbestand", {"a4", "b3"}),
        ("Rechnung Zahlungsziel", {"a5", "b4"}),
        ("Druck Filter", {"a1", "a3", "b5"}),
        ("Wartungsplan Lager", {"a6"}),
    ]
    before_a = {}
    before_ab = {}
    for query, _ in cases:
        body = retriever._build_es_query_body(query, 10, _FilterParams(datasource_ids=[1]))
        before_a[query] = svc.search_explicit_targets(es, (targets[0],), body)
        body = retriever._build_es_query_body(query, 10, _FilterParams(datasource_ids=[1, 2]))
        before_ab[query] = svc.search_explicit_targets(es, targets[:2], body)
    c = [
        {
            "id": f"c{i}",
            "text": "Hydraulikpumpe Druck Wartung Filter " * (1 + i % 8),
            "metadata": {"source": "/test/fremd.pdf", "ds_ids": [3]},
        }
        for i in range(600)
    ]
    _write(es, targets[2], c)
    _write(es, targets[4], c)
    report = {
        "es_version": version,
        "prefix": config["retrieval"]["elasticsearch_index"],
        "documents": {"A": len(a), "B": len(b), "C_added": len(c)},
        "analyzer": "_INDEX_SETTINGS unchanged; one primary, zero replicas",
        "score_tolerance": {"rel": 1e-6, "abs": 1e-6},
        "cases": [],
    }
    latency = {
        "selected_ab_search_and_window": [],
        "old_shared_abc_filtered_search": [],
        "hybrid_scope_inventory_window_search": [],
        "admin_scope_inventory_window_search": [],
    }
    for query, gold in cases:
        body_a = retriever._build_es_query_body(query, 10, _FilterParams(datasource_ids=[1]))
        after_a = svc.search_explicit_targets(es, (targets[0],), body_a)
        assert _scores(after_a) == _scores(before_a[query])
        assert list(_scores(after_a)) == list(_scores(before_a[query]))
        body = retriever._build_es_query_body(query, 10, _FilterParams(datasource_ids=[1, 2]))
        actual = svc.search_explicit_targets(es, targets[:2], body)
        reference = svc.search_explicit_targets(es, (targets[3],), body)
        assert _scores(actual) == pytest.approx(_scores(reference), rel=1e-6, abs=1e-6)
        assert _scores(actual) == _scores(before_ab[query])
        assert list(_scores(actual)) == list(_scores(before_ab[query]))
        old = es.search(index=targets[4].index_name, body=body, allow_partial_search_results=False)
        # Deterministic semantic rank fixture isolates unchanged RRF mechanics;
        # no live dense recall or end-to-end answer-quality claim.
        semantic = [
            {"_doc_id": d["id"], "text": d["text"], "metadata": d["metadata"], "score": 1.0}
            for d in a + b
            if d["id"] in gold
        ]
        new_rrf = retriever._rrf_fusion_multi([semantic, retriever._parse_es_hits(actual)], 3)
        old_rrf = retriever._rrf_fusion_multi([semantic, retriever._parse_es_hits(old)], 3)
        new_ids = [r["chunk_id"] for r in new_rrf]
        old_ids = [r["chunk_id"] for r in old_rrf]
        new_keyword = list(_scores(actual))[:3]
        old_keyword = list(_scores(old))[:3]
        report["cases"].append(
            {
                "query": query,
                "gold": sorted(gold),
                "selected_scores": _scores(actual),
                "reference_scores": _scores(reference),
                "old_shared_scores": _scores(old),
                "keyword_recall_at_3": {
                    "selected": len(set(new_keyword) & gold) / len(gold),
                    "old_shared": len(set(old_keyword) & gold) / len(gold),
                },
                "rrf_at_3": {"selected": new_ids, "old_shared": old_ids},
                "rrf_recall_at_3": {
                    "selected": len(set(new_ids) & gold) / len(gold),
                    "old_shared": len(set(old_ids) & gold) / len(gold),
                },
            }
        )
        # One warmup above, five measured repetitions per query; serial local
        # cluster traffic, intentionally no expensive workload benchmark.
        for _ in range(5):
            start = time.perf_counter()
            svc.search_explicit_targets(es, targets[:2], body)
            latency["selected_ab_search_and_window"].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            es.search(index=targets[4].index_name, body=body, allow_partial_search_results=False)
            latency["old_shared_abc_filtered_search"].append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            hybrid = retriever._es_search(query, k=10, filters=_FilterParams(datasource_ids=[1, 2]))
            latency["hybrid_scope_inventory_window_search"].append(
                (time.perf_counter() - start) * 1000
            )
            start = time.perf_counter()
            admin, total = api_search._search_text(
                config, api_search.SearchRequest(query=query), [1, 2]
            )
            latency["admin_scope_inventory_window_search"].append(
                (time.perf_counter() - start) * 1000
            )
            assert {r["_doc_id"]: r["score"] for r in hybrid} == pytest.approx(_scores(actual))
            assert [r["id"] for r in admin] == list(_scores(actual))
            assert total == len(admin)
    report["latency_ms"] = {
        key: {
            "n": len(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
        for key, values in latency.items()
    }
    report["limits"] = (
        "Synthetic German enterprise queries, judged fixture gold, fixed semantic rank fixture; no live PG/Qdrant/LLM, production scale, answer quality or concurrent latency claim. Exact created test indices removed by fixture."
    )
    print(json.dumps(report, ensure_ascii=False))
    if os.environ.get("KIARA_S3_REPORT"):
        Path(os.environ["KIARA_S3_REPORT"]).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
