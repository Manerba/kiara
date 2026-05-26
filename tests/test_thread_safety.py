"""Tests fuer Thread-Safety (Fix 0): Locks und parallele Zugriffe.

Verifiziert, dass alle shared mutable State-Zugriffe korrekt
synchronisiert sind: _seen_keys, _bm25_cache, CrossEncoder-Laden,
deanonymize_params().
"""

import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

# =====================================================================
# SearchDocumentsTool: Thread-Safety fuer _seen_keys
# =====================================================================


class TestSearchDocumentsToolThreadSafety:
    def _make_tool(self):
        """Erzeugt ein SearchDocumentsTool mit Mocks."""
        from kiara.tools.builtin_tools import SearchDocumentsTool

        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {
                "text": f"chunk{i}",
                "metadata": {"source": f"doc{i}.txt", "chunk_index": 0, "chunk_total": 1},
                "score": 0.9 - i * 0.1,
            }
            for i in range(5)
        ]
        reranker = MagicMock()
        reranker.enabled = False
        config = {"retrieval": {"agent_dedup": True}}
        return SearchDocumentsTool(retriever, reranker, config)

    def test_parallel_execute_no_race(self):
        """3 parallele execute()-Calls auf derselben Instanz → kein Race."""
        tool = self._make_tool()
        import asyncio

        errors = []

        def run_one(query):
            try:
                result = asyncio.run(tool.execute({"query": query}))
                assert result.success
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run_one, args=(f"query{i}",)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Race conditions: {errors}"

    def test_seen_keys_dedup_across_parallel_calls(self):
        """_seen_keys dedupliziert korrekt ueber parallele Calls."""
        tool = self._make_tool()
        import asyncio

        results = []

        def run_one(query):
            result = asyncio.run(tool.execute({"query": query}))
            results.append(result)

        threads = [threading.Thread(target=run_one, args=(f"q{i}",)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Alle 3 Aufrufe erfolgreich
        assert len(results) == 3
        assert all(r.success for r in results)

        # Erster Call liefert Ergebnisse, nachfolgende weniger (Dedup)
        total_chunks = sum(len(r.data.get("results", [])) if r.data else 0 for r in results)
        # Genau 5 unique Chunks (doc0-doc4) — nicht 15
        assert total_chunks == 5


# =====================================================================
# HybridRetriever: Thread-Safety fuer BM25-Cache
# =====================================================================


class TestHybridRetrieverThreadSafety:
    def test_concurrent_build_blocks(self):
        """Concurrent build blockiert korrekt (_bm25_building bleibt True)."""
        from kiara.retrieval.retriever import HybridRetriever

        # Reset class-level state
        HybridRetriever._bm25_cache = {}
        HybridRetriever._bm25_building = False

        build_started = threading.Event()
        build_proceed = threading.Event()

        original_get_stats = MagicMock(return_value={"total_chunks": 10})

        with patch.object(HybridRetriever, "__init__", lambda self, *a, **kw: None):
            r = HybridRetriever.__new__(HybridRetriever)
            r._bm25_enabled = True
            r._bm25_max_corpus = 1_000_000
            r._store = MagicMock()
            r._store.get_collection_name.return_value = "test_coll"
            r._store.get_stats = original_get_stats

            # Erster Build: blockiert in get_all_documents
            def slow_get_all(*a, **kw):
                build_started.set()
                build_proceed.wait(timeout=5)
                return {"ids": [], "documents": [], "metadatas": []}

            r._store.get_all_documents.side_effect = slow_get_all

            t1 = threading.Thread(target=r.build_bm25_index)
            t1.start()

            build_started.wait(timeout=5)

            # Zweiter Build: _bm25_building ist True → return sofort
            r.build_bm25_index()

            build_proceed.set()
            t1.join(timeout=5)

        # Cleanup
        HybridRetriever._bm25_building = False
        HybridRetriever._bm25_cache = {}

    def test_ensure_bm25_cache_waits_for_active_builder(self):
        """Cache-Miss wartet auf laufenden Build statt sofort ohne BM25 weiterzulaufen."""
        from kiara.retrieval.retriever import HybridRetriever

        HybridRetriever._bm25_cache = {}
        HybridRetriever._bm25_building = True

        with patch.object(HybridRetriever, "__init__", lambda self, *a, **kw: None):
            r = HybridRetriever.__new__(HybridRetriever)
            r.build_bm25_index = MagicMock()

            def complete_build():
                with HybridRetriever._bm25_condition:
                    HybridRetriever._bm25_cache["test_coll"] = (
                        MagicMock(),
                        [],
                        [],
                        [],
                        0,
                    )
                    HybridRetriever._bm25_building = False
                    HybridRetriever._bm25_condition.notify_all()

            t = threading.Timer(0.05, complete_build)
            t.start()
            try:
                assert r._ensure_bm25_cache("test_coll") is True
            finally:
                t.cancel()
                HybridRetriever._bm25_building = False
                HybridRetriever._bm25_cache = {}

        r.build_bm25_index.assert_not_called()


# =====================================================================
# CrossEncoderReranker: Double-Checked Locking
# =====================================================================


class TestCrossEncoderRerankerThreadSafety:
    def test_model_loaded_once(self):
        """2 parallele rerank()-Calls → Modell wird nur einmal geladen."""
        from kiara.retrieval.reranker import CrossEncoderReranker

        config = {"retrieval": {"reranking": True, "rerank_top_k": 5}}
        reranker = CrossEncoderReranker(config)

        load_count = {"n": 0}

        def mock_load(self_ref):
            load_count["n"] += 1
            # Simuliere Modell-Laden
            self_ref._cross_encoder = MagicMock()
            self_ref._cross_encoder.predict.return_value = [0.9, 0.8, 0.7]

        with patch.object(CrossEncoderReranker, "_load_model", mock_load):
            results = [
                {"text": f"doc{i}", "metadata": {"source": f"s{i}"}, "score": 0.5} for i in range(3)
            ]

            barrier = threading.Barrier(2, timeout=5)
            outputs = [None, None]

            def run(idx):
                barrier.wait()
                outputs[idx] = reranker.rerank("test", list(results))

            threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # Modell nur einmal geladen (DCL)
        assert load_count["n"] == 1
        assert all(o is not None for o in outputs)


# =====================================================================
# deanonymize_params: Immer neues Dict (Thread-Safety)
# =====================================================================


class TestDeanonymizeParamsThreadSafety:
    def test_anthropic_returns_new_dict_without_anonymizer(self):
        """deanonymize_params(anonymizer=None) gibt neues Dict zurueck."""
        from kiara.generation.adapters.anthropic import AnthropicToolAdapter

        with patch.dict(sys.modules, {"anthropic": MagicMock()}):
            adapter = AnthropicToolAdapter(
                {"api_key": "sk", "model": "claude", "provider": "anthropic"},
                {"cloud": {"max_tokens": 100, "temperature": 0.3}},
            )

        params = {"query": "test", "k": 5}
        result = adapter.deanonymize_params(params)
        assert result == params
        assert result is not params  # Neues Objekt

    def test_openai_returns_new_dict_without_anonymizer(self):
        """OpenAI: deanonymize_params gibt neues Dict zurueck."""
        from kiara.generation.adapters.openai import OpenAIToolAdapter

        with patch.dict(sys.modules, {"openai": MagicMock()}):
            adapter = OpenAIToolAdapter(
                {"api_key": "sk", "model": "gpt-4", "provider": "openai"},
                {"cloud": {"max_tokens": 100, "temperature": 0.3}},
            )

        params = {"query": "test"}
        result = adapter.deanonymize_params(params)
        assert result == params
        assert result is not params

    def test_ollama_returns_new_dict(self):
        """Ollama: deanonymize_params gibt immer neues Dict zurueck."""
        from kiara.generation.adapters.ollama import OllamaToolAdapter

        with patch("ollama.Client"), patch("httpx.Timeout"):
            adapter = OllamaToolAdapter(
                {"url": "http://localhost:11434", "model": "m"},
                {"chat": {"agent": {"num_ctx": 8192, "temperature": 0.3}}},
            )

        params = {"query": "test"}
        result = adapter.deanonymize_params(params)
        assert result == params
        assert result is not params

    def test_parallel_deanonymize_separate_objects(self):
        """2 parallele deanonymize_params()-Calls → separate Dict-Objekte."""
        from kiara.generation.adapters.anthropic import AnthropicToolAdapter

        with patch.dict(sys.modules, {"anthropic": MagicMock()}):
            adapter = AnthropicToolAdapter(
                {"api_key": "sk", "model": "claude", "provider": "anthropic"},
                {"cloud": {"max_tokens": 100, "temperature": 0.3}},
            )

        params = {"query": "shared_params", "k": 5}
        results = [None, None]

        def run(idx):
            results[idx] = adapter.deanonymize_params(params)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results[0] == results[1] == params
        assert results[0] is not results[1]  # Verschiedene Objekte
        assert results[0] is not params
        assert results[1] is not params


# =====================================================================
# BM25Okapi: Parallele get_scores sind threadsafe (read-only)
# =====================================================================


class TestBM25OkapiThreadSafety:
    def test_parallel_get_scores_identical(self):
        """3 parallele get_scores()-Calls → identische Ergebnisse."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            pytest.skip("rank_bm25 nicht installiert")

        corpus = [
            ["hallo", "welt", "test"],
            ["python", "code", "test"],
            ["agent", "suche", "hallo"],
        ]
        bm25 = BM25Okapi(corpus)
        query = ["hallo", "test"]

        results = [None, None, None]

        def run(idx):
            results[idx] = list(bm25.get_scores(query))

        threads = [threading.Thread(target=run, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r is not None for r in results)
        # Alle Ergebnisse muessen identisch sein
        assert results[0] == results[1] == results[2]


# =====================================================================
# HybridRetriever: Paralleler build + retrieve / leerer Corpus
# =====================================================================


class TestHybridRetrieverParallelRetrieve:
    def _make_retriever(self):
        """Erzeugt einen HybridRetriever mit gemocktem Store."""
        from kiara.retrieval.retriever import HybridRetriever

        # Reset class-level state
        HybridRetriever._bm25_cache = {}
        HybridRetriever._bm25_building = False

        with patch.object(HybridRetriever, "__init__", lambda self, *a, **kw: None):
            r = HybridRetriever.__new__(HybridRetriever)
            r._bm25_enabled = True
            r._hybrid_mode = "bm25"
            r._es_enabled = False
            r._graph_enabled = False
            r._graph_retriever = None
            r._rrf_k = 60
            r._bm25_ttl = 3600
            r._bm25_max_corpus = 1_000_000
            r._store = MagicMock()
            r._store.get_collection_name.return_value = "test_coll"
            r._store.get_stats.return_value = {"total_chunks": 10}
            docs = [f"dokument nummer {i} mit hallo welt test" for i in range(10)]
            metas = [{"source": f"doc{i}.txt"} for i in range(10)]
            r._store.get_all_documents.return_value = {
                "ids": [f"id{i}" for i in range(10)],
                "documents": docs,
                "metadatas": metas,
            }
            r._store.query.return_value = [
                {"text": docs[0], "metadata": metas[0], "score": 0.9},
                {"text": docs[1], "metadata": metas[1], "score": 0.8},
            ]
            return r

    def test_parallel_retrieve_no_cache_corruption(self):
        """3 parallele retrieve()-Calls → kein Cache-Korruption."""
        from kiara.retrieval.retriever import HybridRetriever

        r = self._make_retriever()
        # Zuerst Index bauen
        r.build_bm25_index()

        errors = []
        results_list = [None, None, None]

        def run(idx):
            try:
                results_list[idx] = r.retrieve("hallo test", k=1)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Race conditions: {errors}"
        assert all(r is not None for r in results_list)
        # Cleanup
        HybridRetriever._bm25_cache = {}

    def test_parallel_build_and_retrieve_no_crash(self):
        """Paralleler build + retrieve → kein Crash (Fallback auf Vektor)."""
        from kiara.retrieval.retriever import HybridRetriever

        r = self._make_retriever()

        build_started = threading.Event()
        build_proceed = threading.Event()

        original_get_all = r._store.get_all_documents

        def slow_get_all(*a, **kw):
            build_started.set()
            build_proceed.wait(timeout=5)
            return original_get_all(*a, **kw)

        r._store.get_all_documents.side_effect = slow_get_all

        errors = []

        def do_build():
            try:
                r.build_bm25_index()
            except Exception as exc:
                errors.append(exc)

        def do_retrieve():
            try:
                build_started.wait(timeout=5)
                # Waehrend Build laeuft: retrieve faellt auf Vektorsuche zurueck
                result = r.retrieve("hallo test", k=1)
                assert result is not None
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=do_build)
        t2 = threading.Thread(target=do_retrieve)
        t1.start()
        t2.start()
        build_proceed.set()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not errors, f"Errors: {errors}"
        # Cleanup
        HybridRetriever._bm25_cache = {}
        HybridRetriever._bm25_building = False

    def test_build_empty_corpus_thread_safe(self):
        """build mit leerem Corpus (doc_texts=[]) → Cache-Pop thread-safe."""
        from kiara.retrieval.retriever import HybridRetriever

        r = self._make_retriever()

        # Erst einen Cache-Eintrag erzeugen
        r.build_bm25_index()
        coll_name = r._store.get_collection_name()
        assert coll_name in HybridRetriever._bm25_cache

        # Jetzt leeren Corpus simulieren
        r._store.get_all_documents.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }

        # Build mit leerem Corpus entfernt den Cache-Eintrag
        r.build_bm25_index()
        assert coll_name not in HybridRetriever._bm25_cache

        # Cleanup
        HybridRetriever._bm25_building = False

    def test_bm25_search_few_docs_no_argpartition_crash(self):
        """BM25-Suche mit wenigen Docs → kein argpartition out-of-bounds.

        Regression: np.argpartition(arr, kth=len(arr)) ist ungueltig.
        Bei fetch_k >= len(scores_arr) muss argsort statt argpartition
        verwendet werden.
        """
        from kiara.retrieval.retriever import HybridRetriever

        # Reset + kleiner Corpus (3 Docs)
        HybridRetriever._bm25_cache = {}
        HybridRetriever._bm25_building = False

        with patch.object(HybridRetriever, "__init__", lambda self, *a, **kw: None):
            r = HybridRetriever.__new__(HybridRetriever)
            r._bm25_enabled = True
            r._hybrid_mode = "bm25"
            r._es_enabled = False
            r._graph_enabled = False
            r._graph_retriever = None
            r._rrf_k = 60
            r._bm25_ttl = 3600
            r._bm25_max_corpus = 1_000_000
            r._store = MagicMock()
            r._store.get_collection_name.return_value = "small_coll"
            r._store.get_stats.return_value = {"total_chunks": 3}
            r._store.get_all_documents.return_value = {
                "ids": ["id1", "id2", "id3"],
                "documents": ["hallo welt", "python code", "agent suche"],
                "metadatas": [{"source": "a.txt"}, {"source": "b.txt"}, {"source": "c.txt"}],
            }
            r._store.query.return_value = [
                {"text": "hallo welt", "metadata": {"source": "a.txt"}, "score": 0.9},
            ]

        r.build_bm25_index()

        # k=2 → retrieve → _bm25_search(k=2) → fetch_k=min(8,3)=3
        # Vor dem Fix: ValueError: kth(=3) out of bounds (3)
        result = r.retrieve("hallo", k=2)
        assert result is not None
        assert len(result) >= 1

        # Cleanup
        HybridRetriever._bm25_cache = {}

    def test_invalidate_during_build_discards_stale_snapshot(self):
        """Invalidation waehrend laufendem Build → Build verwirft sein Writeback.

        Regression: Vor dem Generation-Counter konnte ein Build, dessen
        Snapshot vor invalidate_bm25_cache geladen wurde, beim Writeback
        einen stale Cache wiederherstellen — geloeschte Chunks blieben
        ueber Keyword-Retrieval auffindbar.
        """
        from kiara.retrieval.retriever import HybridRetriever

        HybridRetriever._bm25_cache = {}
        HybridRetriever._bm25_building = False
        HybridRetriever._bm25_generations = {}

        snapshot_loaded = threading.Event()
        invalidate_done = threading.Event()

        with patch.object(HybridRetriever, "__init__", lambda self, *a, **kw: None):
            r = HybridRetriever.__new__(HybridRetriever)
            r._bm25_enabled = True
            r._bm25_max_corpus = 1_000_000
            r._store = MagicMock()
            r._store.get_collection_name.return_value = "race_coll"
            r._store.get_stats.return_value = {"total_chunks": 5}

            def slow_get_all(*a, **kw):
                # Snapshot laden, dann blockieren bis Invalidation passiert
                docs = {
                    "ids": [f"id{i}" for i in range(5)],
                    "documents": [f"hallo welt dokument {i} mit text" for i in range(5)],
                    "metadatas": [{"source": f"doc{i}.txt"} for i in range(5)],
                }
                snapshot_loaded.set()
                invalidate_done.wait(timeout=5)
                return docs

            r._store.get_all_documents.side_effect = slow_get_all

            # Build-Thread starten
            t = threading.Thread(target=r.build_bm25_index)
            t.start()
            try:
                # Warten bis Snapshot geladen ist
                assert snapshot_loaded.wait(timeout=5), "Snapshot wurde nicht geladen"
                # Invalidieren waehrend der Build noch laeuft
                HybridRetriever.invalidate_bm25_cache("race_coll")
                invalidate_done.set()
                t.join(timeout=5)
            finally:
                invalidate_done.set()
                t.join(timeout=5)

        # Build muss seinen stale Snapshot verworfen haben
        assert "race_coll" not in HybridRetriever._bm25_cache, (
            "Stale Snapshot wurde nach Invalidation publiziert — Generation-Check hat versagt"
        )

        # Cleanup
        HybridRetriever._bm25_cache = {}
        HybridRetriever._bm25_building = False
        HybridRetriever._bm25_generations = {}


# =====================================================================
# Pseudonymizer: Thread-Safety fuer parallele deanonymize()
# =====================================================================


class TestPseudonymizerThreadSafety:
    def test_parallel_deanonymize_correct(self):
        """2 parallele deanonymize()-Calls → korrekte Ergebnisse."""
        from kiara.anonymization.anonymizer import Anonymizer
        from kiara.anonymization.automaton import AnonymizationAutomaton

        # Automaton mit einer Firma aufbauen
        automaton = AnonymizationAutomaton()
        automaton.add_entry("Hetzner Online", "company", "K-001")
        automaton.build()

        config = {"anonymization": {"enabled": True}}
        pseudo = Anonymizer(automaton, config)

        # Zuerst anonymisieren (fuellt _reverse_map)
        anon_text = pseudo.anonymize("Rechnung von Hetzner Online")
        assert "Hetzner Online" not in anon_text

        # Parallele deanonymize()-Calls
        results = [None, None]
        barrier = threading.Barrier(2, timeout=5)

        def run(idx, text):
            barrier.wait()
            results[idx] = pseudo.deanonymize(text)

        threads = [threading.Thread(target=run, args=(i, anon_text)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Beide Results korrekt de-anonymisiert
        assert all(r is not None for r in results)
        assert all("Hetzner Online" in r for r in results)
        assert results[0] == results[1]
