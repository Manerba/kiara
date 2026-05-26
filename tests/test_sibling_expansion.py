"""Tests fuer die Sibling-Chunk-Expansion in RAGChain.

Stellt sicher, dass nach dem Reranking Geschwister-Chunks
derselben Quelldatei ergaenzt werden, damit das LLM den
vollen Dokumentkontext sieht.
"""

from unittest.mock import MagicMock, patch

from kiara.generation.backends.base import LLMBackend


def _make_mock_backend():
    mock_backend = MagicMock(spec=LLMBackend)
    mock_backend.provider = "ollama"
    mock_backend.model = "test-model"
    mock_backend.last_usage = {}
    mock_backend.generate.return_value = "Antwort"
    mock_backend.generate_stream.return_value = iter(["Antwort"])
    return mock_backend


def _base_config(**overrides):
    cfg = {
        "datasource": {"path": "/tmp/test"},
        "ollama": {
            "url": "http://localhost:11434",
            "llm_model": "test-model",
            "embedding_model": "test-embed",
            "prompt_template": "Kontext:\n{context}\n\nFrage: {question}",
        },
        "vectordb": {
            "backend": "qdrant",
            "path": "/tmp/vectordb",
            "collection": "test",
        },
        "retrieval": {
            "reranking": True,
            "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "rerank_top_k": 5,
            "rerank_fetch_multiplier": 3,
            "max_sibling_chunks": 5,
        },
    }
    for key, val in overrides.items():
        if isinstance(val, dict) and key in cfg:
            cfg[key].update(val)
        else:
            cfg[key] = val
    return cfg


def _make_chunked_doc(source: str, num_chunks: int, texts: list[str] | None = None):
    """Erzeugt Chunks fuer ein Dokument mit source und chunk_index/chunk_total."""
    chunks = []
    for i in range(num_chunks):
        text = texts[i] if texts and i < len(texts) else f"Chunk {i} von {source}"
        chunks.append(
            {
                "text": text,
                "metadata": {
                    "source": source,
                    "chunk_index": i,
                    "chunk_total": num_chunks,
                },
                "score": round(1.0 - i * 0.05, 2),
            }
        )
    return chunks


class TestSiblingExpansionDirect:
    """Direkte Tests von _expand_sibling_chunks()."""

    def _make_chain(self, config=None):
        with (
            patch("kiara.generation.chain.HybridRetriever"),
            patch("kiara.generation.chain.CrossEncoderReranker"),
        ):
            from kiara.generation.chain import RAGChain

            return RAGChain(
                config or _base_config(),
                backend=_make_mock_backend(),
                prompt_template_override="Kontext:\n{context}\n\nFrage: {question}",
            )

    def test_small_doc_all_siblings_added(self):
        """Bei chunk_total <= max_sibling_chunks werden ALLE Geschwister ergaenzt."""
        chain = self._make_chain()

        # 3-Chunk-Dokument: Reranker waehlt nur chunk 1
        all_chunks = _make_chunked_doc(
            "email.txt",
            3,
            [
                "Betreff: Rechnung\nBetrag: 5,03 €",
                "Bitte beachten Sie die neue Nummerierung",
                "Registergericht Ansbach",
            ],
        )
        selected = [all_chunks[1]]  # Nur Boilerplate-Chunk
        candidates = all_chunks

        result = chain._expand_sibling_chunks(selected, candidates)

        # Alle 3 Chunks muessen enthalten sein
        assert len(result) == 3
        # Sortiert nach chunk_index
        assert result[0]["metadata"]["chunk_index"] == 0
        assert result[1]["metadata"]["chunk_index"] == 1
        assert result[2]["metadata"]["chunk_index"] == 2
        # Der Betrag ist jetzt im Kontext
        assert "5,03 €" in result[0]["text"]

    def test_large_doc_only_neighbors(self):
        """Bei chunk_total > max_sibling_chunks nur direkte Nachbarn (±1)."""
        chain = self._make_chain(_base_config(retrieval={"max_sibling_chunks": 3}))

        # 10-Chunk-Dokument: Reranker waehlt chunk 5
        all_chunks = _make_chunked_doc("handbuch.pdf", 10)
        selected = [all_chunks[5]]
        candidates = all_chunks

        result = chain._expand_sibling_chunks(selected, candidates)

        # chunk 4, 5, 6 muessen enthalten sein
        assert len(result) == 3
        indices = [r["metadata"]["chunk_index"] for r in result]
        assert sorted(indices) == [4, 5, 6]

    def test_no_expansion_when_already_complete(self):
        """Keine Expansion wenn alle Chunks bereits ausgewaehlt sind."""
        chain = self._make_chain()

        all_chunks = _make_chunked_doc("email.txt", 2)
        selected = all_chunks[:]  # Beide Chunks bereits gewaehlt
        candidates = all_chunks

        result = chain._expand_sibling_chunks(selected, candidates)
        assert len(result) == 2

    def test_no_expansion_without_metadata(self):
        """Chunks ohne chunk_index/chunk_total werden ignoriert."""
        chain = self._make_chain()

        selected = [{"text": "Hallo", "metadata": {"source": "x.txt"}, "score": 0.9}]
        candidates = [*selected, {"text": "Welt", "metadata": {"source": "x.txt"}, "score": 0.8}]

        result = chain._expand_sibling_chunks(selected, candidates)
        # Keine Expansion, da chunk_index fehlt
        assert len(result) == 1

    def test_multiple_sources_expanded(self):
        """Expansion funktioniert fuer mehrere Quellen gleichzeitig."""
        chain = self._make_chain()

        email1 = _make_chunked_doc("rechnung1.txt", 3)
        email2 = _make_chunked_doc("rechnung2.txt", 2)
        unrelated = _make_chunked_doc("handbuch.pdf", 100)

        # Reranker waehlt chunk 1 von email1, chunk 0 von email2 und chunk 50 von handbuch
        selected = [email1[1], email2[0], unrelated[50]]
        candidates = email1 + email2 + unrelated[:60]

        result = chain._expand_sibling_chunks(selected, candidates)

        sources = {r["metadata"]["source"] for r in result}
        assert "rechnung1.txt" in sources
        assert "rechnung2.txt" in sources
        assert "handbuch.pdf" in sources

        # email1: alle 3 Chunks (klein)
        email1_chunks = [r for r in result if r["metadata"]["source"] == "rechnung1.txt"]
        assert len(email1_chunks) == 3

        # email2: alle 2 Chunks (klein)
        email2_chunks = [r for r in result if r["metadata"]["source"] == "rechnung2.txt"]
        assert len(email2_chunks) == 2

        # handbuch: chunk 49, 50, 51 (gross, nur ±1)
        hb_chunks = [r for r in result if r["metadata"]["source"] == "handbuch.pdf"]
        assert len(hb_chunks) == 3
        hb_indices = sorted(r["metadata"]["chunk_index"] for r in hb_chunks)
        assert hb_indices == [49, 50, 51]

    def test_sorted_by_relevance_then_index(self):
        """Ergebnis nach Rerank-Relevanz gruppiert, innerhalb nach chunk_index."""
        chain = self._make_chain()

        a_chunks = _make_chunked_doc("a.txt", 3)
        b_chunks = _make_chunked_doc("b.txt", 2)

        # Reranker gibt b.txt zuerst (hoehere Relevanz), obwohl "a" < "b"
        selected = [b_chunks[1], a_chunks[2]]
        candidates = a_chunks + b_chunks

        result = chain._expand_sibling_chunks(selected, candidates)

        # b.txt zuerst (hoehere Rerank-Relevanz), dann a.txt
        result_sources = [r["metadata"]["source"] for r in result]
        assert result_sources == ["b.txt", "b.txt", "a.txt", "a.txt", "a.txt"]

        # Innerhalb jeder Gruppe nach chunk_index sortiert
        b_idx = [r["metadata"]["chunk_index"] for r in result if r["metadata"]["source"] == "b.txt"]
        a_idx = [r["metadata"]["chunk_index"] for r in result if r["metadata"]["source"] == "a.txt"]
        assert b_idx == [0, 1]
        assert a_idx == [0, 1, 2]

    def test_context_cap_keeps_reranked_hits_before_siblings(self):
        """Sibling-Expansion darf Top-Rerank-Treffer nicht aus dem Cap draengen."""
        chain = self._make_chain(
            _base_config(
                retrieval={
                    "max_sibling_chunks": 5,
                    "max_context_chunks": 2,
                },
            )
        )

        a_chunks = _make_chunked_doc("a.txt", 5)
        b_chunks = _make_chunked_doc("b.txt", 5)
        selected = [a_chunks[2], b_chunks[2]]
        candidates = a_chunks + b_chunks

        result = chain._expand_sibling_chunks(selected, candidates)

        assert result == selected

    def test_sibling_not_in_candidates_not_added(self):
        """Geschwister die nicht in candidates sind werden nicht ergaenzt."""
        chain = self._make_chain()
        chain._retriever.load_chunks_by_source.return_value = []

        # 3-Chunk-Dokument, aber nur chunk 1 und 2 in candidates
        chunks = _make_chunked_doc("email.txt", 3)
        selected = [chunks[1]]
        candidates = [chunks[1], chunks[2]]  # chunk 0 fehlt

        result = chain._expand_sibling_chunks(selected, candidates)

        # Nur chunk 1 und 2, chunk 0 ist nicht verfuegbar
        assert len(result) == 2
        indices = [r["metadata"]["chunk_index"] for r in result]
        assert sorted(indices) == [1, 2]

    def test_sibling_not_in_candidates_loaded_from_retriever(self):
        """Fehlende Geschwister werden per source aus dem Retriever nachgeladen."""
        chain = self._make_chain()

        chunks = _make_chunked_doc(
            "email.txt",
            3,
            [
                "Betrag: 5,03 EUR",
                "Boilerplate",
                "Footer",
            ],
        )
        chain._retriever.load_chunks_by_source.return_value = chunks

        result = chain._expand_sibling_chunks([chunks[1]], [chunks[1]])

        assert len(result) == 3
        assert [r["metadata"]["chunk_index"] for r in result] == [0, 1, 2]
        assert "5,03 EUR" in result[0]["text"]
        chain._retriever.load_chunks_by_source.assert_called_once_with(
            "email.txt",
            filter_datasource_ids=None,
        )

    def test_edge_at_document_boundary(self):
        """Nachbar-Expansion am Rand: chunk 0 hat keinen -1, letzter keinen +1."""
        chain = self._make_chain(_base_config(retrieval={"max_sibling_chunks": 2}))

        # 8-Chunk-Dokument (> max_sibling_chunks=2, also nur ±1)
        chunks = _make_chunked_doc("doc.pdf", 8)

        # chunk 0 gewaehlt → nur +1 Nachbar
        selected = [chunks[0]]
        candidates = chunks
        result = chain._expand_sibling_chunks(selected, candidates)
        indices = sorted(r["metadata"]["chunk_index"] for r in result)
        assert indices == [0, 1]

        # Letzter chunk gewaehlt → nur -1 Nachbar
        selected = [chunks[7]]
        result = chain._expand_sibling_chunks(selected, candidates)
        indices = sorted(r["metadata"]["chunk_index"] for r in result)
        assert indices == [6, 7]


class TestSiblingExpansionIntegration:
    """Integration: Sibling-Expansion in _retrieve_and_rerank()."""

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.generation.chain.CrossEncoderReranker")
    def test_expansion_called_when_reranker_enabled(self, MockReranker, MockRetriever):
        """Bei aktivem Reranking wird Sibling-Expansion ausgefuehrt."""
        from kiara.generation.chain import RAGChain

        # 3 Emails à 3 Chunks: Retriever findet alle 9
        email1 = _make_chunked_doc("re1.txt", 3, ["Betrag: 5€", "Boilerplate1", "Footer1"])
        email2 = _make_chunked_doc("re2.txt", 3, ["Betrag: 3€", "Boilerplate2", "Footer2"])
        email3 = _make_chunked_doc("re3.txt", 3, ["Betrag: 7€", "Boilerplate3", "Footer3"])
        all_candidates = email1 + email2 + email3

        mock_retriever = MockRetriever.return_value
        mock_retriever.retrieve.return_value = all_candidates

        # Reranker waehlt nur Boilerplate-Chunks (chunk_index=1)
        mock_reranker = MockReranker.return_value
        mock_reranker.enabled = True
        mock_reranker.rerank.return_value = [email1[1], email2[1], email3[1]]

        mock_backend = _make_mock_backend()

        config = _base_config(retrieval={"reranking": True})
        chain = RAGChain(
            config,
            backend=mock_backend,
            prompt_template_override="Kontext:\n{context}\n\nFrage: {question}",
        )

        result = chain.query("Wie viel bezahlt?", k=3)

        # LLM-Prompt muss ALLE Chunks enthalten (durch Expansion)
        # generate() wird mit (system_prompt, user_message) aufgerufen
        user_message = mock_backend.generate.call_args[0][1]
        assert "Betrag: 5€" in user_message
        assert "Betrag: 3€" in user_message
        assert "Betrag: 7€" in user_message

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.generation.chain.CrossEncoderReranker")
    def test_expansion_loads_missing_siblings_after_rerank(self, MockReranker, MockRetriever):
        """Rerank-Kandidaten duerfen nicht die einzige Quelle fuer Geschwister sein."""
        from kiara.generation.chain import RAGChain

        email = _make_chunked_doc(
            "re.txt",
            3,
            [
                "Betrag: 5 EUR",
                "Boilerplate",
                "Footer",
            ],
        )

        mock_retriever = MockRetriever.return_value
        mock_retriever.retrieve.return_value = [email[1]]
        mock_retriever.load_chunks_by_source.return_value = email

        mock_reranker = MockReranker.return_value
        mock_reranker.enabled = True
        mock_reranker.rerank.return_value = [email[1]]

        mock_backend = _make_mock_backend()
        chain = RAGChain(
            _base_config(retrieval={"reranking": True}),
            backend=mock_backend,
            prompt_template_override="Kontext:\n{context}\n\nFrage: {question}",
        )

        chain.query("Wie viel bezahlt?", k=1, filter_datasource_ids=[7])

        user_message = mock_backend.generate.call_args[0][1]
        assert "Betrag: 5 EUR" in user_message
        mock_retriever.load_chunks_by_source.assert_called_once_with(
            "re.txt",
            filter_datasource_ids=[7],
        )

    @patch("kiara.generation.chain.HybridRetriever")
    @patch("kiara.generation.chain.CrossEncoderReranker")
    def test_no_expansion_when_reranker_disabled(self, MockReranker, MockRetriever):
        """Bei deaktiviertem Reranking keine Expansion."""
        from kiara.generation.chain import RAGChain

        email = _make_chunked_doc("re.txt", 3, ["Betrag", "Boilerplate", "Footer"])

        mock_retriever = MockRetriever.return_value
        mock_retriever.retrieve.return_value = email

        mock_reranker = MockReranker.return_value
        mock_reranker.enabled = False
        mock_reranker.rerank.return_value = [email[1]]  # Nur Boilerplate

        mock_backend = _make_mock_backend()

        config = _base_config(retrieval={"reranking": False})
        chain = RAGChain(
            config,
            backend=mock_backend,
            prompt_template_override="Kontext:\n{context}\n\nFrage: {question}",
        )

        result = chain.query("Test?", k=1)

        # Nur der eine Chunk im Prompt (keine Expansion)
        # generate() wird mit (system_prompt, user_message) aufgerufen
        user_message = mock_backend.generate.call_args[0][1]
        assert "Boilerplate" in user_message
        assert "Betrag" not in user_message
