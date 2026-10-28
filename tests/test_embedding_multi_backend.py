"""Tests fuer Multi-Backend-Funktionen im Embedding-Modul.

Testet:
- _get_url_for_model: Backend-Lookup, BackendUnavailableError, NotImplementedError
- _get_ollama_client: Thread-safe Caching
- embed_texts / embed_query: Normalisierung, ResponseError-Klassifizierung
- embed_texts_late: Late-Chunking Embeddings via httpx POST
"""

import logging
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.usefixtures("s4_guard_passthrough")

from kiara.backend_selector import BackendUnavailableError
from kiara.retrieval.embedding import (
    _client_lock,
    _get_ollama_client,
    _ollama_clients,
    _ollama_http_clients,
    _openai_clients,
    _voyage_clients,
    embed_query,
    embed_texts,
    invalidate_clients,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_client_cache():
    """Client-Cache vor/nach jedem Test zuruecksetzen."""
    with _client_lock:
        _ollama_clients.clear()
        _ollama_http_clients.clear()
        _openai_clients.clear()
        _voyage_clients.clear()
    yield
    with _client_lock:
        _ollama_clients.clear()
        _ollama_http_clients.clear()
        _openai_clients.clear()
        _voyage_clients.clear()


_FAKE_BACKEND = {
    "url": "http://localhost:11434",
    "provider": "ollama",
    "model": "nomic-embed-text",
    "name": "test-backend",
}
_FAKE_BACKEND_OPENAI = {
    "url": "http://openai-proxy:8080",
    "provider": "openai",
    "model": "text-embedding-3-small",
    "name": "openai-embed",
}
_CONFIG = {
    "ollama": {"url": "http://localhost:11434", "embedding_model": "nomic-embed-text"},
}
FAKE_EMBEDDINGS = [[0.1, 0.2, 0.3]]


# ---------------------------------------------------------------------------
# _get_ollama_client
# ---------------------------------------------------------------------------


class TestGetOllamaClient:
    @patch("kiara.retrieval.embedding.ollama_client.Client")
    def test_creates_new_client(self, mock_client_cls):
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance
        client = _get_ollama_client("http://localhost:11434")
        mock_client_cls.assert_called_once_with(host="http://localhost:11434", timeout=120.0)
        assert client is mock_instance

    @patch("kiara.retrieval.embedding.ollama_client.Client")
    def test_caches_client_per_url(self, mock_client_cls):
        """Zweiter Aufruf mit gleicher URL gibt gecachten Client zurueck."""
        mock_client_cls.return_value = MagicMock()
        client1 = _get_ollama_client("http://localhost:11434")
        client2 = _get_ollama_client("http://localhost:11434")
        assert client1 is client2
        assert mock_client_cls.call_count == 1

    @patch("kiara.retrieval.embedding.ollama_client.Client")
    def test_different_urls_different_clients(self, mock_client_cls):
        """Unterschiedliche URLs bekommen unterschiedliche Clients."""
        mock_client_cls.side_effect = [MagicMock(), MagicMock()]
        client1 = _get_ollama_client("http://server-a:11434")
        client2 = _get_ollama_client("http://server-b:11434")
        assert client1 is not client2
        assert mock_client_cls.call_count == 2

    @patch("kiara.retrieval.embedding.ollama_client.Client")
    def test_thread_safe(self, mock_client_cls):
        """Concurrent Zugriffe erzeugen nur einen Client pro URL."""
        mock_client_cls.return_value = MagicMock()
        results = []
        barrier = threading.Barrier(4)

        def get_client():
            barrier.wait()
            results.append(_get_ollama_client("http://localhost:11434"))

        threads = [threading.Thread(target=get_client) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(id(c) for c in results)) == 1  # alle dasselbe Objekt
        assert mock_client_cls.call_count == 1


class TestInvalidateEmbeddingClients:
    def test_invalidate_clients_clears_all_embedding_client_caches(self):
        """Backend-Cache-Invalidierung schliesst alte Connection-Pools."""
        clients = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
        with _client_lock:
            _ollama_clients["http://ollama"] = clients[0]
            _ollama_http_clients["http://ollama"] = clients[1]
            _openai_clients[("old-key", "")] = clients[2]
            _voyage_clients["old-key"] = clients[3]

        invalidate_clients("embedding")

        with _client_lock:
            assert _ollama_clients == {}
            assert _ollama_http_clients == {}
            assert _openai_clients == {}
            assert _voyage_clients == {}
        for client in clients:
            client.close.assert_called_once()

    def test_invalidate_clients_ignores_unrelated_category(self):
        client = MagicMock()
        with _client_lock:
            _openai_clients[("key", "")] = client

        invalidate_clients("llm")

        with _client_lock:
            assert _openai_clients[("key", "")] is client
        client.close.assert_not_called()


# ---------------------------------------------------------------------------
# embed_texts — Normalisierung + ResponseError
# ---------------------------------------------------------------------------


def _httpx_response(embeddings):
    resp = MagicMock()
    resp.json.return_value = {"embeddings": embeddings}
    resp.raise_for_status = MagicMock()
    return resp


def _httpx_http_error(status_code: int, body: str):
    """Erzeugt eine httpx.HTTPStatusError-Instanz mit gesetztem response."""
    import httpx as httpx_mod

    err_response = MagicMock()
    err_response.status_code = status_code
    err_response.text = body
    exc = httpx_mod.HTTPStatusError(
        f"HTTP {status_code}", request=MagicMock(), response=err_response
    )
    return exc


class TestEmbedTextsMultiBackend:
    @patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=3)
    @patch("kiara.retrieval.embedding._get_ollama_http_client")
    @patch("kiara.retrieval.embedding.get_embedding_backend_for_model", return_value=_FAKE_BACKEND)
    def test_normalizes_model_name(self, mock_backend, mock_client, mock_dim):
        """':latest' wird vor API-Call entfernt."""
        mock_http = MagicMock()
        mock_client.return_value = mock_http
        mock_http.post.return_value = _httpx_response(FAKE_EMBEDDINGS)

        embed_texts(_CONFIG, "Nomic-Embed-Text:latest", ["hello"])

        # Modell in der HTTP-Payload ist normalisiert
        mock_http.post.assert_called_once()
        payload = mock_http.post.call_args.kwargs["json"]
        assert payload["model"] == "nomic-embed-text"

    def test_empty_model_raises_value_error(self):
        with pytest.raises(ValueError, match="Kein Embedding-Modell"):
            embed_texts(_CONFIG, "", ["hello"])

    def test_none_model_raises_value_error(self):
        with pytest.raises(ValueError, match="Kein Embedding-Modell"):
            embed_texts(_CONFIG, None, ["hello"])

    @patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=3)
    @patch("kiara.retrieval.embedding._get_ollama_http_client")
    @patch("kiara.retrieval.embedding.get_embedding_backend_for_model", return_value=_FAKE_BACKEND)
    def test_http_404_becomes_backend_unavailable(self, mock_backend, mock_client, mock_dim):
        """HTTP 404 'model not found' -> BackendUnavailableError."""
        resp = MagicMock()
        resp.raise_for_status.side_effect = _httpx_http_error(404, "model 'xyz' not found")
        mock_http = MagicMock()
        mock_client.return_value = mock_http
        mock_http.post.return_value = resp

        with pytest.raises(BackendUnavailableError, match="nicht.*gefunden"):
            embed_texts(_CONFIG, "nomic-embed-text", ["hello"])

    @patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=3)
    @patch("kiara.retrieval.embedding._get_ollama_http_client")
    @patch("kiara.retrieval.embedding.get_embedding_backend_for_model", return_value=_FAKE_BACKEND)
    def test_http_500_becomes_backend_unavailable(self, mock_backend, mock_client, mock_dim):
        """Anderer HTTP-Fehler wird ebenfalls als BackendUnavailableError gewrapt."""
        resp = MagicMock()
        resp.raise_for_status.side_effect = _httpx_http_error(500, "internal server error")
        mock_http = MagicMock()
        mock_client.return_value = mock_http
        mock_http.post.return_value = resp

        with pytest.raises(BackendUnavailableError, match="HTTP 500"):
            embed_texts(_CONFIG, "nomic-embed-text", ["hello"])


# ---------------------------------------------------------------------------
# embed_query — Normalisierung + ResponseError
# ---------------------------------------------------------------------------


class TestEmbedQueryMultiBackend:
    @patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=3)
    @patch("kiara.retrieval.embedding._get_ollama_http_client")
    @patch("kiara.retrieval.embedding.get_embedding_backend_for_model", return_value=_FAKE_BACKEND)
    def test_normalizes_model_name(self, mock_backend, mock_client, mock_dim):
        mock_http = MagicMock()
        mock_client.return_value = mock_http
        mock_http.post.return_value = _httpx_response(FAKE_EMBEDDINGS)

        embed_query(_CONFIG, "NOMIC-EMBED-TEXT:LATEST", "test query")

        payload = mock_http.post.call_args.kwargs["json"]
        assert payload["model"] == "nomic-embed-text"

    def test_empty_model_raises_value_error(self):
        with pytest.raises(ValueError, match="Kein Embedding-Modell"):
            embed_query(_CONFIG, "", "test query")

    @patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=3)
    @patch("kiara.retrieval.embedding._get_ollama_http_client")
    @patch("kiara.retrieval.embedding.get_embedding_backend_for_model", return_value=_FAKE_BACKEND)
    def test_http_404_not_found(self, mock_backend, mock_client, mock_dim):
        resp = MagicMock()
        resp.raise_for_status.side_effect = _httpx_http_error(404, "model 'xyz' not found")
        mock_http = MagicMock()
        mock_client.return_value = mock_http
        mock_http.post.return_value = resp

        with pytest.raises(BackendUnavailableError):
            embed_query(_CONFIG, "nomic-embed-text", "test query")

    @patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=3)
    @patch("kiara.retrieval.embedding._get_ollama_http_client")
    @patch("kiara.retrieval.embedding.get_embedding_backend_for_model", return_value=_FAKE_BACKEND)
    def test_returns_single_vector(self, mock_backend, mock_client, mock_dim):
        """embed_query gibt einen einzelnen Vektor zurueck, keine Liste von Listen."""
        mock_http = MagicMock()
        mock_client.return_value = mock_http
        mock_http.post.return_value = _httpx_response([[0.1, 0.2, 0.3]])

        result = embed_query(_CONFIG, "nomic-embed-text", "hello")
        assert result == [0.1, 0.2, 0.3]
        assert not isinstance(result[0], list)


class TestVoyageOperationRoles:
    def test_document_and_query_operations_map_to_native_voyage_roles(self):
        """Die Provider-Abbildung stammt aus der kanonischen KIara-Operation."""
        fake_voyage = SimpleNamespace(
            error=SimpleNamespace(AuthenticationError=type("AuthenticationError", (Exception,), {}))
        )
        backend = {
            "url": "https://api.voyageai.com",
            "provider": "voyage",
            "api_key": "test-key",
            "base_url": "",
            "model": "arbitrary-model-name",
        }
        client = MagicMock()
        client.embed.side_effect = [
            SimpleNamespace(embeddings=[[0.1, 0.2]]),
            SimpleNamespace(embeddings=[[0.3, 0.4]]),
        ]

        with (
            patch.dict(sys.modules, {"voyageai": fake_voyage}),
            patch("kiara.retrieval.embedding._resolve_backend", return_value=backend),
            patch("kiara.retrieval.embedding._get_voyage_client", return_value=client),
            patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=2),
        ):
            assert embed_texts(_CONFIG, "arbitrary-model-name", ["document"]) == [[0.1, 0.2]]
            assert embed_query(_CONFIG, "arbitrary-model-name", "query") == [0.3, 0.4]

        assert client.embed.call_args_list[0].kwargs["input_type"] == "document"
        assert client.embed.call_args_list[1].kwargs["input_type"] == "query"


# ---------------------------------------------------------------------------
# embed_texts_late — Late-Chunking Embeddings
# ---------------------------------------------------------------------------


class TestEmbedTextsLate:
    """Tests fuer embed_texts_late(): Late-Chunking via httpx POST."""

    @patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=2)
    @patch("kiara.retrieval.embedding.httpx.post")
    @patch("kiara.retrieval.embedding._resolve_backend", return_value=_FAKE_BACKEND)
    def test_embed_texts_late_success(self, mock_backend, mock_post, mock_dim):
        """Erfolgreicher Aufruf: 2 Chunks → 2 Embeddings."""
        from kiara.retrieval.embedding import embed_texts_late

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        mock_post.return_value = mock_response

        result = embed_texts_late(
            _CONFIG,
            "nomic-embed-text",
            "full document text",
            ["chunk1", "chunk2"],
            [(0, 6), (7, 13)],
        )
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        assert len(result) == 2
        assert mock_post.call_args.kwargs["json"]["input_type"] == "search_document"

    @patch("kiara.retrieval.embedding.httpx.post")
    @patch("kiara.retrieval.embedding._resolve_backend", return_value=_FAKE_BACKEND)
    def test_embed_texts_late_connection_error(self, mock_backend, mock_post, caplog):
        """httpx ConnectionError → None, Warning geloggt."""
        import httpx as httpx_mod

        from kiara.retrieval.embedding import embed_texts_late

        mock_post.side_effect = httpx_mod.ConnectError("Connection refused")

        with caplog.at_level(logging.WARNING):
            result = embed_texts_late(
                _CONFIG,
                "nomic-embed-text",
                "doc text",
                ["chunk"],
                [(0, 5)],
            )
        assert result is None
        assert any("fehlgeschlagen" in r.message for r in caplog.records)

    @patch("kiara.retrieval.embedding.httpx.post")
    @patch("kiara.retrieval.embedding._resolve_backend", return_value=_FAKE_BACKEND)
    def test_embed_texts_late_http_400(self, mock_backend, mock_post):
        """httpx HTTP 400 → None."""
        import httpx as httpx_mod

        from kiara.retrieval.embedding import embed_texts_late

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx_mod.HTTPStatusError(
            "400 Bad Request",
            request=MagicMock(),
            response=MagicMock(),
        )
        mock_post.return_value = mock_response

        result = embed_texts_late(
            _CONFIG,
            "nomic-embed-text",
            "doc text",
            ["chunk"],
            [(0, 5)],
        )
        assert result is None

    @patch("kiara.retrieval.embedding._resolve_backend", return_value=_FAKE_BACKEND)
    def test_embed_texts_late_empty_chunks(self, mock_backend):
        """Leere Chunk-Liste → leere Liste []."""
        from kiara.retrieval.embedding import embed_texts_late

        result = embed_texts_late(_CONFIG, "nomic-embed-text", "doc", [], [])
        assert result == []

    @patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=2)
    @patch("kiara.retrieval.embedding.httpx.post")
    @patch(
        "kiara.retrieval.embedding._resolve_backend",
        return_value={**_FAKE_BACKEND, "model": "mxbai-embed-large:latest"},
    )
    def test_embed_texts_late_has_no_model_name_allowlist(self, mock_backend, mock_post, mock_dim):
        """Ein verifiziertes Late-Profil gilt unabhaengig vom Modellnamen."""
        from kiara.retrieval.embedding import embed_texts_late

        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"embeddings": [[0.1, 0.2]]}
        mock_post.return_value = response
        result = embed_texts_late(
            _CONFIG,
            "mxbai-embed-large",
            "doc",
            ["chunk"],
            [(0, 5)],
        )
        assert result == [[0.1, 0.2]]
        payload = mock_post.call_args.kwargs["json"]
        assert payload["input_type"] == "search_document"
        assert payload["model"] == "mxbai-embed-large:latest"

    @patch("kiara.retrieval.embedding._resolve_backend")
    def test_embed_texts_late_backend_error(self, mock_backend):
        """_resolve_backend wirft BackendUnavailableError → wird durchgereicht (fail-closed)."""
        from kiara.retrieval.embedding import embed_texts_late

        mock_backend.side_effect = BackendUnavailableError("Kein Backend erreichbar")

        with pytest.raises(BackendUnavailableError):
            embed_texts_late(
                _CONFIG,
                "nomic-embed-text",
                "doc",
                ["chunk"],
                [(0, 5)],
            )

    @patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=1)
    @patch("kiara.retrieval.embedding.httpx.post")
    @patch("kiara.retrieval.embedding._resolve_backend", return_value=_FAKE_BACKEND)
    def test_embed_texts_late_model_normalization(self, mock_backend, mock_post, mock_dim):
        """Modell 'Nomic-Embed-Text:latest' → HTTP-Payload verwendet normalisiertes 'nomic-embed-text'."""
        from kiara.retrieval.embedding import embed_texts_late

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1]]}
        mock_post.return_value = mock_response

        embed_texts_late(
            _CONFIG,
            "Nomic-Embed-Text:latest",
            "doc",
            ["chunk"],
            [(0, 5)],
        )

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["model"] == "nomic-embed-text"

    @patch("kiara.retrieval.embedding.httpx.post")
    @patch("kiara.retrieval.embedding._resolve_backend", return_value=_FAKE_BACKEND)
    def test_embed_texts_late_wrong_embedding_count(self, mock_backend, mock_post):
        """Backend gibt 1 Embedding fuer 2 Chunks → None."""
        from kiara.retrieval.embedding import embed_texts_late

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2]]}
        mock_post.return_value = mock_response

        result = embed_texts_late(
            _CONFIG,
            "nomic-embed-text",
            "doc",
            ["chunk1", "chunk2"],
            [(0, 6), (7, 13)],
        )
        assert result is None

    def test_embed_texts_late_spans_mismatch(self):
        """len(chunk_spans) != len(chunks) → None."""
        from kiara.retrieval.embedding import embed_texts_late

        result = embed_texts_late(
            _CONFIG,
            "nomic-embed-text",
            "doc",
            ["chunk1", "chunk2"],
            [(0, 6)],  # 2 Chunks, 1 Span
        )
        assert result is None

    @patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=1)
    @patch("kiara.retrieval.embedding.httpx.post")
    @patch("kiara.retrieval.embedding._resolve_backend", return_value=_FAKE_BACKEND)
    def test_embed_texts_late_payload_format(self, mock_backend, mock_post, mock_dim):
        """HTTP-Payload hat korrekte Struktur mit text, char_start, char_end pro Chunk."""
        from kiara.retrieval.embedding import embed_texts_late

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1], [0.2]]}
        mock_post.return_value = mock_response

        embed_texts_late(
            _CONFIG,
            "nomic-embed-text",
            "Hello World",
            ["Hello", "World"],
            [(0, 5), (6, 11)],
        )

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["model"] == "nomic-embed-text"
        assert payload["document"] == "Hello World"
        assert len(payload["chunks"]) == 2
        assert payload["chunks"][0] == {"text": "Hello", "char_start": 0, "char_end": 5}
        assert payload["chunks"][1] == {"text": "World", "char_start": 6, "char_end": 11}

    @patch("kiara.retrieval.embedding.httpx.post")
    @patch("kiara.retrieval.embedding._resolve_backend", return_value=_FAKE_BACKEND_OPENAI)
    def test_embed_texts_late_non_ollama_provider(self, mock_backend, mock_post):
        """_resolve_backend gibt provider='openai' → None, kein httpx.post Aufruf."""
        from kiara.retrieval.embedding import embed_texts_late

        result = embed_texts_late(
            _CONFIG,
            "nomic-embed-text",
            "doc",
            ["chunk"],
            [(0, 5)],
        )
        assert result is None
        mock_post.assert_not_called()

    @patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=1)
    @patch("kiara.retrieval.embedding.httpx.post")
    @patch("kiara.retrieval.embedding._resolve_backend", return_value=_FAKE_BACKEND)
    def test_embed_texts_late_uses_backend_url(self, mock_backend, mock_post, mock_dim):
        """Request geht an backend['url']/api/embed_late."""
        from kiara.retrieval.embedding import embed_texts_late

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1]]}
        mock_post.return_value = mock_response

        embed_texts_late(
            _CONFIG,
            "nomic-embed-text",
            "doc",
            ["chunk"],
            [(0, 5)],
        )

        call_args = mock_post.call_args
        url = call_args.args[0] if call_args.args else call_args.kwargs.get("url")
        assert url == "http://localhost:11434/api/embed_late"

    def test_embed_texts_late_empty_model(self):
        """Leerer Modellname → ValueError."""
        from kiara.retrieval.embedding import embed_texts_late

        with pytest.raises(ValueError, match="Kein Embedding-Modell"):
            embed_texts_late(_CONFIG, "", "doc", ["chunk"], [(0, 5)])

    @patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=768)
    @patch("kiara.retrieval.embedding.httpx.post")
    @patch("kiara.retrieval.embedding._resolve_backend", return_value=_FAKE_BACKEND)
    def test_embed_texts_late_dimension_mismatch_raises(self, mock_backend, mock_post, mock_dim):
        """Backend liefert 2-dim Vektor, erwartet aber 768 → ValueError (Modell-Drift)."""
        from kiara.retrieval.embedding import embed_texts_late

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2]]}
        mock_post.return_value = mock_response

        with pytest.raises(ValueError, match="Embedding-Dimension-Mismatch"):
            embed_texts_late(
                _CONFIG,
                "nomic-embed-text",
                "doc",
                ["chunk"],
                [(0, 5)],
            )
