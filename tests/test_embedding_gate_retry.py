"""Tests fuer den GPU-Gate-Retry in _ollama_embed (kiara.retrieval.embedding)."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from kiara.retrieval import embedding as emb
from kiara.retrieval.embedding import BackendUnavailableError, _ollama_embed

GATE_BODY = json.dumps(
    {
        "error": "GPU-Operation blockiert",
        "hint": "Request spaeter wiederholen oder num_gpu=0 explizit setzen.",
        "reason": "gpu_overlay_active",
    }
)


def _response(status_code: int, text: str = "", payload: dict | None = None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if payload is not None:
        resp.json.return_value = payload
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _ok_response():
    return _response(200, payload={"embeddings": [[0.1, 0.2]]})


class TestGateRetry:
    def test_gate_503_retried_until_success(self):
        client = MagicMock()
        client.post.side_effect = [
            _response(503, GATE_BODY),
            _response(503, GATE_BODY),
            _ok_response(),
        ]
        with (
            patch.object(emb, "_get_ollama_http_client", return_value=client),
            patch.object(emb.time, "sleep") as mock_sleep,
        ):
            result = _ollama_embed("http://kiron:11434", "nomic-embed-text", ["q"], "search_query")
        assert result == [[0.1, 0.2]]
        assert client.post.call_count == 3
        assert mock_sleep.call_count == 2

    def test_gate_409_retried(self):
        client = MagicMock()
        client.post.side_effect = [_response(409, GATE_BODY), _ok_response()]
        with (
            patch.object(emb, "_get_ollama_http_client", return_value=client),
            patch.object(emb.time, "sleep"),
        ):
            result = _ollama_embed("http://kiron:11434", "nomic-embed-text", ["q"], "search_query")
        assert result == [[0.1, 0.2]]

    def test_gate_retry_gives_up_after_deadline(self):
        client = MagicMock()
        client.post.return_value = _response(503, GATE_BODY)
        clock = iter(range(0, 1000, 10))  # 10s pro Aufruf -> Deadline nach ~9 Versuchen
        with (
            patch.object(emb, "_get_ollama_http_client", return_value=client),
            patch.object(emb.time, "sleep"),
            patch.object(emb.time, "monotonic", side_effect=lambda: next(clock)),
            pytest.raises(BackendUnavailableError, match="HTTP 503"),
        ):
            _ollama_embed("http://kiron:11434", "nomic-embed-text", ["q"], "search_query")
        assert client.post.call_count > 1

    def test_plain_503_without_marker_fails_immediately(self):
        client = MagicMock()
        client.post.return_value = _response(503, "Service Unavailable")
        with (
            patch.object(emb, "_get_ollama_http_client", return_value=client),
            pytest.raises(BackendUnavailableError, match="HTTP 503"),
        ):
            _ollama_embed("http://kiron:11434", "nomic-embed-text", ["q"], "search_query")
        assert client.post.call_count == 1

    def test_404_fails_immediately_with_model_hint(self):
        client = MagicMock()
        client.post.return_value = _response(404, "model not found")
        with (
            patch.object(emb, "_get_ollama_http_client", return_value=client),
            pytest.raises(BackendUnavailableError, match="nicht auf Embedding-Server"),
        ):
            _ollama_embed("http://kiron:11434", "nomic-embed-text", ["q"], "search_query")

    def test_transport_error_fails_immediately(self):
        client = MagicMock()
        client.post.side_effect = httpx.ConnectError("nein")
        with (
            patch.object(emb, "_get_ollama_http_client", return_value=client),
            pytest.raises(BackendUnavailableError, match="nicht erreichbar"),
        ):
            _ollama_embed("http://kiron:11434", "nomic-embed-text", ["q"], "search_query")
