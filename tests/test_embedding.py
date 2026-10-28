"""Tests for kiara.retrieval.embedding module."""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.usefixtures("s4_guard_passthrough")

from kiara.retrieval.embedding import (
    collection_name_for_model,
    embed_query,
    embed_texts,
    model_name_from_collection,
)


def _mock_httpx_response(embeddings):
    """Simuliert httpx.post-Response mit .json() + .raise_for_status()."""
    resp = MagicMock()
    resp.json.return_value = {"embeddings": embeddings}
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config():
    """Minimal config dict accepted by embed_texts / embed_query."""
    return {
        "ollama": {
            "url": "http://localhost:11434",
            "embedding_model": "nomic-embed-text",
            "embedding_url": "",
        },
    }


FAKE_EMBEDDINGS = [[0.1, 0.2, 0.3]]

# Mock-Backend fuer get_embedding_backend_for_model
_FAKE_BACKEND = {
    "url": "http://localhost:11434",
    "provider": "ollama",
    "model": "nomic-embed-text",
    "name": "test",
}


# ---------------------------------------------------------------------------
# embed_texts
# ---------------------------------------------------------------------------


@patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=3)
@patch("kiara.retrieval.embedding._get_ollama_http_client")
@patch("kiara.retrieval.embedding.get_embedding_backend_for_model", return_value=_FAKE_BACKEND)
def test_embed_texts_nomic_sends_input_type(mock_backend, mock_client, mock_dim, config):
    """Fuer nomic-embed-text wird input_type=search_document mitgesendet,
    KEIN lokaler Prefix (KIron-Proxy setzt ihn serverseitig)."""
    mock_http = MagicMock()
    mock_client.return_value = mock_http
    mock_http.post.return_value = _mock_httpx_response(FAKE_EMBEDDINGS)

    result = embed_texts(config, "nomic-embed-text", ["hello world"])

    mock_client.assert_called_once_with("http://localhost:11434")
    mock_http.post.assert_called_once()
    args, kwargs = mock_http.post.call_args
    assert args[0] == "/api/embed"
    payload = kwargs["json"]
    assert payload["model"] == "nomic-embed-text"
    assert payload["input"] == ["hello world"]
    assert payload["input_type"] == "search_document"
    assert result == FAKE_EMBEDDINGS


@patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=3)
@patch("kiara.retrieval.embedding._get_ollama_http_client")
@patch(
    "kiara.retrieval.embedding.get_embedding_backend_for_model",
    return_value={**_FAKE_BACKEND, "model": "unknown-model:latest"},
)
def test_embed_texts_arbitrary_model_still_sends_document_role(
    mock_backend, mock_client, mock_dim, config
):
    """Die Rolle stammt vom Dokumentvorgang, niemals vom Modellnamen."""
    mock_http = MagicMock()
    mock_client.return_value = mock_http
    mock_http.post.return_value = _mock_httpx_response(FAKE_EMBEDDINGS)

    result = embed_texts(config, "unknown-model", ["hello world"])

    _, kwargs = mock_http.post.call_args
    payload = kwargs["json"]
    assert payload["model"] == "unknown-model:latest"
    assert payload["input"] == ["hello world"]
    assert payload["input_type"] == "search_document"
    assert result == FAKE_EMBEDDINGS


# ---------------------------------------------------------------------------
# embed_query
# ---------------------------------------------------------------------------


@patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=3)
@patch("kiara.retrieval.embedding._get_ollama_http_client")
@patch("kiara.retrieval.embedding.get_embedding_backend_for_model", return_value=_FAKE_BACKEND)
def test_embed_query_nomic_sends_input_type(mock_backend, mock_client, mock_dim, config):
    """Fuer nomic-embed-text wird input_type=search_query mitgesendet,
    KEIN lokaler 'search_query: '-Prefix mehr."""
    mock_http = MagicMock()
    mock_client.return_value = mock_http
    mock_http.post.return_value = _mock_httpx_response(FAKE_EMBEDDINGS)

    result = embed_query(config, "nomic-embed-text", "hello world")

    args, kwargs = mock_http.post.call_args
    assert args[0] == "/api/embed"
    payload = kwargs["json"]
    assert payload["model"] == "nomic-embed-text"
    assert payload["input"] == ["hello world"]  # KEIN "search_query: "-Prefix
    assert payload["input_type"] == "search_query"
    assert result == FAKE_EMBEDDINGS[0]


@patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=3)
@patch("kiara.retrieval.embedding._get_ollama_http_client")
@patch(
    "kiara.retrieval.embedding.get_embedding_backend_for_model",
    return_value={
        "url": "http://localhost:11434",
        "provider": "ollama",
        "model": "mxbai-embed-large",
        "name": "test",
    },
)
def test_embed_query_mxbai_has_no_local_prefix_and_sends_role(
    mock_backend, mock_client, mock_dim, config
):
    """Auch ein historischer Sonderfall folgt nur noch dem Query-Vorgang."""
    mock_http = MagicMock()
    mock_client.return_value = mock_http
    mock_http.post.return_value = _mock_httpx_response(FAKE_EMBEDDINGS)

    result = embed_query(config, "mxbai-embed-large", "hello world")

    _, kwargs = mock_http.post.call_args
    payload = kwargs["json"]
    assert payload["input"] == ["hello world"]
    assert payload["input_type"] == "search_query"
    assert result == FAKE_EMBEDDINGS[0]


# ---------------------------------------------------------------------------
# collection_name_for_model
# ---------------------------------------------------------------------------


def test_collection_name_derivation():
    """Model name is converted to a safe collection name with 'kb_' prefix."""
    assert collection_name_for_model("nomic-embed-text") == "kb_nomic-embed-text"


def test_collection_name_normalizes_latest_tag():
    """Input mit :latest wird zentral normalisiert."""
    assert collection_name_for_model("nomic-embed-text:latest") == "kb_nomic-embed-text"


def test_collection_name_preserves_version_tag():
    """Non-latest version tags werden reversibel als _hh kodiert."""
    assert collection_name_for_model("nomic-embed-text:v2") == "kb_nomic-embed-text_3av2"


def test_collection_name_roundtrip():
    """Encoder -> Decoder Roundtrip fuer verschiedene Modellnamen."""
    models = [
        "nomic-embed-text",
        "hellord/e5-mistral-7b-instruct:q4_0",
        "llama3.1:70b-instruct-q4_k_m",
        "my model/v2@beta",
    ]
    for m in models:
        encoded = collection_name_for_model(m)
        decoded = model_name_from_collection(encoded)
        assert decoded == m, f"Roundtrip fehlgeschlagen: {m!r} -> {encoded!r} -> {decoded!r}"


def test_collection_name_empty_raises():
    """Leerer Modellname wirft ValueError."""
    with pytest.raises(ValueError, match="nicht-leeren"):
        collection_name_for_model("")


def test_model_name_from_collection_invalid_prefix():
    """Collection ohne kb_-Prefix wirft ValueError."""
    with pytest.raises(ValueError, match="Kein KIara-Collection-Name"):
        model_name_from_collection("other_collection")
