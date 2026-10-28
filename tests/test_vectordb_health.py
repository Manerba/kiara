"""Tests fuer den VectorDB Health-Check (Qdrant).

Testet check_vectordb() in kiara/health/checker.py.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from kiara.health.checker import ComponentHealth, HealthChecker

# ---------------------------------------------------------------------------
# Gemeinsame Configs
# ---------------------------------------------------------------------------


def _config(**extra_vectordb):
    """Erzeugt eine Minimal-Config mit Qdrant-Backend."""
    vdb = {
        "backend": "qdrant",
        "path": "/tmp/test_vectordb",
        "collection": "knowledgebase",
        "qdrant_url": "http://localhost:6333",
    }
    vdb.update(extra_vectordb)
    return {
        "vectordb": vdb,
        "ollama": {"url": "http://localhost:11434", "embedding_model": "nomic-embed-text"},
    }


# ---------------------------------------------------------------------------
# 1. test_health_check_qdrant_backend
# ---------------------------------------------------------------------------


def test_health_check_qdrant_backend():
    """Qdrant-Backend: _check_qdrant() wird via HTTP aufgerufen."""
    checker = HealthChecker(_config())

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": {"collections": [{"name": "kb_nomic-embed-text"}]}}

    # Mock httpx.AsyncClient als Context-Manager
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("kiara.health.checks_vectordb.httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(checker.check_vectordb())

    assert isinstance(result, ComponentHealth)
    assert result.name == "vectordb"
    assert result.status == "healthy"
    assert result.details["backend"] == "qdrant"


# ---------------------------------------------------------------------------
# 2. test_health_check_qdrant_unreachable
# ---------------------------------------------------------------------------


def test_health_check_qdrant_unreachable():
    """Qdrant nicht erreichbar -> unhealthy mit Fehlermeldung."""
    checker = HealthChecker(_config())

    mock_client = AsyncMock()
    mock_client.get.side_effect = ConnectionError("Connection refused")
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("kiara.health.checks_vectordb.httpx.AsyncClient", return_value=mock_client):
        result = asyncio.run(checker.check_vectordb())

    assert result.name == "vectordb"
    assert result.status == "unhealthy"
    assert "qdrant" in result.message.lower()


# ---------------------------------------------------------------------------
# 3. test_health_response_includes_backend_info
# ---------------------------------------------------------------------------


def test_health_response_includes_backend_info():
    """Die Health-Response details enthaelt den Backend-Typ 'qdrant'."""
    checker = HealthChecker(_config())

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": {"collections": []}}
    mock_async_client = AsyncMock()
    mock_async_client.get.return_value = mock_response
    mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
    mock_async_client.__aexit__ = AsyncMock(return_value=False)

    with patch("kiara.health.checks_vectordb.httpx.AsyncClient", return_value=mock_async_client):
        result = asyncio.run(checker.check_vectordb())

    assert result.details.get("backend") == "qdrant"
