"""Tests fuer Sprint 08: Elasticsearch Admin-Funktionen.

Testet elasticsearch.status, .rebuild, .delete, .verify, .task — 6 Unit-Tests.
"""

import asyncio
from unittest.mock import patch


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


# ---------------------------------------------------------------------------
# elasticsearch.status
# ---------------------------------------------------------------------------


class TestElasticsearchStatus:
    @patch("kiara.admin.services.elasticsearch_service.get_es_status")
    def test_es_status(self, mock_status):
        """Test 13: ES-Status mit Chunk-Counts."""
        mock_status.return_value = {
            "connected": True,
            "index_exists": True,
            "doc_count": 500,
            "qdrant_count": 500,
            "task_running": False,
            "task_action": None,
            "task_progress": None,
        }

        from kiara.admin.functions.elasticsearch import elasticsearch_status

        result = _run(elasticsearch_status(MOCK_USER))

        assert result.success is True
        assert result.data["connected"] is True
        assert result.data["qdrant_count"] == 500


# ---------------------------------------------------------------------------
# elasticsearch.rebuild
# ---------------------------------------------------------------------------


class TestElasticsearchRebuild:
    @patch("kiara.admin.services.elasticsearch_service.start_rebuild")
    @patch("kiara.admin.services.elasticsearch_service.is_running", return_value=False)
    def test_es_rebuild(self, mock_running, mock_start):
        """Test 14: Rebuild startet mit Confirmation."""
        from kiara.admin.functions.elasticsearch import elasticsearch_rebuild

        result = _run(elasticsearch_rebuild(MOCK_USER))

        assert result.success is True
        assert "gestartet" in result.data["message"]
        mock_start.assert_called_once()

    @patch("kiara.admin.services.elasticsearch_service.is_running", return_value=True)
    def test_es_rebuild_concurrent_guard(self, mock_running):
        """Test 15: Doppelter Rebuild wird abgelehnt."""
        from kiara.admin.functions.elasticsearch import elasticsearch_rebuild

        result = _run(elasticsearch_rebuild(MOCK_USER))

        assert result.success is False
        assert "läuft bereits" in result.message


# ---------------------------------------------------------------------------
# elasticsearch.delete
# ---------------------------------------------------------------------------


class TestElasticsearchDelete:
    @patch("kiara.admin.services.elasticsearch_service.delete_index")
    def test_es_delete(self, mock_delete):
        """Test 16: Index-Loeschung mit Confirmation."""
        mock_delete.return_value = {"message": "Index geloescht"}

        from kiara.admin.functions.elasticsearch import elasticsearch_delete

        result = _run(elasticsearch_delete(MOCK_USER))

        assert result.success is True
        assert "gelöscht" in result.message


# ---------------------------------------------------------------------------
# elasticsearch.verify
# ---------------------------------------------------------------------------


class TestElasticsearchVerify:
    @patch("kiara.admin.services.elasticsearch_service.start_verify")
    @patch("kiara.admin.services.elasticsearch_service.is_running", return_value=False)
    def test_es_verify(self, mock_running, mock_start):
        """Test 17: Verify startet Background-Thread."""
        from kiara.admin.functions.elasticsearch import elasticsearch_verify

        result = _run(elasticsearch_verify(MOCK_USER))

        assert result.success is True
        assert "gestartet" in result.data["message"]


# ---------------------------------------------------------------------------
# elasticsearch.task
# ---------------------------------------------------------------------------


class TestElasticsearchTask:
    @patch("kiara.admin.services.elasticsearch_service.get_task_status")
    def test_es_task_status(self, mock_status):
        """Test 18: Task-Status korrekt."""
        mock_status.return_value = {
            "running": True,
            "action": "rebuild",
            "progress": {"indexed": 100, "total": 500},
            "result": None,
        }

        from kiara.admin.functions.elasticsearch import elasticsearch_task

        result = _run(elasticsearch_task(MOCK_USER))

        assert result.success is True
        assert result.data["action"] == "rebuild"
        assert result.data["progress"]["indexed"] == 100


# ---------------------------------------------------------------------------
# ES Error-Tracking
# ---------------------------------------------------------------------------


class TestElasticsearchErrorTracking:
    @patch("kiara.admin.services.elasticsearch_service.get_task_status")
    def test_es_task_error_result(self, mock_status):
        """Task-Status nach Thread-Crash → status=error mit Detail."""
        mock_status.return_value = {
            "running": False,
            "action": None,
            "progress": {"indexed": 42, "total": 500},
            "result": {"status": "error", "error": "ConnectionError: ES unreachable"},
        }

        from kiara.admin.functions.elasticsearch import elasticsearch_task

        result = _run(elasticsearch_task(MOCK_USER))

        assert result.success is True  # Status-Abfrage ok
        assert result.data["result"]["status"] == "error"
        assert "ES unreachable" in result.data["result"]["error"]

    @patch("kiara.admin.services.elasticsearch_service.get_task_status")
    def test_es_task_idle_state(self, mock_status):
        """Idle-State: kein Task lief → alles None/False."""
        mock_status.return_value = {
            "running": False,
            "action": None,
            "progress": None,
            "result": None,
        }

        from kiara.admin.functions.elasticsearch import elasticsearch_task

        result = _run(elasticsearch_task(MOCK_USER))

        assert result.success is True
        assert result.data["running"] is False
        assert result.data["result"] is None

    @patch("kiara.admin.services.elasticsearch_service.start_rebuild")
    @patch("kiara.admin.services.elasticsearch_service.is_running", return_value=False)
    def test_es_rebuild_not_configured(self, mock_running, mock_start):
        """ES nicht konfiguriert → RuntimeError → AdminResult.fail."""
        mock_start.side_effect = RuntimeError(
            "Elasticsearch ist nicht als Hybrid-Suche konfiguriert"
        )

        from kiara.admin.functions.elasticsearch import elasticsearch_rebuild

        result = _run(elasticsearch_rebuild(MOCK_USER))

        assert result.success is False
        assert "nicht als Hybrid-Suche konfiguriert" in result.message

    @patch("kiara.admin.services.elasticsearch_service.get_es_status")
    def test_es_status_exception(self, mock_status):
        """get_es_status() Exception → AdminResult.fail."""
        mock_status.side_effect = Exception("ES connection timeout")

        from kiara.admin.functions.elasticsearch import elasticsearch_status

        result = _run(elasticsearch_status(MOCK_USER))

        assert result.success is False
        assert "fehlgeschlagen" in result.message
