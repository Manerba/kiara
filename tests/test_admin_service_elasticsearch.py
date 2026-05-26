"""Direkte Unit-Tests fuer elasticsearch_service.py.

Testet Service-Logik ohne Admin-Funktions-Layer:
Background-Thread-Fehlerbehandlung, Status-Tracking, Guards,
Progress-Callback.
"""

import time
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Hilfsfunktion: Service-State zuruecksetzen
# ---------------------------------------------------------------------------


def _reset_es_state():
    """Setzt den globalen ES-Task-Status zurueck."""
    from kiara.admin.services.elasticsearch_service import _es_task

    _es_task.update(
        {
            "running": False,
            "action": None,
            "progress": None,
            "result": None,
        }
    )


# ---------------------------------------------------------------------------
# is_running / get_task_status
# ---------------------------------------------------------------------------


class TestESServiceState:
    def setup_method(self):
        _reset_es_state()

    def test_initial_state(self):
        """Initialzustand: nicht laufend, alles None."""
        from kiara.admin.services.elasticsearch_service import get_task_status, is_running

        assert is_running() is False
        status = get_task_status()
        assert status["running"] is False
        assert status["action"] is None

    def test_get_task_status_returns_copy(self):
        """get_task_status() gibt Kopie zurueck."""
        from kiara.admin.services.elasticsearch_service import _es_task, get_task_status

        status = get_task_status()
        status["running"] = True
        assert _es_task["running"] is False


# ---------------------------------------------------------------------------
# start_rebuild — Guards
# ---------------------------------------------------------------------------


class TestESServiceRebuild:
    def setup_method(self):
        _reset_es_state()

    @patch("kiara.retrieval.elasticsearch_service.is_configured")
    @patch("kiara.config.load_config")
    def test_rebuild_not_configured(self, mock_config, mock_configured):
        """ES nicht konfiguriert → RuntimeError."""
        mock_config.return_value = {}
        mock_configured.return_value = False

        from kiara.admin.services.elasticsearch_service import start_rebuild

        with pytest.raises(RuntimeError, match="nicht als Hybrid-Suche"):
            start_rebuild()

    @patch("kiara.retrieval.elasticsearch_service.is_configured")
    @patch("kiara.config.load_config")
    def test_rebuild_concurrent_guard(self, mock_config, mock_configured):
        """Zweiter Rebuild waehrend laufendem → RuntimeError."""
        mock_config.return_value = {}
        mock_configured.return_value = True
        from kiara.admin.services.elasticsearch_service import _es_task, start_rebuild

        _es_task["running"] = True

        with pytest.raises(RuntimeError, match="läuft bereits"):
            start_rebuild()

        _es_task["running"] = False

    @patch("kiara.retrieval.elasticsearch_service.rebuild_from_vectorstore")
    @patch("kiara.retrieval.elasticsearch_service.is_configured")
    @patch("kiara.config.load_config")
    def test_rebuild_thread_error(self, mock_config, mock_configured, mock_rebuild):
        """Thread-Exception → result mit status=error."""
        mock_config.return_value = {}
        mock_configured.return_value = True
        mock_rebuild.side_effect = ConnectionError("ES unreachable")

        from kiara.admin.services.elasticsearch_service import get_task_status, start_rebuild

        start_rebuild()

        for _ in range(20):
            time.sleep(0.1)
            if not get_task_status()["running"]:
                break

        status = get_task_status()
        assert status["running"] is False
        assert status["action"] is None  # Reset im finally
        assert status["result"]["status"] == "error"
        assert "ES unreachable" in status["result"]["error"]

    @patch("kiara.retrieval.elasticsearch_service.rebuild_from_vectorstore")
    @patch("kiara.retrieval.elasticsearch_service.is_configured")
    @patch("kiara.config.load_config")
    def test_rebuild_progress_callback(self, mock_config, mock_configured, mock_rebuild):
        """Progress-Callback aktualisiert _es_task['progress']."""
        mock_config.return_value = {}
        mock_configured.return_value = True

        captured_progress = []

        def mock_rebuild_fn(config, progress_callback=None):
            if progress_callback:
                progress_callback(50, 200)
                captured_progress.append(True)
            return {"indexed": 200, "errors": 0}

        mock_rebuild.side_effect = mock_rebuild_fn

        from kiara.admin.services.elasticsearch_service import get_task_status, start_rebuild

        start_rebuild()

        for _ in range(20):
            time.sleep(0.1)
            if not get_task_status()["running"]:
                break

        status = get_task_status()
        assert status["result"]["status"] == "success"
        assert len(captured_progress) == 1  # Callback wurde aufgerufen


# ---------------------------------------------------------------------------
# delete_index — Guards
# ---------------------------------------------------------------------------


class TestESServiceDelete:
    def setup_method(self):
        _reset_es_state()

    def test_delete_while_running(self):
        """Loeschen waehrend laufendem Task → RuntimeError."""
        from kiara.admin.services.elasticsearch_service import _es_task, delete_index

        _es_task["running"] = True

        with pytest.raises(RuntimeError, match="läuft bereits"):
            delete_index()

        _es_task["running"] = False

    @patch("kiara.retrieval.elasticsearch_service.delete_index")
    @patch("kiara.config.load_config")
    def test_delete_success(self, mock_config, mock_delete):
        """Erfolgreiche Loeschung → dict mit message."""
        mock_config.return_value = {}

        from kiara.admin.services.elasticsearch_service import delete_index

        result = delete_index()
        assert result["message"] == "Index geloescht"
        mock_delete.assert_called_once()


# ---------------------------------------------------------------------------
# start_verify — Thread-Fehlerbehandlung
# ---------------------------------------------------------------------------


class TestESServiceVerify:
    def setup_method(self):
        _reset_es_state()

    def test_verify_concurrent_guard(self):
        """Verify waehrend laufendem Task → RuntimeError."""
        from kiara.admin.services.elasticsearch_service import _es_task, start_verify

        _es_task["running"] = True

        with pytest.raises(RuntimeError, match="läuft bereits"):
            start_verify()

        _es_task["running"] = False

    @patch("kiara.retrieval.elasticsearch_service.verify_integrity")
    @patch("kiara.config.load_config")
    def test_verify_thread_error(self, mock_config, mock_verify):
        """Thread-Exception → result mit status=error."""
        mock_config.return_value = {}
        mock_verify.side_effect = Exception("Index not found")

        from kiara.admin.services.elasticsearch_service import get_task_status, start_verify

        start_verify()

        for _ in range(20):
            time.sleep(0.1)
            if not get_task_status()["running"]:
                break

        status = get_task_status()
        assert status["running"] is False
        assert status["action"] is None
        assert status["result"]["status"] == "error"
        assert "Index not found" in status["result"]["error"]

    @patch("kiara.retrieval.elasticsearch_service.verify_integrity")
    @patch("kiara.config.load_config")
    def test_verify_success(self, mock_config, mock_verify):
        """Erfolgreiche Verify → result mit status=success."""
        mock_config.return_value = {}
        mock_verify.return_value = {"es_count": 500, "qdrant_count": 500, "mismatches": 0}

        from kiara.admin.services.elasticsearch_service import get_task_status, start_verify

        start_verify()

        for _ in range(20):
            time.sleep(0.1)
            if not get_task_status()["running"]:
                break

        status = get_task_status()
        assert status["result"]["status"] == "success"
        assert status["result"]["es_count"] == 500


# ---------------------------------------------------------------------------
# get_es_status
# ---------------------------------------------------------------------------


class TestESServiceStatus:
    def setup_method(self):
        _reset_es_state()

    @patch("kiara.retrieval.vectorstore.VectorStore")
    @patch("kiara.retrieval.elasticsearch_service.get_status")
    @patch("kiara.config.load_config")
    def test_qdrant_failure_defaults_zero(self, mock_config, mock_es_status, mock_vs_cls):
        """Qdrant nicht erreichbar → qdrant_count = 0, kein Crash."""
        mock_config.return_value = {}
        mock_es_status.return_value = {"connected": True, "doc_count": 100}
        mock_vs_cls.side_effect = Exception("Qdrant connection refused")

        from kiara.admin.services.elasticsearch_service import get_es_status

        status = get_es_status()

        assert status["qdrant_count"] == 0
        assert status["connected"] is True
        assert status["task_running"] is False
