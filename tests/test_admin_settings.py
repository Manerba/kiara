"""Tests fuer Sprint 08: Settings Admin-Funktionen.

Testet system.settings.read, .update, system.vectordb.test — 7 Unit-Tests.
"""

import asyncio
from unittest.mock import MagicMock, patch


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


# ---------------------------------------------------------------------------
# system.settings.read
# ---------------------------------------------------------------------------


class TestSettingsRead:
    @patch("kiara.config.load_config")
    def test_settings_read(self, mock_config):
        """Test 1: Config laden und Rueckgabe-Format pruefen."""
        mock_config.return_value = {"ollama": {"url": "http://gpu-server.local:11434"}}

        from kiara.admin.functions.system import system_settings_read

        result = _run(system_settings_read(MOCK_USER))

        assert result.success is True
        assert "config" in result.data
        assert result.data["config"]["ollama"]["url"] == "http://gpu-server.local:11434"


# ---------------------------------------------------------------------------
# system.settings.update
# ---------------------------------------------------------------------------


class TestSettingsUpdate:
    @patch("kiara.config.save_config")
    def test_settings_update(self, mock_save):
        """Test 2: Config speichern mit Confirmation."""
        from kiara.admin.functions.system import system_settings_update

        result = _run(
            system_settings_update(MOCK_USER, config={"ollama": {"url": "http://new:11434"}})
        )

        assert result.success is True
        assert result.data["message"] == "Konfiguration gespeichert"
        mock_save.assert_called_once_with({"ollama": {"url": "http://new:11434"}})


# ---------------------------------------------------------------------------
# system.vectordb.test
# ---------------------------------------------------------------------------


class TestVectorDBTest:
    @patch("qdrant_client.QdrantClient")
    def test_vectordb_test_success(self, mock_client_cls):
        """Test 3: Qdrant-Verbindung erfolgreich."""
        mock_client = MagicMock()
        mock_colls = MagicMock()
        mock_colls.collections = [MagicMock(), MagicMock()]
        mock_client.get_collections.return_value = mock_colls
        mock_client_cls.return_value = mock_client

        from kiara.admin.functions.system import system_vectordb_test

        result = _run(system_vectordb_test(MOCK_USER, qdrant_url="http://localhost:6333"))

        assert result.success is True
        assert result.data["success"] is True
        assert "2 Collection(s)" in result.data["message"]

    @patch("qdrant_client.QdrantClient")
    def test_vectordb_test_fail(self, mock_client_cls):
        """Test 4: Qdrant nicht erreichbar → AdminResult.fail."""
        mock_client_cls.side_effect = Exception("Connection refused")

        from kiara.admin.functions.system import system_vectordb_test

        result = _run(system_vectordb_test(MOCK_USER, qdrant_url="http://localhost:6333"))

        assert result.success is False
        assert "Connection refused" in result.message


# ---------------------------------------------------------------------------
# system.settings.update — Error-Handling
# ---------------------------------------------------------------------------


class TestSettingsUpdateErrors:
    @patch("kiara.config.save_config")
    def test_settings_update_validation_error(self, mock_save):
        """save_config() ValueError → AdminResult.fail mit Detail."""
        mock_save.side_effect = ValueError("ollama.url muss http(s) sein")

        from kiara.admin.functions.system import system_settings_update

        result = _run(
            system_settings_update(MOCK_USER, config={"ollama": {"url": "ftp://invalid"}})
        )

        assert result.success is False
        assert "Validierungsfehler" in result.message
        assert "ollama.url" in result.message

    @patch("kiara.config.save_config")
    def test_settings_update_permission_error(self, mock_save):
        """save_config() PermissionError → AdminResult.fail."""
        mock_save.side_effect = PermissionError("Permission denied: /etc/kiara/kiara.yaml")

        from kiara.admin.functions.system import system_settings_update

        result = _run(
            system_settings_update(MOCK_USER, config={"ollama": {"url": "http://ok:11434"}})
        )

        assert result.success is False
        assert "Fehler beim Speichern" in result.message

    def test_settings_update_invalid_type(self):
        """Nicht-dict Config → AdminResult.fail."""
        from kiara.admin.functions.system import system_settings_update

        result = _run(system_settings_update(MOCK_USER, config="das ist kein dict"))

        assert result.success is False
        assert "Ungültiges" in result.message
