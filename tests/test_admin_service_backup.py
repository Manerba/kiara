"""Direkte Unit-Tests fuer backup_service.py.

Testet Service-Logik ohne Admin-Funktions-Layer:
Background-Thread-Fehlerbehandlung, Status-Tracking, Path-Traversal,
Restore-Validierung.
"""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Hilfsfunktion: Service-State zuruecksetzen
# ---------------------------------------------------------------------------


def _reset_backup_state():
    """Setzt den globalen Backup-Status zurueck."""
    from kiara.admin.services.backup_service import _backup_status

    _backup_status.update({"running": False, "result": None, "error": None})


# ---------------------------------------------------------------------------
# is_running / get_status
# ---------------------------------------------------------------------------


class TestBackupServiceState:
    def setup_method(self):
        _reset_backup_state()

    def test_initial_state(self):
        """Initialzustand: nicht laufend, kein Ergebnis."""
        from kiara.admin.services.backup_service import get_status, is_running

        assert is_running() is False
        status = get_status()
        assert status["running"] is False
        assert status["result"] is None
        assert status["error"] is None

    def test_get_status_returns_copy(self):
        """get_status() gibt eine Kopie zurueck, kein shared dict."""
        from kiara.admin.services.backup_service import _backup_status, get_status

        status = get_status()
        status["running"] = True
        assert _backup_status["running"] is False  # Original unveraendert


# ---------------------------------------------------------------------------
# list_backups
# ---------------------------------------------------------------------------


class TestBackupServiceList:
    @patch("kiara.config.get_backup_dir")
    @patch("kiara.config.load_config")
    def test_list_nonexistent_dir(self, mock_config, mock_dir):
        """Nicht-existierendes Verzeichnis → leere Liste."""
        mock_config.return_value = {}
        mock_dir.return_value = "/tmp/does_not_exist_backup_test_xyz"

        from kiara.admin.services.backup_service import list_backups

        result = list_backups()
        assert result == []

    @patch("kiara.config.get_backup_dir")
    @patch("kiara.config.load_config")
    def test_list_ignores_non_targz(self, mock_config, mock_dir):
        """Nur .tar.gz Dateien werden gelistet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.return_value = {}
            mock_dir.return_value = tmpdir

            (Path(tmpdir) / "backup.tar.gz").write_bytes(b"x" * 100)
            (Path(tmpdir) / "notes.txt").write_text("not a backup")
            (Path(tmpdir) / "data.zip").write_bytes(b"y" * 50)

            from kiara.admin.services.backup_service import list_backups

            result = list_backups()
            assert len(result) == 1
            assert result[0]["filename"] == "backup.tar.gz"


# ---------------------------------------------------------------------------
# start_backup / _run_backup — Thread-Fehlerbehandlung
# ---------------------------------------------------------------------------


class TestBackupServiceThread:
    def setup_method(self):
        _reset_backup_state()

    @patch("kiara.backup.create_backup")
    @patch("kiara.config.load_config")
    def test_thread_error_sets_error_field(self, mock_config, mock_backup):
        """Thread-Exception setzt error-Feld, running=False."""
        mock_config.return_value = {}
        mock_backup.side_effect = RuntimeError("Disk full")

        from kiara.admin.services.backup_service import get_status, start_backup

        start_backup(include_vectordb_data=False)

        # Warten bis Thread fertig (max 2s)
        for _ in range(20):
            time.sleep(0.1)
            if not get_status()["running"]:
                break

        status = get_status()
        assert status["running"] is False
        assert status["error"] == "Disk full"
        assert status["result"] is None

    @patch("kiara.backup.create_backup")
    @patch("kiara.config.load_config")
    def test_thread_success_sets_result(self, mock_config, mock_backup):
        """Erfolgreicher Backup → result mit filename + size."""
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            f.write(b"x" * 500)
            tmp_path = f.name

        try:
            mock_config.return_value = {}
            mock_backup.return_value = tmp_path

            from kiara.admin.services.backup_service import get_status, start_backup

            start_backup()

            for _ in range(20):
                time.sleep(0.1)
                if not get_status()["running"]:
                    break

            status = get_status()
            assert status["running"] is False
            assert status["error"] is None
            assert status["result"]["filename"] == os.path.basename(tmp_path)
            assert status["result"]["size_bytes"] == 500
        finally:
            os.unlink(tmp_path)

    def test_concurrent_guard(self):
        """Doppelter start_backup → RuntimeError."""
        from kiara.admin.services.backup_service import _backup_status, start_backup

        _backup_status["running"] = True

        with pytest.raises(RuntimeError, match="läuft bereits"):
            start_backup()

        _backup_status["running"] = False  # Cleanup


# ---------------------------------------------------------------------------
# restore_backup — Path-Traversal
# ---------------------------------------------------------------------------


class TestBackupServiceRestore:
    @patch("kiara.config.get_backup_dir")
    @patch("kiara.config.load_config")
    def test_path_traversal_rejected(self, mock_config, mock_dir):
        """../../../etc/passwd → ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.return_value = {}
            mock_dir.return_value = tmpdir

            from kiara.admin.services.backup_service import restore_backup

            with pytest.raises(ValueError, match="Path Traversal"):
                restore_backup(filename="../../../etc/passwd")

    @patch("kiara.config.get_backup_dir")
    @patch("kiara.config.load_config")
    def test_file_not_found(self, mock_config, mock_dir):
        """Nicht-existierende Datei → ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.return_value = {}
            mock_dir.return_value = tmpdir

            from kiara.admin.services.backup_service import restore_backup

            with pytest.raises(ValueError, match="nicht gefunden"):
                restore_backup(filename="ghost.tar.gz")

    @patch("kiara.backup.restore_backup")
    @patch("kiara.backup.validate_backup")
    @patch("kiara.config.get_backup_dir")
    @patch("kiara.config.load_config")
    def test_validation_failure_without_force(
        self, mock_config, mock_dir, mock_validate, mock_restore
    ):
        """Validierung schlaegt fehl, force=False → ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.return_value = {}
            mock_dir.return_value = tmpdir
            # Echte Datei erstellen
            (Path(tmpdir) / "bad.tar.gz").write_bytes(b"corrupt")
            mock_validate.return_value = (False, None, ["Checksum mismatch"])

            from kiara.admin.services.backup_service import restore_backup

            with pytest.raises(ValueError, match="Checksum mismatch"):
                restore_backup(filename="bad.tar.gz")

            mock_restore.assert_not_called()

    @patch("kiara.backup.restore_backup")
    @patch("kiara.backup.validate_backup")
    @patch("kiara.config.get_backup_dir")
    @patch("kiara.config.load_config")
    def test_validation_failure_with_force(
        self, mock_config, mock_dir, mock_validate, mock_restore
    ):
        """Validierung schlaegt fehl, force=True → Restore bleibt blockiert."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.return_value = {}
            mock_dir.return_value = tmpdir
            (Path(tmpdir) / "bad.tar.gz").write_bytes(b"corrupt")
            mock_validate.return_value = (False, None, ["Checksum mismatch"])
            mock_restore.return_value = {"restored": ["db"]}

            from kiara.admin.services.backup_service import restore_backup

            with pytest.raises(ValueError, match="Checksum mismatch"):
                restore_backup(filename="bad.tar.gz", force=True)

            mock_restore.assert_not_called()
