"""Tests fuer Sprint 08: Backup Admin-Funktionen.

Testet backup.list, .create, .status, .restore — 8 Unit-Tests.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


# ---------------------------------------------------------------------------
# backup.list
# ---------------------------------------------------------------------------


class TestBackupList:
    @patch("kiara.config.get_backup_dir")
    @patch("kiara.config.load_config")
    def test_backup_list_empty(self, mock_config, mock_backup_dir):
        """Test 5: Leeres Backup-Verzeichnis → leere Liste."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.return_value = {}
            mock_backup_dir.return_value = tmpdir

            from kiara.admin.functions.backup import backup_list

            result = _run(backup_list(MOCK_USER))

            assert result.success is True
            assert result.data["backups"] == []

    @patch("kiara.config.get_backup_dir")
    @patch("kiara.config.load_config")
    def test_backup_list_with_files(self, mock_config, mock_backup_dir):
        """Test 6: Backups sortiert nach Datum."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.return_value = {}
            mock_backup_dir.return_value = tmpdir

            # Zwei Backup-Dateien erstellen
            import time

            (Path(tmpdir) / "backup_old.tar.gz").write_bytes(b"x" * 100)
            time.sleep(0.05)  # Sicherstellen, dass mtime unterschiedlich ist
            (Path(tmpdir) / "backup_new.tar.gz").write_bytes(b"y" * 200)

            from kiara.admin.functions.backup import backup_list

            result = _run(backup_list(MOCK_USER))

            assert result.success is True
            assert len(result.data["backups"]) == 2
            # Neueste zuerst
            assert result.data["backups"][0]["size_bytes"] == 200


# ---------------------------------------------------------------------------
# backup.create
# ---------------------------------------------------------------------------


class TestBackupCreate:
    @patch("kiara.admin.services.backup_service.start_backup")
    @patch("kiara.admin.services.backup_service.is_running", return_value=False)
    def test_backup_create(self, mock_running, mock_start):
        """Test 7: Background-Thread startet."""
        from kiara.admin.functions.backup import backup_create

        result = _run(backup_create(MOCK_USER))

        assert result.success is True
        assert "gestartet" in result.data["message"]
        mock_start.assert_called_once()

    @patch("kiara.admin.services.backup_service.is_running", return_value=True)
    def test_backup_create_concurrent_guard(self, mock_running):
        """Test 8: Zweites Backup wird abgelehnt."""
        from kiara.admin.functions.backup import backup_create

        result = _run(backup_create(MOCK_USER))

        assert result.success is False
        assert "läuft bereits" in result.message


# ---------------------------------------------------------------------------
# backup.status
# ---------------------------------------------------------------------------


class TestBackupStatus:
    @patch("kiara.admin.services.backup_service.get_status")
    def test_backup_status(self, mock_status):
        """Test 9: Status-Dict korrekt zurueckgegeben."""
        mock_status.return_value = {"running": True, "result": None, "error": None}

        from kiara.admin.functions.backup import backup_status

        result = _run(backup_status(MOCK_USER))

        assert result.success is True
        assert result.data["running"] is True


# ---------------------------------------------------------------------------
# backup.restore
# ---------------------------------------------------------------------------


class TestBackupRestore:
    @patch("kiara.admin.services.backup_service.restore_backup")
    def test_backup_restore_success(self, mock_restore):
        """Test 10: Restore mit Confirmation."""
        mock_restore.return_value = {"message": "Backup wiederhergestellt", "restored": ["db"]}

        from kiara.admin.functions.backup import backup_restore

        result = _run(backup_restore(MOCK_USER, filename="test.tar.gz"))

        assert result.success is True
        assert "wiederhergestellt" in result.message
        mock_restore.assert_called_once_with(filename="test.tar.gz", force=False)

    @patch("kiara.admin.services.backup_service.restore_backup")
    def test_backup_restore_path_traversal(self, mock_restore):
        """Test 11: ../../../etc/passwd → ValueError."""
        mock_restore.side_effect = ValueError("Ungültiger Dateiname (Path Traversal)")

        from kiara.admin.functions.backup import backup_restore

        result = _run(backup_restore(MOCK_USER, filename="../../../etc/passwd"))

        assert result.success is False
        assert "Path Traversal" in result.message

    @patch("kiara.admin.services.backup_service.restore_backup")
    def test_backup_restore_not_found(self, mock_restore):
        """Test 12: Nicht existierende Datei → Fehler."""
        mock_restore.side_effect = ValueError("Backup-Datei nicht gefunden: ghost.tar.gz")

        from kiara.admin.functions.backup import backup_restore

        result = _run(backup_restore(MOCK_USER, filename="ghost.tar.gz"))

        assert result.success is False
        assert "nicht gefunden" in result.message


# ---------------------------------------------------------------------------
# backup.status — Error-Tracking
# ---------------------------------------------------------------------------


class TestBackupStatusError:
    @patch("kiara.admin.services.backup_service.get_status")
    def test_backup_status_with_error(self, mock_status):
        """Status nach Thread-Crash → error-Feld befuellt."""
        mock_status.return_value = {
            "running": False,
            "result": None,
            "error": "Disk full: /var/lib/kiara/backups",
        }

        from kiara.admin.functions.backup import backup_status

        result = _run(backup_status(MOCK_USER))

        assert result.success is True  # Status-Abfrage selbst ist ok
        assert result.data["error"] == "Disk full: /var/lib/kiara/backups"
        assert result.data["running"] is False
        assert result.data["result"] is None

    @patch("kiara.admin.services.backup_service.get_status")
    def test_backup_status_completed(self, mock_status):
        """Status nach erfolgreichem Backup → result befuellt."""
        mock_status.return_value = {
            "running": False,
            "result": {"filename": "kiara_2026-03-28.tar.gz", "size_bytes": 245000000},
            "error": None,
        }

        from kiara.admin.functions.backup import backup_status

        result = _run(backup_status(MOCK_USER))

        assert result.data["result"]["filename"] == "kiara_2026-03-28.tar.gz"
        assert result.data["error"] is None

    @patch("kiara.admin.services.backup_service.start_backup")
    @patch("kiara.admin.services.backup_service.is_running", return_value=False)
    def test_backup_create_runtime_error(self, mock_running, mock_start):
        """start_backup() wirft RuntimeError → AdminResult.fail."""
        mock_start.side_effect = RuntimeError("Backup-Verzeichnis nicht beschreibbar")

        from kiara.admin.functions.backup import backup_create

        result = _run(backup_create(MOCK_USER))

        assert result.success is False
        assert "nicht beschreibbar" in result.message
