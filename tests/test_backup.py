"""Tests fuer kiara.backup."""

import hashlib
import json
import os
import tarfile
import time

import pytest

from kiara.backup import (
    BackupError,
    _cleanup_old_backups,
    _sha256,
    create_backup,
    restore_backup,
    validate_backup,
)


@pytest.fixture
def mock_pg_tools(monkeypatch):
    """Mockt pg_dump/pg_restore/pg_tool-Checks — keine echte DB noetig."""

    def _fake_run_pg_dump(dsn_parts, output_path):
        # Eine Dummy-Datei erzeugen, damit _sha256 und tar.add nicht scheitern
        with open(output_path, "wb") as f:
            f.write(b"FAKE_PG_DUMP_CONTENT")

    def _fake_run_pg_restore(dsn_parts, dump_path):
        assert os.path.exists(dump_path)

    monkeypatch.setattr("kiara.backup._check_pg_tool", lambda _b: None)
    monkeypatch.setattr("kiara.backup._run_pg_dump", _fake_run_pg_dump)
    monkeypatch.setattr("kiara.backup._run_pg_restore", _fake_run_pg_restore)


def _make_config(tmp_path):
    """Erzeugt eine Test-Config mit tmp_path-basierten Pfaden."""
    vectordb_data_path = str(tmp_path / "vectordb_data")
    backup_dir = str(tmp_path / "backups")
    config_path = str(tmp_path / "kiara.yaml")
    template_dir = str(tmp_path / "templates")

    return {
        "vectordb": {"backend": "qdrant", "path": vectordb_data_path},
        "backup": {"dir": backup_dir, "max_backups": 5},
        "workflows": {"template_dir": template_dir},
        "database": {"postgresql": {"dsn": "postgresql://kiara:kiara@localhost:5432/kiara_test"}},
    }, {
        "vectordb_data_path": vectordb_data_path,
        "backup_dir": backup_dir,
        "config_path": config_path,
        "template_dir": template_dir,
    }


def _setup_test_data(paths):
    """Erstellt Test-Dateien fuer Backup."""
    # Config file
    with open(paths["config_path"], "w") as f:
        f.write("ollama:\n  url: http://localhost:11434\n")

    # VectorDB data dir (simuliert)
    os.makedirs(paths["vectordb_data_path"], exist_ok=True)
    with open(os.path.join(paths["vectordb_data_path"], "chroma.sqlite3"), "w") as f:
        f.write("fake vectordb data")

    # Templates
    os.makedirs(paths["template_dir"], exist_ok=True)
    with open(os.path.join(paths["template_dir"], "default.html"), "w") as f:
        f.write("<html>template</html>")


class TestBackup:
    def test_create_backup(self, tmp_path, monkeypatch, mock_pg_tools):
        """create_backup() erstellt eine gueltige tar.gz Datei."""
        config, paths = _make_config(tmp_path)
        _setup_test_data(paths)
        monkeypatch.setenv("KIARA_CONFIG", paths["config_path"])

        output = str(tmp_path / "test-backup.tar.gz")
        result = create_backup(config, output_path=output, include_vectordb_data=False)

        assert os.path.exists(result)
        assert result == output

        # Pruefen dass tar.gz gueltig ist und DB-Dump enthaelt
        with tarfile.open(result, "r:gz") as tar:
            names = tar.getnames()
            assert "kiara.yaml" in names
            assert "manifest.json" in names
            assert "database.dump" in names

    def test_validate_backup_valid(self, tmp_path, monkeypatch, mock_pg_tools):
        """validate_backup() gibt True fuer gueltiges Backup."""
        config, paths = _make_config(tmp_path)
        _setup_test_data(paths)
        monkeypatch.setenv("KIARA_CONFIG", paths["config_path"])

        output = str(tmp_path / "test-backup.tar.gz")
        create_backup(config, output_path=output, include_vectordb_data=False)

        is_valid, manifest, issues = validate_backup(output)
        assert is_valid is True
        assert len(issues) == 0
        assert "created_at" in manifest
        assert manifest["database_included"] is True

    def test_validate_backup_missing_file(self, tmp_path):
        """validate_backup() gibt False fuer nicht existierende Datei."""
        is_valid, manifest, issues = validate_backup(str(tmp_path / "nonexistent.tar.gz"))
        assert is_valid is False
        assert len(issues) > 0
        assert "nicht gefunden" in issues[0]

    def test_restore_dry_run(self, tmp_path, monkeypatch, mock_pg_tools):
        """restore_backup(dry_run=True) stellt nichts her, listet aber Inhalt."""
        config, paths = _make_config(tmp_path)
        _setup_test_data(paths)
        monkeypatch.setenv("KIARA_CONFIG", paths["config_path"])

        output = str(tmp_path / "test-backup.tar.gz")
        create_backup(config, output_path=output, include_vectordb_data=False)

        # Restore-Ziel in anderem tmp dir
        restore_dir = tmp_path / "restore_target"
        restore_dir.mkdir()
        restore_config = {
            "vectordb": {"path": str(restore_dir / "vectordb_data")},
            "workflows": {"template_dir": str(restore_dir / "templates")},
            "database": {
                "postgresql": {"dsn": "postgresql://kiara:kiara@localhost:5432/kiara_test"}
            },
        }

        result = restore_backup(output, restore_config, dry_run=True)
        assert result["success"] is True
        assert result["dry_run"] is True
        assert "would_restore" in result

    def test_restore_actual(self, tmp_path, monkeypatch, mock_pg_tools):
        """restore_backup() stellt Config korrekt wieder her."""
        config, paths = _make_config(tmp_path)
        _setup_test_data(paths)
        monkeypatch.setenv("KIARA_CONFIG", paths["config_path"])

        output = str(tmp_path / "test-backup.tar.gz")
        create_backup(config, output_path=output, include_vectordb_data=False)

        # Restore in anderes Verzeichnis
        restore_dir = tmp_path / "restore_target"
        restore_dir.mkdir()
        restore_config_path = str(restore_dir / "kiara.yaml")
        monkeypatch.setenv("KIARA_CONFIG", restore_config_path)

        restore_config = {
            "vectordb": {"path": str(restore_dir / "vectordb_data")},
            "workflows": {"template_dir": str(restore_dir / "templates")},
            "database": {
                "postgresql": {"dsn": "postgresql://kiara:kiara@localhost:5432/kiara_test"}
            },
        }

        result = restore_backup(output, restore_config, force=True)
        assert result["success"] is True
        assert result["dry_run"] is False
        assert "kiara.yaml" in result["restored"]
        assert "database.dump" in result["restored"]

    def test_restore_without_force_does_not_restore_database(
        self, tmp_path, monkeypatch, mock_pg_tools
    ):
        """restore_backup(force=False) darf die bestehende PostgreSQL-DB nicht ueberschreiben."""
        config, paths = _make_config(tmp_path)
        _setup_test_data(paths)
        monkeypatch.setenv("KIARA_CONFIG", paths["config_path"])

        output = str(tmp_path / "test-backup.tar.gz")
        create_backup(config, output_path=output, include_vectordb_data=False)

        restore_calls = []
        monkeypatch.setattr(
            "kiara.backup._run_pg_restore", lambda *args: restore_calls.append(args)
        )

        restore_config = {
            "vectordb": {"backend": "qdrant", "path": str(tmp_path / "restore_vdb")},
            "workflows": {"template_dir": str(tmp_path / "restore_templates")},
            "database": {
                "postgresql": {"dsn": "postgresql://kiara:kiara@localhost:5432/kiara_test"}
            },
        }
        result = restore_backup(output, restore_config, force=False)

        assert result["success"] is False
        assert "ueberschreiben" in result["message"]
        assert any("database.dump" in issue for issue in result["issues"])
        assert restore_calls == []

    def test_cleanup_old_backups(self, tmp_path):
        """_cleanup_old_backups() loescht die aeltesten Dateien."""
        backup_dir = str(tmp_path / "backups")
        os.makedirs(backup_dir)

        # Erstelle 7 Fake-Backups mit unterschiedlichen Timestamps
        for i in range(7):
            path = os.path.join(backup_dir, f"kiara-backup-2024010{i}-120000.tar.gz")
            with open(path, "w") as f:
                f.write(f"backup {i}")
            # Setze unterschiedliche mtime
            mtime = time.time() - (7 - i) * 100
            os.utime(path, (mtime, mtime))

        _cleanup_old_backups(backup_dir, max_backups=3)

        remaining = list((tmp_path / "backups").glob("kiara-backup-*.tar.gz"))
        assert len(remaining) == 3

    def test_validate_backup_corrupted(self, tmp_path):
        """validate_backup() gibt False fuer beschaedigte Datei."""
        corrupt_path = str(tmp_path / "corrupt.tar.gz")
        with open(corrupt_path, "wb") as f:
            f.write(b"this is not a tar file")

        is_valid, manifest, issues = validate_backup(corrupt_path)
        assert is_valid is False
        assert len(issues) > 0
        assert "beschaedigt" in issues[0]

    def test_create_backup_without_vectordb_data(self, tmp_path, monkeypatch, mock_pg_tools):
        """create_backup(include_vectordb_data=False) erstellt Backup ohne VectorDB-Daten."""
        config, paths = _make_config(tmp_path)
        _setup_test_data(paths)
        monkeypatch.setenv("KIARA_CONFIG", paths["config_path"])

        output = str(tmp_path / "no-vectordb-data-backup.tar.gz")
        result = create_backup(config, output_path=output, include_vectordb_data=False)

        with tarfile.open(result, "r:gz") as tar:
            names = tar.getnames()
            assert "kiara.yaml" in names
            # vectordb_data dir should not be present
            assert not any(n.startswith("vectordb_data") for n in names)

            manifest_file = tar.extractfile("manifest.json")
            manifest = json.load(manifest_file)
            assert manifest["include_vectordb_data"] is False

    def test_create_backup_qdrant_vectordb_data_fails_closed(
        self, tmp_path, monkeypatch, mock_pg_tools
    ):
        """Qdrant darf nicht als alter ChromaDB-Dateipfad ins Backup kopiert werden."""
        config, paths = _make_config(tmp_path)
        _setup_test_data(paths)
        monkeypatch.setenv("KIARA_CONFIG", paths["config_path"])

        with pytest.raises(BackupError, match="Qdrant"):
            create_backup(
                config,
                output_path=str(tmp_path / "qdrant-vdb-backup.tar.gz"),
                include_vectordb_data=True,
            )

    def test_create_backup_file_vectordb_data_explicit_backend(
        self, tmp_path, monkeypatch, mock_pg_tools
    ):
        """File-backed VectorDB-Daten werden nur bei explizit nicht-Qdrant Backend gesichert."""
        config, paths = _make_config(tmp_path)
        config["vectordb"]["backend"] = "file"
        _setup_test_data(paths)
        monkeypatch.setenv("KIARA_CONFIG", paths["config_path"])

        output = str(tmp_path / "file-vectordb-data-backup.tar.gz")
        result = create_backup(config, output_path=output, include_vectordb_data=True)

        with tarfile.open(result, "r:gz") as tar:
            names = tar.getnames()
            assert any(n.startswith("vectordb_data/") for n in names)

    def test_restore_force_does_not_bypass_checksum_validation(self, tmp_path, monkeypatch):
        """force=True darf SHA256-/Manifest-Fehler nicht uebergehen."""
        backup_path = tmp_path / "bad-checksum.tar.gz"
        work = tmp_path / "bad"
        work.mkdir()
        dump = work / "database.dump"
        dump.write_bytes(b"tampered dump")
        manifest = {
            "database_included": True,
            "files": {"database.dump": hashlib.sha256(b"expected dump").hexdigest()},
        }
        (work / "manifest.json").write_text(json.dumps(manifest))
        with tarfile.open(backup_path, "w:gz") as tar:
            tar.add(dump, arcname="database.dump")
            tar.add(work / "manifest.json", arcname="manifest.json")

        restore_calls = []
        monkeypatch.setattr(
            "kiara.backup._run_pg_restore", lambda *args: restore_calls.append(args)
        )
        result = restore_backup(
            str(backup_path),
            {
                "vectordb": {"backend": "qdrant", "path": str(tmp_path / "restore_vdb")},
                "workflows": {"template_dir": str(tmp_path / "restore_templates")},
                "database": {
                    "postgresql": {"dsn": "postgresql://kiara:kiara@localhost:5432/kiara_test"}
                },
            },
            force=True,
        )

        assert result["success"] is False
        assert "Validierung" in result["message"]
        assert any("Checksumme" in issue for issue in result["issues"])
        assert restore_calls == []

    def test_create_backup_without_dsn_fails(self, tmp_path, monkeypatch):
        """create_backup() ohne DSN schlaegt fehl (fail-closed, kein DB-Dump moeglich)."""
        config, paths = _make_config(tmp_path)
        _setup_test_data(paths)
        del config["database"]  # DSN entfernen
        monkeypatch.setenv("KIARA_CONFIG", paths["config_path"])

        with pytest.raises(BackupError, match="database.postgresql.dsn"):
            create_backup(config, output_path=str(tmp_path / "x.tar.gz"))

    def test_create_backup_without_pg_dump_fails(self, tmp_path, monkeypatch):
        """create_backup() ohne pg_dump im PATH schlaegt fehl."""
        config, paths = _make_config(tmp_path)
        _setup_test_data(paths)
        monkeypatch.setenv("KIARA_CONFIG", paths["config_path"])
        monkeypatch.setattr("shutil.which", lambda _b: None)

        with pytest.raises(BackupError, match="pg_dump nicht gefunden"):
            create_backup(config, output_path=str(tmp_path / "x.tar.gz"))

    def test_sha256_consistency(self, tmp_path):
        """_sha256() liefert konsistente Hashes."""
        filepath = str(tmp_path / "testfile.txt")
        with open(filepath, "w") as f:
            f.write("test content")

        hash1 = _sha256(filepath)
        hash2 = _sha256(filepath)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest length
