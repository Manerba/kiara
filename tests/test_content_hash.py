"""Tests fuer content-basiertes Hashing und HashStore hash_mode_meta.

Abdeckung:
- _content_file_hash: mtime-immun, content-sensitiv, size-sensitiv, leere Dateien
- compute_file_hash: Dispatcher, unbekannter Mode
- HashStore: get/set_hash_mode, migrate_hash_mode, delete_datasource cleanup

Alle Tests verwenden eindeutige datasource_ids (9200+) und raeumen
am Anfang ihre Daten auf, um Isolation in der geteilten PG-DB zu gewaehrleisten.
"""

import os

import pytest

from kiara.ingestion._file_utils import (
    HashStore,
    _content_file_hash,
    _file_hash,
    compute_file_hash,
)

# Eindeutige DS-IDs fuer dieses Test-Modul
_DS_BASE = 9200


@pytest.fixture()
def tmp_file(tmp_path):
    """Erzeugt eine temporaere Datei mit bekanntem Inhalt."""
    p = tmp_path / "test.eml"
    p.write_bytes(b"From: alice@example.com\nSubject: Test\n\nHello World")
    return str(p)


@pytest.fixture()
def hash_store(tmp_path):
    """Erzeugt einen temporaeren HashStore und raeumt Test-IDs auf."""
    hs = HashStore()
    # Aufraeumen aller IDs die in diesem Modul verwendet werden
    for ds_id in range(_DS_BASE, _DS_BASE + 20):
        hs.delete_datasource(ds_id)
    return hs


# ── _content_file_hash ───────────────────────────────────────────────────────


class TestContentFileHash:
    def test_stable_across_mtime(self, tmp_file):
        """mtime-Aenderung darf den Hash NICHT aendern."""
        h1 = _content_file_hash(tmp_file)

        # mtime um 1 Stunde aendern
        stat = os.stat(tmp_file)
        os.utime(tmp_file, (stat.st_atime, stat.st_mtime + 3600))

        h2 = _content_file_hash(tmp_file)
        assert h1 == h2

    def test_changes_on_content(self, tmp_path):
        """Inhaltsaenderung MUSS den Hash aendern."""
        p = tmp_path / "mail.eml"
        p.write_bytes(b"From: alice@example.com\nSubject: V1\n\nBody")
        h1 = _content_file_hash(str(p))

        p.write_bytes(b"From: alice@example.com\nSubject: V2\n\nBody")
        h2 = _content_file_hash(str(p))
        assert h1 != h2

    def test_changes_on_size(self, tmp_path):
        """Gleiche ersten 8KB aber andere Groesse → anderer Hash."""
        p = tmp_path / "large.eml"
        # Erstmal 8KB + extra Daten
        content = b"A" * 8192 + b"EXTRA_TAIL"
        p.write_bytes(content)
        h1 = _content_file_hash(str(p))

        # Gleiche ersten 8KB, aber andere Gesamtgroesse
        content2 = b"A" * 8192 + b"EXTRA_TAIL_LONGER"
        p.write_bytes(content2)
        h2 = _content_file_hash(str(p))
        assert h1 != h2

    def test_empty_file(self, tmp_path):
        """Leere Datei → stabiler, valider Hash."""
        p = tmp_path / "empty.eml"
        p.write_bytes(b"")
        h1 = _content_file_hash(str(p))
        assert isinstance(h1, str)
        assert len(h1) == 32  # MD5 hex

        # Erneuter Aufruf → identisch
        h2 = _content_file_hash(str(p))
        assert h1 == h2


# ── compute_file_hash (Dispatcher) ──────────────────────────────────────────


class TestComputeFileHash:
    def test_dispatcher_different_modes(self, tmp_file):
        """mtime und content Mode ergeben unterschiedliche Hashes."""
        h_mtime = compute_file_hash(tmp_file, "mtime")
        h_content = compute_file_hash(tmp_file, "content")
        assert h_mtime != h_content

    def test_mtime_default(self, tmp_file):
        """Default-Mode ist 'mtime'."""
        h_default = compute_file_hash(tmp_file)
        h_mtime = _file_hash(tmp_file)
        assert h_default == h_mtime

    def test_unknown_mode_raises(self, tmp_file):
        """Unbekannter Mode → ValueError."""
        with pytest.raises(ValueError, match="Unbekannter hash_mode"):
            compute_file_hash(tmp_file, "unknown")

    def test_typo_mode_raises(self, tmp_file):
        """Tippfehler im Mode → ValueError."""
        with pytest.raises(ValueError, match="Unbekannter hash_mode"):
            compute_file_hash(tmp_file, "conten")


# ── HashStore: hash_mode_meta ────────────────────────────────────────────────


class TestHashModeMetadata:
    def test_roundtrip(self, hash_store):
        """set_hash_mode + get_hash_mode Roundtrip."""
        ds_id = _DS_BASE
        hash_store.set_hash_mode(ds_id, "nomic-embed-text", "content")
        assert hash_store.get_hash_mode(ds_id, "nomic-embed-text") == "content"

    def test_default(self, hash_store):
        """get_hash_mode ohne vorherigen Set → 'mtime'."""
        assert hash_store.get_hash_mode(_DS_BASE + 19, "some-model") == "mtime"

    def test_overwrite(self, hash_store):
        """set_hash_mode ueberschreibt vorherigen Wert."""
        ds_id = _DS_BASE + 1
        hash_store.set_hash_mode(ds_id, "model", "content")
        hash_store.set_hash_mode(ds_id, "model", "mtime")
        assert hash_store.get_hash_mode(ds_id, "model") == "mtime"

    def test_different_models_independent(self, hash_store):
        """Verschiedene Modelle haben unabhaengige hash_modes."""
        ds_id = _DS_BASE + 2
        hash_store.set_hash_mode(ds_id, "model-a", "content")
        hash_store.set_hash_mode(ds_id, "model-b", "mtime")
        assert hash_store.get_hash_mode(ds_id, "model-a") == "content"
        assert hash_store.get_hash_mode(ds_id, "model-b") == "mtime"


# ── HashStore: migrate_hash_mode ─────────────────────────────────────────────


class TestMigrateHashMode:
    def test_migrate_mtime_to_content(self, hash_store, tmp_path):
        """Hashes mit mtime speichern, auf content migrieren → neue Hashes."""
        ds_id = _DS_BASE + 3
        # Dateien anlegen
        files = {}
        for i in range(3):
            p = tmp_path / f"mail_{i}.eml"
            p.write_bytes(f"From: user{i}@test.de\nBody {i}".encode())
            fp = str(p)
            files[fp] = (_file_hash(fp), os.path.getsize(fp))

        hash_store.upsert_many(ds_id, "nomic", files)
        hash_store.set_hash_mode(ds_id, "nomic", "mtime")

        count = hash_store.migrate_hash_mode(ds_id, "nomic", "content")
        assert count == 3

        # Mode aktualisiert
        assert hash_store.get_hash_mode(ds_id, "nomic") == "content"

        # Hashes sind jetzt content-basiert
        stored = hash_store.load(ds_id, "nomic")
        for fp in files:
            expected = compute_file_hash(fp, "content")
            assert stored[fp] == expected

        # file_size muss nach Migration gesetzt sein
        from kiara.shared.db import db_query

        with db_query() as conn:
            rows = conn.execute(
                "SELECT file_path, file_size FROM file_hashes "
                "WHERE datasource_id = %s AND embedding_model = %s",
                (ds_id, "nomic"),
            ).fetchall()
        for row in rows:
            assert row["file_size"] is not None, f"file_size NULL fuer {row['file_path']}"
            assert row["file_size"] > 0

    def test_migrate_missing_files(self, hash_store, tmp_path):
        """Migration mit geloeschten Dateien → entfernt, kein Fehler."""
        ds_id = _DS_BASE + 4
        p = tmp_path / "exists.eml"
        p.write_bytes(b"content")
        fp_exists = str(p)

        fp_missing = str(tmp_path / "gone.eml")

        hashes = {
            fp_exists: (_file_hash(fp_exists), os.path.getsize(fp_exists)),
            fp_missing: ("fakehash123", 100),
        }
        hash_store.upsert_many(ds_id, "nomic", hashes)

        count = hash_store.migrate_hash_mode(ds_id, "nomic", "content")
        assert count == 1  # Nur die existierende Datei

        stored = hash_store.load(ds_id, "nomic")
        assert fp_exists in stored
        assert fp_missing not in stored
        assert hash_store.get_hash_mode(ds_id, "nomic") == "content"

    def test_migrate_empty_store(self, hash_store):
        """Migration bei leerem Store → 0 migriert, Mode gesetzt."""
        ds_id = _DS_BASE + 5
        count = hash_store.migrate_hash_mode(ds_id, "model", "content")
        assert count == 0
        assert hash_store.get_hash_mode(ds_id, "model") == "content"

    def test_migrate_idempotent(self, hash_store, tmp_path):
        """Zweimaliges migrate auf gleichen Mode → identische Ergebnisse."""
        ds_id = _DS_BASE + 6
        p = tmp_path / "test.eml"
        p.write_bytes(b"Hello")
        fp = str(p)

        hash_store.upsert_many(ds_id, "m", {fp: (_file_hash(fp), os.path.getsize(fp))})
        hash_store.migrate_hash_mode(ds_id, "m", "content")
        stored1 = hash_store.load(ds_id, "m")

        hash_store.migrate_hash_mode(ds_id, "m", "content")
        stored2 = hash_store.load(ds_id, "m")

        assert stored1 == stored2


# ── HashStore: delete_datasource cleanup ─────────────────────────────────────


class TestDeleteDatasourceCleanup:
    def test_cleans_hash_mode_meta(self, hash_store, tmp_path):
        """delete_datasource entfernt auch hash_mode_meta-Eintraege."""
        ds_id = _DS_BASE + 7
        p = tmp_path / "test.eml"
        p.write_bytes(b"test")
        fp = str(p)

        hash_store.upsert_many(ds_id, "nomic", {fp: ("hash1", 100)})
        hash_store.set_hash_mode(ds_id, "nomic", "content")

        hash_store.delete_datasource(ds_id)

        assert hash_store.get_hash_mode(ds_id, "nomic") == "mtime"  # Default
        assert hash_store.load(ds_id, "nomic") == {}

    def test_cleans_specific_model(self, hash_store):
        """delete_datasource mit embedding_model entfernt nur dieses Modell."""
        ds_id = _DS_BASE + 8
        hash_store.set_hash_mode(ds_id, "model-a", "content")
        hash_store.set_hash_mode(ds_id, "model-b", "content")

        hash_store.delete_datasource(ds_id, embedding_model="model-a")

        assert hash_store.get_hash_mode(ds_id, "model-a") == "mtime"  # geloescht → Default
        assert hash_store.get_hash_mode(ds_id, "model-b") == "content"  # unberuehrt

    def test_cleans_all_models(self, hash_store):
        """delete_datasource ohne embedding_model entfernt alle Modelle."""
        ds_id = _DS_BASE + 9
        hash_store.set_hash_mode(ds_id, "model-a", "content")
        hash_store.set_hash_mode(ds_id, "model-b", "content")

        hash_store.delete_datasource(ds_id)

        assert hash_store.get_hash_mode(ds_id, "model-a") == "mtime"
        assert hash_store.get_hash_mode(ds_id, "model-b") == "mtime"
