"""Tests fuer Build 69: Ingestion Resume-Mechanismus.

Testet:
- Hash-Checkpoint: periodisches Schreiben nach _HASH_CHECKPOINT_INTERVAL Dateien
- CancelledError: sauberer Abbruch + Checkpoint bei Cancel
- _recover_stale_jobs(): running→error, Auto-Resume mit resumed_from_job
- Duplikat-Schutz: kein zweiter pending-Job bei bestehendem pending
- IMAP Scan-Cache: .scan_done Marker erstellen, TTL-Pruefung, TTL-Ablauf

Alle Tests laufen gegen die geteilte PG-Datenbank (kiara_test).
Datasource-Rows werden per RETURNING id eingefuegt, um FK-Konflikte zu vermeiden.
"""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import kiara.ingestion.runtime as runtime_mod
from kiara.shared.db import get_connection

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def job_db():
    """PostgreSQL-DB: saubere Testdaten mit korrekten DS-IDs."""
    conn = get_connection()
    # Schema sicherstellen (andere Tests koennen datasources droppen+recreaten)
    try:
        conn.execute("SELECT ingestion_status FROM datasources LIMIT 0")
    except Exception:
        conn.rollback()
        # Tabelle hat falsches Schema — Spalte nachtraeglich hinzufuegen
        try:
            conn.execute(
                "ALTER TABLE datasources ADD COLUMN IF NOT EXISTS "
                "ingestion_status TEXT DEFAULT 'idle'"
            )
            conn.commit()
        except Exception:
            conn.rollback()
            # Fallback: komplett neu erstellen
            conn.execute("DROP TABLE IF EXISTS ingestion_file_logs CASCADE")
            conn.execute("DROP TABLE IF EXISTS ingestion_jobs CASCADE")
            conn.execute("DROP TABLE IF EXISTS datasource_embeddings CASCADE")
            conn.execute("DROP TABLE IF EXISTS datasources CASCADE")
            from kiara.web.schema import create_tables

            create_tables(conn)
            conn.execute(
                "ALTER TABLE datasources ADD COLUMN IF NOT EXISTS "
                "ingestion_status TEXT DEFAULT 'idle'"
            )
            conn.commit()
    # Bestehende Testdaten aufraeumen
    conn.execute(
        "DELETE FROM ingestion_jobs WHERE datasource_id IN "
        "(SELECT id FROM datasources WHERE name IN ('ResumeTest DS 1', 'ResumeTest DS 2'))"
    )
    conn.execute(
        "DELETE FROM datasource_embeddings WHERE datasource_id IN "
        "(SELECT id FROM datasources WHERE name IN ('ResumeTest DS 1', 'ResumeTest DS 2'))"
    )
    conn.execute("DELETE FROM datasources WHERE name IN ('ResumeTest DS 1', 'ResumeTest DS 2')")
    conn.commit()

    cur1 = conn.execute(
        "INSERT INTO datasources (name, type, path, is_active) "
        "VALUES ('ResumeTest DS 1', 'local', '/tmp/test1', true) RETURNING id"
    )
    ds1_id = cur1.fetchone()["id"]

    cur2 = conn.execute(
        "INSERT INTO datasources (name, type, path, is_active) "
        "VALUES ('ResumeTest DS 2', 'local', '/tmp/test2', true) RETURNING id"
    )
    ds2_id = cur2.fetchone()["id"]
    conn.commit()
    conn.close()
    return {"ds1_id": ds1_id, "ds2_id": ds2_id}


@pytest.fixture
def mock_db(job_db, monkeypatch):
    """Patcht db_query/db_transaction auf PostgreSQL-Verbindungen."""
    import kiara.shared.db as shared_db
    import kiara.web.db_utils as db_utils

    @contextmanager
    def fake_db_query():
        conn = get_connection()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def fake_db_transaction():
        conn = get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    monkeypatch.setattr(db_utils, "db_query", fake_db_query)
    monkeypatch.setattr(db_utils, "db_transaction", fake_db_transaction)
    monkeypatch.setattr(shared_db, "db_query", fake_db_query)
    monkeypatch.setattr(shared_db, "db_transaction", fake_db_transaction)
    import kiara.web.ingestion_worker as worker_mod

    monkeypatch.setattr(worker_mod, "db_query", fake_db_query)
    monkeypatch.setattr(worker_mod, "db_transaction", fake_db_transaction)

    return job_db


@pytest.fixture
def ingestion_config(tmp_path):
    """Minimale Config fuer run_ingestion Tests."""
    ds_path = tmp_path / "datasource"
    ds_path.mkdir()
    vdb_path = tmp_path / "vectordb"
    vdb_path.mkdir()
    return {
        "datasource": {
            "path": str(ds_path),
            "file_types": [".txt"],
            "exclude_patterns": [],
        },
        "ingestion": {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "batch_size": 10,
        },
        "vectordb": {
            "path": str(vdb_path),
        },
        "ollama": {
            "url": "http://localhost:11434",
            "model": "test-model",
        },
    }


# ---------------------------------------------------------------------------
# Hash-Checkpoint Tests
# ---------------------------------------------------------------------------


class TestHashCheckpoint:
    """Hash-Checkpoint wird nach _HASH_CHECKPOINT_INTERVAL Dateien geschrieben."""

    def test_checkpoint_after_n_files(self, ingestion_config, tmp_path):
        """_save_hashes wird periodisch nach jedem N-ten File aufgerufen."""
        from kiara.shared.constants import HASH_CHECKPOINT_INTERVAL as _HASH_CHECKPOINT_INTERVAL

        ds_path = Path(ingestion_config["datasource"]["path"])

        # Genau _HASH_CHECKPOINT_INTERVAL + 5 Dateien erstellen
        num_files = _HASH_CHECKPOINT_INTERVAL + 5
        for i in range(num_files):
            (ds_path / f"file_{i:03d}.txt").write_text(f"Inhalt {i}")

        mock_store = MagicMock()
        mock_store.add = MagicMock()
        mock_store.delete_by_sources = MagicMock()

        mock_hs = MagicMock()
        mock_hs.load.return_value = {}

        import kiara.ingestion.pipeline as pipeline_mod

        mock_chunk_ds = MagicMock()
        mock_chunk_ds.chunks_exist.return_value = set()
        mock_chunk_ds.get_datasource_ids_for_chunks.return_value = {}

        with (
            patch.object(pipeline_mod, "VectorStore", return_value=mock_store),
            patch.object(runtime_mod, "load_file") as mock_load,
            patch.object(runtime_mod, "chunk_document") as mock_chunk,
            patch.object(pipeline_mod, "HashStore", return_value=mock_hs),
            patch.object(pipeline_mod, "ChunkDatasources", return_value=mock_chunk_ds),
            patch.object(
                runtime_mod,
                "_run_graph_extraction",
                return_value={"entities": 0, "relations": 0, "errors": 0},
            ),
        ):
            mock_load.side_effect = lambda fp, **kw: {
                "text": f"text for {fp}",
                "metadata": {
                    "source": fp,
                    "chunk_index": 0,
                },
            }
            mock_chunk.side_effect = lambda text, meta, **kw: [
                {
                    "text": text,
                    "metadata": {**meta, "chunk_index": 0},
                }
            ]

            from kiara.ingestion.pipeline import run_ingestion

            stats = run_ingestion(
                ingestion_config,
                incremental=True,
                datasource_id=1,
                embedding_model="test-model",
            )

        assert stats["processed_files"] == num_files

        # Checkpoint nach genau _HASH_CHECKPOINT_INTERVAL + finaler Save
        # upsert_many wird bei Checkpoint (processed % interval == 0) + am Ende aufgerufen
        checkpoint_calls = mock_hs.upsert_many.call_count
        # Mindestens 2: ein Checkpoint bei File 50 + finaler Save
        assert checkpoint_calls >= 2, (
            f"Erwartet mindestens 2 upsert_many-Aufrufe "
            f"(Checkpoint + Final), bekommen: {checkpoint_calls}"
        )

    def test_no_checkpoint_before_interval(self, ingestion_config, tmp_path):
        """Vor _HASH_CHECKPOINT_INTERVAL wird kein Checkpoint geschrieben (nur final)."""
        from kiara.shared.constants import HASH_CHECKPOINT_INTERVAL as _HASH_CHECKPOINT_INTERVAL

        ds_path = Path(ingestion_config["datasource"]["path"])

        # Weniger als ein Checkpoint-Intervall
        num_files = _HASH_CHECKPOINT_INTERVAL - 1
        for i in range(num_files):
            (ds_path / f"file_{i:03d}.txt").write_text(f"Inhalt {i}")

        mock_store = MagicMock()
        mock_hs = MagicMock()
        mock_hs.load.return_value = {}

        import kiara.ingestion.pipeline as pipeline_mod

        mock_chunk_ds = MagicMock()
        mock_chunk_ds.chunks_exist.return_value = set()
        mock_chunk_ds.get_datasource_ids_for_chunks.return_value = {}

        with (
            patch.object(pipeline_mod, "VectorStore", return_value=mock_store),
            patch.object(runtime_mod, "load_file") as mock_load,
            patch.object(runtime_mod, "chunk_document") as mock_chunk,
            patch.object(pipeline_mod, "HashStore", return_value=mock_hs),
            patch.object(pipeline_mod, "ChunkDatasources", return_value=mock_chunk_ds),
            patch.object(
                runtime_mod,
                "_run_graph_extraction",
                return_value={"entities": 0, "relations": 0, "errors": 0},
            ),
        ):
            mock_load.side_effect = lambda fp, **kw: {
                "text": f"text for {fp}",
                "metadata": {"source": fp, "chunk_index": 0},
            }
            mock_chunk.side_effect = lambda text, meta, **kw: [
                {
                    "text": text,
                    "metadata": {**meta, "chunk_index": 0},
                }
            ]

            from kiara.ingestion.pipeline import run_ingestion

            run_ingestion(
                ingestion_config, incremental=True, datasource_id=1, embedding_model="test-model"
            )

        # Nur 1x am Ende (kein Checkpoint, da processed < interval)
        assert mock_hs.upsert_many.call_count == 1


# ---------------------------------------------------------------------------
# CancelledError Tests
# ---------------------------------------------------------------------------


class TestCancelledError:
    """CancelledError bricht Verarbeitung sauber ab und speichert Hashes."""

    def test_cancelled_error_saves_checkpoint(self, ingestion_config, tmp_path):
        """Bei _CancelledError werden die bisherigen Hashes gespeichert."""
        ds_path = Path(ingestion_config["datasource"]["path"])
        for i in range(5):
            (ds_path / f"file_{i}.txt").write_text(f"Inhalt {i}")

        mock_store = MagicMock()

        call_count = {"n": 0}

        def control_callback():
            call_count["n"] += 1
            # Ab dem 4. Aufruf Cancel (nach 2 Dateien verarbeitet,
            # da control_callback 1x pro Datei-Loop + 1x pro Embedding-Batch)
            if call_count["n"] >= 4:
                return "cancel"
            return None

        mock_hs = MagicMock()
        mock_hs.load.return_value = {}

        import kiara.ingestion.pipeline as pipeline_mod

        mock_chunk_ds = MagicMock()
        mock_chunk_ds.chunks_exist.return_value = set()
        mock_chunk_ds.get_datasource_ids_for_chunks.return_value = {}

        with (
            patch.object(pipeline_mod, "VectorStore", return_value=mock_store),
            patch.object(runtime_mod, "load_file") as mock_load,
            patch.object(runtime_mod, "chunk_document") as mock_chunk,
            patch.object(pipeline_mod, "HashStore", return_value=mock_hs),
            patch.object(pipeline_mod, "ChunkDatasources", return_value=mock_chunk_ds),
            patch.object(
                runtime_mod,
                "_run_graph_extraction",
                return_value={"entities": 0, "relations": 0, "errors": 0},
            ),
        ):
            mock_load.side_effect = lambda fp, **kw: {
                "text": f"text for {fp}",
                "metadata": {"source": fp, "chunk_index": 0},
            }
            mock_chunk.side_effect = lambda text, meta, **kw: [
                {
                    "text": text,
                    "metadata": {**meta, "chunk_index": 0},
                }
            ]

            from kiara.ingestion.pipeline import run_ingestion

            stats = run_ingestion(
                ingestion_config,
                incremental=True,
                datasource_id=1,
                embedding_model="test-model",
                control_callback=control_callback,
            )

        # Weniger als alle Dateien verarbeitet (Cancel hat gewirkt)
        assert stats["processed_files"] < 5

        # Hashes wurden gespeichert (upsert_many bei CancelledError oder am Ende)
        assert mock_hs.upsert_many.call_count >= 1

    def test_cancelled_error_during_embedding_saves_hashes(self, ingestion_config, tmp_path):
        """_CancelledError waehrend _write_embeddings speichert bisherige Hashes."""
        from kiara.ingestion.pipeline import _CancelledError

        ds_path = Path(ingestion_config["datasource"]["path"])
        for i in range(3):
            (ds_path / f"file_{i}.txt").write_text(f"Inhalt {i}")

        mock_store = MagicMock()
        embed_call_count = {"n": 0}

        def store_add_side_effect(*args, **kwargs):
            embed_call_count["n"] += 1
            if embed_call_count["n"] >= 2:
                raise _CancelledError()

        mock_store.add.side_effect = store_add_side_effect
        mock_hs = MagicMock()
        mock_hs.load.return_value = {}

        import kiara.ingestion.pipeline as pipeline_mod

        mock_chunk_ds = MagicMock()
        mock_chunk_ds.chunks_exist.return_value = set()
        mock_chunk_ds.get_datasource_ids_for_chunks.return_value = {}

        with (
            patch.object(pipeline_mod, "VectorStore", return_value=mock_store),
            patch.object(runtime_mod, "load_file") as mock_load,
            patch.object(runtime_mod, "chunk_document") as mock_chunk,
            patch.object(pipeline_mod, "HashStore", return_value=mock_hs),
            patch.object(pipeline_mod, "ChunkDatasources", return_value=mock_chunk_ds),
            patch.object(
                runtime_mod,
                "_run_graph_extraction",
                return_value={"entities": 0, "relations": 0, "errors": 0},
            ),
        ):
            mock_load.side_effect = lambda fp, **kw: {
                "text": f"text for {fp}",
                "metadata": {"source": fp, "chunk_index": 0},
            }
            mock_chunk.side_effect = lambda text, meta, **kw: [
                {
                    "text": text,
                    "metadata": {**meta, "chunk_index": 0},
                }
            ]

            from kiara.ingestion.pipeline import run_ingestion

            stats = run_ingestion(
                ingestion_config,
                incremental=True,
                datasource_id=1,
                embedding_model="test-model",
            )

        # upsert_many muss aufgerufen worden sein (Checkpoint bei CancelledError + final)
        assert mock_hs.upsert_many.call_count >= 1


# ---------------------------------------------------------------------------
# _recover_stale_jobs Tests
# ---------------------------------------------------------------------------


class TestRecoverStaleJobs:
    """_recover_stale_jobs() beim Worker-Start."""

    def test_marks_running_jobs_as_error(self, mock_db):
        """running-Jobs werden als 'error' markiert."""
        ds_id = mock_db["ds1_id"]
        conn = get_connection()
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, embedding_model) "
            "VALUES (%s, 'running', 'nomic')",
            (ds_id,),
        )
        conn.execute(
            "UPDATE datasources SET ingestion_status = 'running' WHERE id = %s",
            (ds_id,),
        )
        conn.commit()
        conn.close()

        from kiara.web.ingestion_worker import IngestionWorker

        worker = IngestionWorker()
        worker._recover_stale_jobs()

        conn = get_connection()
        job = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE datasource_id = %s AND status = 'error'",
            (ds_id,),
        ).fetchone()
        ds = conn.execute("SELECT * FROM datasources WHERE id = %s", (ds_id,)).fetchone()
        conn.close()

        assert job is not None
        assert "Restart" in job["error_message"]
        assert ds["ingestion_status"] == "idle"

    def test_creates_pending_resume_job(self, mock_db):
        """Fuer unterbrochene Jobs wird ein neuer pending-Job mit resumed_from_job erstellt."""
        ds_id = mock_db["ds1_id"]
        conn = get_connection()
        cur = conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, embedding_model) "
            "VALUES (%s, 'running', 'nomic') RETURNING id",
            (ds_id,),
        )
        old_job_id = cur.fetchone()["id"]
        conn.commit()
        conn.close()

        from kiara.web.ingestion_worker import IngestionWorker

        worker = IngestionWorker()
        worker._recover_stale_jobs()

        conn = get_connection()
        # Der neue pending-Job
        new_job = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE datasource_id = %s AND status = 'pending'",
            (ds_id,),
        ).fetchone()
        conn.close()

        assert new_job is not None
        assert new_job["datasource_id"] == ds_id
        assert new_job["job_type"] == "incremental"
        assert new_job["resumed_from_job"] == old_job_id
        assert new_job["embedding_model"] == "nomic"

        # stats_json enthaelt resumed_from_job.
        # jsonb-Spalte: psycopg2 liefert dict direkt, sonst string parsen.
        stats_raw = new_job["stats_json"]
        stats = stats_raw if isinstance(stats_raw, dict) else json.loads(stats_raw or "{}")
        assert stats["resumed_from_job"] == old_job_id

    def test_multiple_running_jobs_each_get_resume(self, mock_db):
        """Mehrere unterbrochene Jobs (verschiedene DS) erhalten jeweils einen Resume-Job."""
        ds1_id = mock_db["ds1_id"]
        ds2_id = mock_db["ds2_id"]
        conn = get_connection()
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, embedding_model) "
            "VALUES (%s, 'running', 'nomic')",
            (ds1_id,),
        )
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, embedding_model) "
            "VALUES (%s, 'running', 'nomic')",
            (ds2_id,),
        )
        conn.commit()
        conn.close()

        from kiara.web.ingestion_worker import IngestionWorker

        worker = IngestionWorker()
        worker._recover_stale_jobs()

        conn = get_connection()
        pending = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE status = 'pending' "
            "AND datasource_id IN (%s, %s) ORDER BY datasource_id",
            (ds1_id, ds2_id),
        ).fetchall()
        conn.close()

        assert len(pending) == 2
        assert pending[0]["datasource_id"] == ds1_id
        assert pending[1]["datasource_id"] == ds2_id

    def test_stale_job_without_model_marked_as_error(self, mock_db):
        """Stale Job ohne embedding_model wird als error markiert (kein Resume)."""
        ds_id = mock_db["ds1_id"]
        conn = get_connection()
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, embedding_model) "
            "VALUES (%s, 'running', '')",
            (ds_id,),
        )
        conn.commit()
        conn.close()

        from kiara.web.ingestion_worker import IngestionWorker

        worker = IngestionWorker()
        worker._recover_stale_jobs()

        conn = get_connection()
        pending = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE datasource_id = %s AND status = 'pending'",
            (ds_id,),
        ).fetchall()
        errors = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE datasource_id = %s AND status = 'error'",
            (ds_id,),
        ).fetchall()
        conn.close()

        assert len(pending) == 0
        assert len(errors) == 1

    def test_no_duplicate_resume_if_pending_exists(self, mock_db):
        """Kein Auto-Resume wenn bereits ein pending-Job fuer diese DS existiert."""
        ds_id = mock_db["ds1_id"]
        conn = get_connection()
        # Bestehender pending-Job
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status) VALUES (%s, 'pending')",
            (ds_id,),
        )
        # Zusaetzlich ein running-Job (z.B. durch Absturz)
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, embedding_model) "
            "VALUES (%s, 'running', '')",
            (ds_id,),
        )
        conn.commit()
        conn.close()

        from kiara.web.ingestion_worker import IngestionWorker

        worker = IngestionWorker()
        worker._recover_stale_jobs()

        conn = get_connection()
        pending_jobs = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE datasource_id = %s AND status = 'pending'",
            (ds_id,),
        ).fetchall()
        conn.close()

        # Nur der eine bestehende pending-Job, kein zweiter
        assert len(pending_jobs) == 1

    def test_no_recovery_when_no_running_jobs(self, mock_db):
        """Wenn keine running-Jobs existieren, passiert nichts."""
        from kiara.web.ingestion_worker import IngestionWorker

        worker = IngestionWorker()
        worker._recover_stale_jobs()

        ds_id = mock_db["ds1_id"]
        conn = get_connection()
        jobs = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE datasource_id = %s",
            (ds_id,),
        ).fetchall()
        conn.close()

        assert len(jobs) == 0

    def test_datasource_status_reset_to_idle(self, mock_db):
        """datasources.ingestion_status wird von 'running' auf 'idle' zurueckgesetzt."""
        ds1_id = mock_db["ds1_id"]
        ds2_id = mock_db["ds2_id"]
        conn = get_connection()
        conn.execute(
            "UPDATE datasources SET ingestion_status = 'running' WHERE id = %s",
            (ds1_id,),
        )
        conn.execute(
            "UPDATE datasources SET ingestion_status = 'running' WHERE id = %s",
            (ds2_id,),
        )
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, embedding_model) "
            "VALUES (%s, 'running', '')",
            (ds1_id,),
        )
        conn.commit()
        conn.close()

        from kiara.web.ingestion_worker import IngestionWorker

        worker = IngestionWorker()
        worker._recover_stale_jobs()

        conn = get_connection()
        ds1 = conn.execute("SELECT * FROM datasources WHERE id = %s", (ds1_id,)).fetchone()
        ds2 = conn.execute("SELECT * FROM datasources WHERE id = %s", (ds2_id,)).fetchone()
        conn.close()

        assert ds1["ingestion_status"] == "idle"
        assert ds2["ingestion_status"] == "idle"


# ---------------------------------------------------------------------------
# IMAP Scan-Cache (.scan_done Marker) Tests
# ---------------------------------------------------------------------------


class TestIMAPScanCache:
    """IMAP-Scan-Cache mit .scan_done Marker-Datei."""

    def test_scan_done_marker_created_after_scan(self, tmp_path):
        """IMAPScanner.scan() erstellt .scan_done Marker nach erfolgreichem Scan."""
        output_dir = str(tmp_path / "mail_cache")
        os.makedirs(output_dir, exist_ok=True)

        from kiara.ingestion.imap_scanner import IMAPScanner

        scanner = IMAPScanner(
            {
                "imap_server": "mail.test.de",
                "imap_port": 993,
                "imap_ssl": True,
                "imap_username": "test@test.de",
                "imap_password": "secret",
                "imap_folders": "INBOX",
                "imap_since_days": 30,
            }
        )

        # IMAP-Verbindung mocken
        mock_conn = MagicMock()
        mock_conn.select.return_value = ("OK", [b"1"])
        mock_conn.search.return_value = ("OK", [b""])  # Keine Mails
        mock_conn.logout.return_value = None

        with patch.object(scanner, "connect", return_value=mock_conn):
            scanner.scan(output_dir)

        marker_path = os.path.join(output_dir, ".scan_done")
        assert os.path.exists(marker_path), ".scan_done Marker wurde nicht erstellt"

        # Marker enthaelt einen ISO-Zeitstempel
        with open(marker_path) as f:
            content = f.read()
        assert len(content) > 0

    def test_scan_cache_skips_recent_scan(self, tmp_path):
        """prepare_and_scan() ueberspringt Scan wenn .scan_done aktuell ist."""
        mail_cache = tmp_path / "mail_cache" / "ds1"
        mail_cache.mkdir(parents=True)

        # Aktuellen Marker erstellen
        marker = mail_cache / ".scan_done"
        marker.write_text("2026-03-02T10:00:00")

        config = {"server": "mail.test.de", "username": "test@test.de", "password": "secret"}
        row = {"id": 1}
        ds_config = {}

        with (
            patch(
                "kiara.connectors.base.DEFAULT_PATHS", {"mail_cache": str(tmp_path / "mail_cache")}
            ),
            patch("kiara.ingestion.imap_scanner.IMAPScanner") as MockScanner,
        ):
            from kiara.connectors.imap_connector import IMAPConnector

            result = IMAPConnector.prepare_and_scan(config, row, ds_config)

        # Scanner.scan() sollte NICHT aufgerufen werden
        MockScanner.return_value.scan.assert_not_called()
        assert result == str(mail_cache)

    def test_scan_cache_expired_triggers_rescan(self, tmp_path):
        """prepare_and_scan() fuehrt Scan durch wenn .scan_done aelter als TTL."""
        mail_cache = tmp_path / "mail_cache" / "ds1"
        mail_cache.mkdir(parents=True)

        # Abgelaufenen Marker erstellen (ueber 24h alt)
        marker = mail_cache / ".scan_done"
        marker.write_text("2026-03-01T00:00:00")
        # mtime auf 25h in die Vergangenheit setzen
        old_time = time.time() - (25 * 3600)
        os.utime(str(marker), (old_time, old_time))

        config = {"server": "mail.test.de", "username": "test@test.de", "password": "secret"}
        row = {"id": 1}
        ds_config = {}

        mock_scanner_instance = MagicMock()
        mock_scanner_instance.scan.return_value = ["/some/file.txt"]

        with (
            patch(
                "kiara.connectors.base.DEFAULT_PATHS", {"mail_cache": str(tmp_path / "mail_cache")}
            ),
            patch("kiara.ingestion.imap_scanner.IMAPScanner", return_value=mock_scanner_instance),
        ):
            from kiara.connectors.imap_connector import IMAPConnector

            result = IMAPConnector.prepare_and_scan(config, row, ds_config)

        # Scanner.scan() MUSS aufgerufen werden (Marker abgelaufen)
        mock_scanner_instance.scan.assert_called_once()
        assert result == str(mail_cache)

    def test_scan_cache_no_marker_triggers_scan(self, tmp_path):
        """prepare_and_scan() fuehrt Scan durch wenn kein .scan_done existiert."""
        mail_cache = tmp_path / "mail_cache" / "ds1"
        mail_cache.mkdir(parents=True)

        # Kein Marker vorhanden
        config = {"server": "mail.test.de", "username": "test@test.de", "password": "secret"}
        row = {"id": 1}
        ds_config = {}

        mock_scanner_instance = MagicMock()
        mock_scanner_instance.scan.return_value = []

        with (
            patch(
                "kiara.connectors.base.DEFAULT_PATHS", {"mail_cache": str(tmp_path / "mail_cache")}
            ),
            patch("kiara.ingestion.imap_scanner.IMAPScanner", return_value=mock_scanner_instance),
        ):
            from kiara.connectors.imap_connector import IMAPConnector

            result = IMAPConnector.prepare_and_scan(config, row, ds_config)

        mock_scanner_instance.scan.assert_called_once()
        assert result == str(mail_cache)

    def test_scan_cache_ttl_boundary(self, tmp_path):
        """Marker genau innerhalb TTL wird noch akzeptiert (kein Re-Scan)."""
        from kiara.connectors.imap_connector import IMAPConnector

        mail_cache = tmp_path / "mail_cache" / "ds1"
        mail_cache.mkdir(parents=True)

        marker = mail_cache / ".scan_done"
        marker.write_text("2026-03-02T10:00:00")
        # mtime auf 23h in die Vergangenheit (innerhalb 24h TTL)
        recent_time = time.time() - (23 * 3600)
        os.utime(str(marker), (recent_time, recent_time))

        config = {"server": "mail.test.de", "username": "test@test.de", "password": "secret"}
        row = {"id": 1}
        ds_config = {}

        with (
            patch(
                "kiara.connectors.base.DEFAULT_PATHS", {"mail_cache": str(tmp_path / "mail_cache")}
            ),
            patch("kiara.ingestion.imap_scanner.IMAPScanner") as MockScanner,
        ):
            result = IMAPConnector.prepare_and_scan(config, row, ds_config)

        MockScanner.return_value.scan.assert_not_called()
        assert result == str(mail_cache)
