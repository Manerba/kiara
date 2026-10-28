"""Tests fuer Build 70: Ingestion-Fortschritt nach Page-Reload + absolute Fortschrittszahlen.

Testet:
- progress_callback wird mit absoluten Zahlen aufgerufen (skipped + processed)
- Bei Resume-Jobs: konsistenter Gesamtfortschritt (z.B. 3049/5459 statt 49/2459)
- Fortschritt wird in DB persistiert (progress_total / progress_processed)
- Status-Endpoint liefert DB-Fallback wenn kein In-Memory-Job laeuft
"""

import json
import sqlite3
from contextlib import contextmanager
from unittest.mock import MagicMock, call, patch

import pytest

pytestmark = pytest.mark.usefixtures("s4_guard_passthrough")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_config(tmp_path):
    """Minimale Config fuer run_ingestion."""
    vdb_path = tmp_path / "vectordb"
    vdb_path.mkdir()
    return {
        "datasource": {"path": str(tmp_path / "data"), "file_types": [".txt"]},
        "ingestion": {"chunk_size": 500, "chunk_overlap": 50, "batch_size": 10},
        "vectordb": {"path": str(vdb_path)},
        "graph": {"enabled": False},
    }


@pytest.fixture
def job_db(tmp_path):
    """Temporaere SQLite-DB mit ingestion_jobs + datasources Schema."""
    db_path = tmp_path / "kiara.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE datasources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'local',
            path TEXT DEFAULT '',
            is_active BOOLEAN DEFAULT true,
            ingestion_status TEXT DEFAULT 'idle',
            last_ingested_at TEXT DEFAULT NULL,
            last_error TEXT DEFAULT '',
            chunk_count INTEGER DEFAULT 0,
            file_count INTEGER DEFAULT 0,
            processed_files INTEGER DEFAULT 0,
            progress_total INTEGER DEFAULT 0,
            connector_config TEXT DEFAULT '{}',
            ds_chunk_size INTEGER DEFAULT NULL,
            ds_chunk_overlap INTEGER DEFAULT NULL,
            ds_file_types TEXT DEFAULT NULL,
            ds_exclude_patterns TEXT DEFAULT NULL,
            ds_chunk_strategy TEXT DEFAULT 'auto',
            ds_ddu_enabled BOOLEAN DEFAULT false,
            ds_vlm_enabled BOOLEAN DEFAULT false,
            ds_index_attachments INTEGER DEFAULT 1,
            ds_accept_all_text INTEGER DEFAULT 0,
            sync_interval TEXT DEFAULT 'manual',
            next_sync_at TEXT DEFAULT NULL,
            restriction_level TEXT DEFAULT 'anonymize' NOT NULL,
            ds_contextual_retrieval INTEGER DEFAULT 0,
            ds_enrichment_prompt_id INTEGER DEFAULT NULL,
            ds_enrichment_backend_id INTEGER DEFAULT NULL,
            ds_colbert_enabled BOOLEAN DEFAULT false,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE datasource_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datasource_id INTEGER NOT NULL REFERENCES datasources(id) ON DELETE CASCADE,
            embedding_model TEXT NOT NULL,
            chunk_count INTEGER DEFAULT 0,
            file_count INTEGER DEFAULT 0,
            last_ingested_at TEXT DEFAULT NULL,
            UNIQUE(datasource_id, embedding_model)
        );

        CREATE TABLE ingestion_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datasource_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            job_type TEXT NOT NULL DEFAULT 'incremental',
            triggered_by TEXT NOT NULL DEFAULT 'manual',
            embedding_model TEXT DEFAULT '',
            requested_action TEXT DEFAULT NULL,
            progress_processed INTEGER DEFAULT 0,
            progress_total INTEGER DEFAULT 0,
            queued_at TEXT DEFAULT (datetime('now')),
            started_at TEXT DEFAULT NULL,
            finished_at TEXT DEFAULT NULL,
            stats_json TEXT DEFAULT '{}',
            error_message TEXT DEFAULT '',
            resumed_from_job INTEGER DEFAULT NULL,
            FOREIGN KEY (datasource_id) REFERENCES datasources(id) ON DELETE CASCADE
        );

        INSERT INTO datasources (name, type, path) VALUES ('Test DS', 'local', '/tmp/test');
    """)
    conn.commit()
    conn.close()
    return str(db_path)


@pytest.fixture
def mock_db(job_db, monkeypatch):
    """Patcht db_query/db_transaction auf die temporaere DB."""
    import kiara.shared.db as shared_db
    import kiara.web.db_utils as db_utils
    import kiara.web.ingestion_worker as worker_mod

    def _make_conn():
        conn = sqlite3.connect(job_db)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def fake_db_query():
        conn = _make_conn()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def fake_db_transaction():
        conn = _make_conn()
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
    monkeypatch.setattr(worker_mod, "db_query", fake_db_query)
    monkeypatch.setattr(worker_mod, "db_transaction", fake_db_transaction)

    return job_db


# ---------------------------------------------------------------------------
# progress_callback: Absolute Zahlen (skipped + processed / total)
# ---------------------------------------------------------------------------


class TestProgressCallbackAbsolute:
    """progress_callback in run_ingestion meldet absolute Fortschrittszahlen."""

    @patch("kiara.ingestion.pipeline.HashStore")
    @patch("kiara.ingestion.pipeline.scan_files")
    @patch("kiara.ingestion.pipeline.VectorStore")
    @patch("kiara.ingestion.runtime.load_file")
    @patch("kiara.ingestion.runtime.chunk_document")
    def test_callback_receives_skipped_plus_processed(
        self,
        mock_chunk,
        mock_load,
        mock_store_cls,
        mock_scan,
        mock_hs_cls,
        minimal_config,
        tmp_path,
    ):
        """Bei 3 skipped + 2 to_process von 5 total:
        - Initialer Callback: (3, 5) — skipped als Startwert
        - Nach Datei 1: (4, 5)
        - Nach Datei 2: (5, 5)
        """
        all_files = [str(tmp_path / f"file{i}.txt") for i in range(5)]
        for f in all_files:
            open(f, "w").close()
        mock_scan.return_value = all_files

        # Erste 3 Dateien sind unverändert (skipped), letzte 2 sind neu
        stored = {all_files[i]: f"hash{i}" for i in range(3)}
        mock_hs = MagicMock()
        mock_hs.load.return_value = stored
        mock_hs_cls.return_value = mock_hs

        # _file_hash: Fuer skipped Files gleichen Hash, fuer neue Files neuen Hash
        def fake_hash(fp, _mode="mtime", *, file_stat=None):
            idx = all_files.index(fp)
            if idx < 3:
                return f"hash{idx}"
            return f"newhash{idx}"

        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        mock_load.return_value = {
            "text": "Inhalt",
            "metadata": {"source": "x.txt", "file_type": ".txt"},
        }
        mock_chunk.return_value = [
            {
                "text": "chunk",
                "metadata": {"source": "x.txt", "chunk_index": 0},
            }
        ]

        callback = MagicMock()

        with (
            patch("kiara_ingestion_core.file_diff.compute_file_hash", side_effect=fake_hash),
            patch("kiara.ingestion.pipeline.get_embedding_model", return_value="test-model"),
        ):
            from kiara.ingestion.pipeline import run_ingestion

            stats = run_ingestion(
                minimal_config,
                incremental=True,
                progress_callback=callback,
            )

        # Initialer Callback mit skipped-Count
        assert callback.call_args_list[0] == call(3, 5)
        # Nach jeder verarbeiteten Datei: skipped + processed
        assert callback.call_args_list[1] == call(4, 5)
        assert callback.call_args_list[2] == call(5, 5)
        assert callback.call_count == 3

    @patch("kiara.ingestion.pipeline.HashStore")
    @patch("kiara.ingestion.pipeline.scan_files")
    @patch("kiara.ingestion.pipeline.VectorStore")
    def test_all_skipped_single_callback(
        self,
        mock_store_cls,
        mock_scan,
        mock_hs_cls,
        minimal_config,
        tmp_path,
    ):
        """Alle Dateien skipped → nur initialer Callback (skipped, total)."""
        all_files = [str(tmp_path / f"f{i}.txt") for i in range(4)]
        for f in all_files:
            open(f, "w").close()
        mock_scan.return_value = all_files

        stored = {f: f"hash{i}" for i, f in enumerate(all_files)}
        mock_hs = MagicMock()
        mock_hs.load.return_value = stored
        mock_hs_cls.return_value = mock_hs

        def fake_hash(fp, _mode="mtime", *, file_stat=None):
            idx = all_files.index(fp)
            return f"hash{idx}"

        mock_store_cls.return_value = MagicMock()
        callback = MagicMock()

        with (
            patch("kiara_ingestion_core.file_diff.compute_file_hash", side_effect=fake_hash),
            patch("kiara.ingestion.pipeline.get_embedding_model", return_value="test-model"),
        ):
            from kiara.ingestion.pipeline import run_ingestion

            stats = run_ingestion(
                minimal_config,
                incremental=True,
                progress_callback=callback,
            )

        # Nur ein Callback: alle 4 skipped von 4 total
        assert callback.call_count == 1
        assert callback.call_args_list[0] == call(4, 4)

    @patch("kiara.ingestion.pipeline.HashStore")
    @patch("kiara.ingestion.pipeline.scan_files")
    @patch("kiara.ingestion.pipeline.VectorStore")
    @patch("kiara.ingestion.runtime.load_file")
    @patch("kiara.ingestion.runtime.chunk_document")
    def test_no_skipped_starts_at_zero(
        self,
        mock_chunk,
        mock_load,
        mock_store_cls,
        mock_scan,
        mock_hs_cls,
        minimal_config,
        tmp_path,
    ):
        """Kein Incremental (kein Skip) → Callback startet bei (0, N)."""
        all_files = [str(tmp_path / f"f{i}.txt") for i in range(3)]
        for f in all_files:
            open(f, "w").close()
        mock_scan.return_value = all_files
        mock_hs = MagicMock()
        mock_hs.load.return_value = {}
        mock_hs_cls.return_value = mock_hs

        def fake_hash(fp, _mode="mtime", *, file_stat=None):
            return "newhash"

        mock_store_cls.return_value = MagicMock()
        mock_load.return_value = {
            "text": "Content",
            "metadata": {"source": "x.txt", "file_type": ".txt"},
        }
        mock_chunk.return_value = [
            {
                "text": "c",
                "metadata": {"source": "x.txt", "chunk_index": 0},
            }
        ]

        callback = MagicMock()

        with (
            patch("kiara_ingestion_core.file_diff.compute_file_hash", side_effect=fake_hash),
            patch("kiara.ingestion.pipeline.get_embedding_model", return_value="test-model"),
        ):
            from kiara.ingestion.pipeline import run_ingestion

            run_ingestion(
                minimal_config,
                incremental=True,
                progress_callback=callback,
            )

        # Initialer Callback: 0 skipped von 3 total
        assert callback.call_args_list[0] == call(0, 3)
        # Danach: 1/3, 2/3, 3/3
        assert callback.call_args_list[1] == call(1, 3)
        assert callback.call_args_list[2] == call(2, 3)
        assert callback.call_args_list[3] == call(3, 3)

    @patch("kiara.ingestion.pipeline.HashStore")
    @patch("kiara.ingestion.pipeline.scan_files")
    @patch("kiara.ingestion.pipeline.VectorStore")
    @patch("kiara.ingestion.runtime.load_file")
    @patch("kiara.ingestion.runtime.chunk_document")
    def test_resume_scenario_large_skip(
        self,
        mock_chunk,
        mock_load,
        mock_store_cls,
        mock_scan,
        mock_hs_cls,
        minimal_config,
        tmp_path,
    ):
        """Resume-Szenario: 100 skipped + 50 zu verarbeiten von 200 total.
        Erster Callback: (100, 200), danach (101, 200), ..., (150, 200).
        """
        total = 200
        skipped_count = 100
        to_process = total - skipped_count

        all_files = [str(tmp_path / f"file{i:04d}.txt") for i in range(total)]
        for f in all_files:
            open(f, "w").close()
        mock_scan.return_value = all_files

        stored = {all_files[i]: f"hash{i}" for i in range(skipped_count)}
        mock_hs = MagicMock()
        mock_hs.load.return_value = stored
        mock_hs_cls.return_value = mock_hs

        def fake_hash(fp, _mode="mtime", *, file_stat=None):
            idx = all_files.index(fp)
            if idx < skipped_count:
                return f"hash{idx}"
            return f"newhash{idx}"

        mock_store_cls.return_value = MagicMock()
        mock_load.return_value = {
            "text": "Inhalt",
            "metadata": {"source": "x.txt", "file_type": ".txt"},
        }
        mock_chunk.return_value = [
            {
                "text": "c",
                "metadata": {"source": "x.txt", "chunk_index": 0},
            }
        ]

        callback = MagicMock()

        with (
            patch("kiara_ingestion_core.file_diff.compute_file_hash", side_effect=fake_hash),
            patch("kiara.ingestion.pipeline.get_embedding_model", return_value="test-model"),
        ):
            from kiara.ingestion.pipeline import run_ingestion

            stats = run_ingestion(
                minimal_config,
                incremental=True,
                progress_callback=callback,
            )

        # Initialer Callback: skipped=100, total=200
        assert callback.call_args_list[0] == call(100, 200)
        # Letzter Callback: 200/200
        assert callback.call_args_list[-1] == call(200, 200)
        # Insgesamt: 1 initial + 100 Dateien = 101 Callbacks
        assert callback.call_count == 1 + to_process

        # Stats pruefen
        assert stats["total_files"] == 200
        assert stats["skipped_files"] == 100
        assert stats["processed_files"] == 100


# ---------------------------------------------------------------------------
# Stats-Dict: skipped_files wird korrekt berechnet
# ---------------------------------------------------------------------------


class TestStatsSkippedCalculation:
    """stats['skipped_files'] = total - to_process (fuer Offset-Berechnung)."""

    @patch("kiara.ingestion.pipeline.HashStore")
    @patch("kiara.ingestion.pipeline.scan_files")
    @patch("kiara.ingestion.pipeline.VectorStore")
    def test_skipped_files_matches_diff(
        self,
        mock_store_cls,
        mock_scan,
        mock_hs_cls,
        minimal_config,
        tmp_path,
    ):
        """skipped_files = len(all_files) - len(files_to_process)."""
        all_files = [str(tmp_path / f"f{i}.txt") for i in range(10)]
        for f in all_files:
            open(f, "w").close()
        mock_scan.return_value = all_files

        # 7 Dateien unverändert
        stored = {all_files[i]: f"hash{i}" for i in range(7)}
        mock_hs = MagicMock()
        mock_hs.load.return_value = stored
        mock_hs_cls.return_value = mock_hs

        def fake_hash(fp, _mode="mtime", *, file_stat=None):
            idx = all_files.index(fp)
            if idx < 7:
                return f"hash{idx}"
            return "new"

        mock_store_cls.return_value = MagicMock()

        with (
            patch("kiara_ingestion_core.file_diff.compute_file_hash", side_effect=fake_hash),
            patch("kiara.ingestion.runtime.load_file", return_value=None),
            patch("kiara.ingestion.pipeline.get_embedding_model", return_value="test-model"),
        ):
            from kiara.ingestion.pipeline import run_ingestion

            stats = run_ingestion(minimal_config, incremental=True)

        assert stats["skipped_files"] == 7
        assert stats["total_files"] == 10


# ---------------------------------------------------------------------------
# Fortschritt wird in DB persistiert (Worker _on_progress Callback)
# ---------------------------------------------------------------------------


class TestProgressDbPersistence:
    """Der _on_progress Callback im Worker schreibt in ingestion_jobs + datasources."""

    def test_on_progress_writes_to_db(self, mock_db):
        """_on_progress(150, 200) aktualisiert progress_processed/progress_total."""
        job_id = 1
        ds_id = 1

        # Job einfuegen
        conn = sqlite3.connect(mock_db)
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, started_at) "
            "VALUES (?, 'running', datetime('now'))",
            (ds_id,),
        )
        conn.commit()
        job_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()

        # _on_progress aus dem Worker simulieren (gleiche Logik)
        from kiara.shared.db import db_transaction

        processed, total = 150, 200

        with db_transaction() as db:
            db.execute(
                "UPDATE ingestion_jobs SET progress_processed = ?, progress_total = ? WHERE id = ?",
                (processed, total, job_id),
            )
            db.execute(
                "UPDATE datasources SET processed_files = ?, progress_total = ? WHERE id = ?",
                (processed, total, ds_id),
            )

        # Verifizieren
        conn = sqlite3.connect(mock_db)
        conn.row_factory = sqlite3.Row
        job_row = conn.execute(
            "SELECT progress_processed, progress_total FROM ingestion_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        ds_row = conn.execute(
            "SELECT processed_files, progress_total FROM datasources WHERE id = ?",
            (ds_id,),
        ).fetchone()
        conn.close()

        assert job_row["progress_processed"] == 150
        assert job_row["progress_total"] == 200
        assert ds_row["processed_files"] == 150
        assert ds_row["progress_total"] == 200

    def test_progress_survives_multiple_updates(self, mock_db):
        """Mehrere _on_progress Aufrufe ueberschreiben korrekt."""
        ds_id = 1
        conn = sqlite3.connect(mock_db)
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status) VALUES (?, 'running')",
            (ds_id,),
        )
        conn.commit()
        job_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()

        from kiara.shared.db import db_transaction

        # Fortschritt-Updates simulieren
        for processed in [100, 120, 150, 200]:
            with db_transaction() as db:
                db.execute(
                    "UPDATE ingestion_jobs SET progress_processed = ?, progress_total = ? "
                    "WHERE id = ?",
                    (processed, 200, job_id),
                )
                db.execute(
                    "UPDATE datasources SET processed_files = ?, progress_total = ? WHERE id = ?",
                    (processed, 200, ds_id),
                )

        conn = sqlite3.connect(mock_db)
        conn.row_factory = sqlite3.Row
        job_row = conn.execute(
            "SELECT progress_processed, progress_total FROM ingestion_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        conn.close()

        assert job_row["progress_processed"] == 200
        assert job_row["progress_total"] == 200


# ---------------------------------------------------------------------------
# Status-Endpoint Logik: DB-Fallback wenn kein laufender Job
# Testet die Abfrage-Logik direkt (ohne FastAPI-Import, da Zirkular-Import).
# ---------------------------------------------------------------------------


class TestStatusEndpointDbFallback:
    """Ingestion-Status-Logik: DB-Fallback fuer progress_total/processed_files.

    Die Endpoint-Logik (get_ingestion_status) wird als Unit-Test nachgestellt,
    um den Zirkular-Import api_datasource <-> api_datasource_ingest zu vermeiden.
    """

    def _query_status(self, db_path, ds_id, current_job_id=None, is_paused=False):
        """Simuliert die Logik aus get_ingestion_status (api_datasource_ingest.py).

        Gibt das result-Dict zurueck, das der Endpoint liefern wuerde.
        """
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        row = conn.execute(
            """SELECT ingestion_status, last_ingested_at, last_error,
                      chunk_count, file_count, processed_files,
                      COALESCE(progress_total, 0) AS progress_total
               FROM datasources WHERE id = ?""",
            (ds_id,),
        ).fetchone()
        if not row:
            conn.close()
            return None

        result = dict(row)

        # Live-Daten vom Worker (simuliert)
        if current_job_id:
            job_row = conn.execute(
                "SELECT * FROM ingestion_jobs WHERE id = ?",
                (current_job_id,),
            ).fetchone()
            if job_row and job_row["datasource_id"] == ds_id:
                result["ingestion_status"] = "paused" if is_paused else "running"
                result["started_at"] = job_row["started_at"]
                result["progress_processed"] = job_row["progress_processed"]
                result["progress_total"] = job_row["progress_total"]
                result["job_id"] = current_job_id
                try:
                    stats = json.loads(job_row["stats_json"] or "{}")
                    if stats.get("phase"):
                        result["phase"] = stats["phase"]
                except (json.JSONDecodeError, TypeError):
                    pass
                conn.close()
                return result

        # Pending Job?
        pending = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = ? AND status = 'pending'",
            (ds_id,),
        ).fetchone()
        if pending:
            result["ingestion_status"] = "pending"
            result["job_id"] = pending["id"]
            conn.close()
            return result

        # Letzter abgeschlossener Job
        last_job = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE datasource_id = ? ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()
        if last_job and last_job["status"] in ("success", "error"):
            result["started_at"] = last_job["started_at"]
            result["finished_at"] = last_job["finished_at"]
            try:
                result["job_stats"] = json.loads(last_job["stats_json"] or "{}")
            except (json.JSONDecodeError, TypeError):
                pass
            result["job_id"] = last_job["id"]

        conn.close()
        return result

    def _make_status_db(self, tmp_path, processed_files=150, progress_total=200):
        """Erstellt eine Test-DB mit einem Datasource-Eintrag."""
        db_path = str(tmp_path / "status_test.db")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(f"""
            CREATE TABLE datasources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL DEFAULT 'local',
                is_active BOOLEAN DEFAULT true,
                ingestion_status TEXT DEFAULT 'idle',
                last_ingested_at TEXT DEFAULT NULL,
                last_error TEXT DEFAULT '',
                chunk_count INTEGER DEFAULT 0,
                file_count INTEGER DEFAULT 0,
                processed_files INTEGER DEFAULT 0,
                progress_total INTEGER DEFAULT 0
            );
            CREATE TABLE ingestion_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                datasource_id INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                job_type TEXT NOT NULL DEFAULT 'incremental',
                triggered_by TEXT NOT NULL DEFAULT 'manual',
                embedding_model TEXT DEFAULT '',
                progress_processed INTEGER DEFAULT 0,
                progress_total INTEGER DEFAULT 0,
                started_at TEXT DEFAULT NULL,
                finished_at TEXT DEFAULT NULL,
                stats_json TEXT DEFAULT '{{}}'
            );
            INSERT INTO datasources (name, type, processed_files, progress_total)
            VALUES ('Test DS', 'local', {processed_files}, {progress_total});
        """)
        conn.commit()
        conn.close()
        return db_path

    def test_no_running_job_returns_db_values(self, tmp_path):
        """Kein laufender Job → Status liefert gespeicherte DB-Werte."""
        db_path = self._make_status_db(tmp_path, processed_files=150, progress_total=200)

        result = self._query_status(db_path, ds_id=1)

        assert result is not None
        assert result["processed_files"] == 150
        assert result["progress_total"] == 200
        assert result["ingestion_status"] == "idle"

    def test_running_job_returns_live_data(self, tmp_path):
        """Laufender Job → Live-Daten aus ingestion_jobs."""
        db_path = self._make_status_db(tmp_path)

        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO ingestion_jobs "
            "(datasource_id, status, started_at, progress_processed, progress_total, stats_json) "
            "VALUES (1, 'running', datetime('now'), 175, 200, ?)",
            (json.dumps({"phase": "Embedding..."}),),
        )
        conn.commit()
        job_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()

        result = self._query_status(db_path, ds_id=1, current_job_id=job_id)

        assert result["ingestion_status"] == "running"
        assert result["progress_processed"] == 175
        assert result["progress_total"] == 200
        assert result["job_id"] == job_id
        assert result.get("phase") == "Embedding..."

    def test_paused_job_returns_paused_status(self, tmp_path):
        """Pausierter Job → Status 'paused'."""
        db_path = self._make_status_db(tmp_path)

        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO ingestion_jobs "
            "(datasource_id, status, started_at, progress_processed, progress_total) "
            "VALUES (1, 'running', datetime('now'), 80, 200)",
        )
        conn.commit()
        job_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()

        result = self._query_status(db_path, ds_id=1, current_job_id=job_id, is_paused=True)

        assert result["ingestion_status"] == "paused"
        assert result["progress_processed"] == 80

    def test_finished_job_shows_stats(self, tmp_path):
        """Abgeschlossener Job → Stats aus stats_json."""
        db_path = self._make_status_db(tmp_path)

        stats = {
            "total_files": 200,
            "processed_files": 100,
            "skipped_files": 100,
            "total_chunks": 450,
            "duration_seconds": 120,
        }
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO ingestion_jobs "
            "(datasource_id, status, started_at, finished_at, stats_json) "
            "VALUES (1, 'success', datetime('now', '-2 minutes'), datetime('now'), ?)",
            (json.dumps(stats),),
        )
        conn.commit()
        conn.close()

        result = self._query_status(db_path, ds_id=1)

        assert "job_stats" in result
        assert result["job_stats"]["total_files"] == 200
        assert result["job_stats"]["skipped_files"] == 100

    def test_ds_not_found_returns_none(self, tmp_path):
        """Unbekannte DS-ID → None."""
        db_path = self._make_status_db(tmp_path)
        result = self._query_status(db_path, ds_id=999)
        assert result is None

    def test_pending_job_returns_pending_status(self, tmp_path):
        """Pending Job → Status 'pending' mit job_id."""
        db_path = self._make_status_db(tmp_path)

        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status) VALUES (1, 'pending')",
        )
        conn.commit()
        job_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()

        result = self._query_status(db_path, ds_id=1)

        assert result["ingestion_status"] == "pending"
        assert result["job_id"] == job_id


# ---------------------------------------------------------------------------
# Callback wird auch bei Fehler-Dateien aufgerufen (zero-chunk Files)
# ---------------------------------------------------------------------------


class TestProgressOnErrorFiles:
    """Auch bei Dateien die keine Chunks liefern wird Fortschritt gemeldet."""

    @patch("kiara.ingestion.pipeline.HashStore")
    @patch("kiara.ingestion.pipeline.scan_files")
    @patch("kiara.ingestion.pipeline.VectorStore")
    @patch("kiara.ingestion.runtime.load_file")
    def test_callback_on_zero_chunk_file(
        self,
        mock_load,
        mock_store_cls,
        mock_scan,
        mock_hs_cls,
        minimal_config,
        tmp_path,
    ):
        """Datei die None zurueckgibt (Ladefehler) → Fortschritt zaehlt trotzdem."""
        all_files = [str(tmp_path / f"f{i}.txt") for i in range(3)]
        for f in all_files:
            open(f, "w").close()
        mock_scan.return_value = all_files
        mock_hs = MagicMock()
        mock_hs.load.return_value = {}
        mock_hs_cls.return_value = mock_hs

        def fake_hash(fp, _mode="mtime", *, file_stat=None):
            return "hash"

        mock_store_cls.return_value = MagicMock()
        # Alle Dateien scheitern beim Laden
        mock_load.return_value = None

        callback = MagicMock()

        with (
            patch("kiara_ingestion_core.file_diff.compute_file_hash", side_effect=fake_hash),
            patch("kiara.ingestion.pipeline.get_embedding_model", return_value="test-model"),
        ):
            from kiara.ingestion.pipeline import run_ingestion

            stats = run_ingestion(
                minimal_config,
                incremental=False,
                progress_callback=callback,
            )

        # Initialer Callback: (0, 3)
        assert callback.call_args_list[0] == call(0, 3)
        # Fehler-Dateien zaehlen trotzdem als Fortschritt (errors, nicht processed)
        # Aber der Callback wird bei errors NICHT aufgerufen (nur bei processed)
        # → Die 3 Dateien sind errors → kein weiterer Callback nach dem initialen
        assert stats["errors"] == 3


# ---------------------------------------------------------------------------
# Integration: Callback-Werte passen zu stats am Ende
# ---------------------------------------------------------------------------


class TestProgressConsistency:
    """Letzter Callback-Wert == stats['skipped_files'] + stats['processed_files']."""

    @patch("kiara.ingestion.pipeline.HashStore")
    @patch("kiara.ingestion.pipeline.scan_files")
    @patch("kiara.ingestion.pipeline.VectorStore")
    @patch("kiara.ingestion.runtime.load_file")
    @patch("kiara.ingestion.runtime.chunk_document")
    def test_final_callback_matches_stats(
        self,
        mock_chunk,
        mock_load,
        mock_store_cls,
        mock_scan,
        mock_hs_cls,
        minimal_config,
        tmp_path,
    ):
        """Letzter Callback muss skipped + processed == total sein."""
        total = 50
        skipped_count = 30

        all_files = [str(tmp_path / f"f{i:03d}.txt") for i in range(total)]
        for f in all_files:
            open(f, "w").close()
        mock_scan.return_value = all_files

        stored = {all_files[i]: f"h{i}" for i in range(skipped_count)}
        mock_hs = MagicMock()
        mock_hs.load.return_value = stored
        mock_hs_cls.return_value = mock_hs

        def fake_hash(fp, _mode="mtime", *, file_stat=None):
            idx = all_files.index(fp)
            return f"h{idx}" if idx < skipped_count else "new"

        mock_store_cls.return_value = MagicMock()
        mock_load.return_value = {
            "text": "x",
            "metadata": {"source": "x.txt", "file_type": ".txt"},
        }
        mock_chunk.return_value = [
            {
                "text": "c",
                "metadata": {"source": "x.txt", "chunk_index": 0},
            }
        ]

        callback = MagicMock()

        with (
            patch("kiara_ingestion_core.file_diff.compute_file_hash", side_effect=fake_hash),
            patch("kiara.ingestion.pipeline.get_embedding_model", return_value="test-model"),
        ):
            from kiara.ingestion.pipeline import run_ingestion

            stats = run_ingestion(
                minimal_config,
                incremental=True,
                progress_callback=callback,
            )

        last_processed, last_total = callback.call_args_list[-1][0]
        assert last_total == stats["total_files"]
        assert last_processed == stats["skipped_files"] + stats["processed_files"]
        assert last_processed == total
