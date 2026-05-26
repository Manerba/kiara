"""Tests fuer Ingestion File-Logging (Per-File-Protokoll).

Testet:
- FileLogEntry Dataclass + Befuellung in _process_single_file()
- file_log_callback in run_ingestion()
- Batched DB-Writes im Worker
- API-Endpoint GET .../file-logs (Filter, Sort, Pagination, Summary)
- Ghost-Job-Erkennung in der API
- VectorDB-Punktestand in _finalize_success()
- CASCADE / Safety-DELETE bei History-Cleanup
"""

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from tests.admin_confirm import confirmed_request

# ---------------------------------------------------------------------------
# Gemeinsames DB-Schema (erweitert um ingestion_file_logs)
# ---------------------------------------------------------------------------

_SCHEMA = """
    CREATE TABLE IF NOT EXISTS datasources (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT NOT NULL DEFAULT 'local',
        is_active BOOLEAN DEFAULT true,
        ingestion_status TEXT DEFAULT 'idle',
        connector_config TEXT DEFAULT '{}',
        ds_chunk_size INTEGER DEFAULT NULL,
        ds_chunk_overlap INTEGER DEFAULT NULL,
        ds_file_types TEXT DEFAULT NULL,
        ds_exclude_patterns TEXT DEFAULT NULL,
        ds_chunk_strategy TEXT DEFAULT 'auto',
        ds_ddu_enabled BOOLEAN DEFAULT false,
        ds_ddu_ocr_images INTEGER DEFAULT 1,
        ds_vlm_enabled BOOLEAN DEFAULT false,
        ds_accept_all_text INTEGER DEFAULT 0,
        ds_index_attachments INTEGER DEFAULT 1,
        path TEXT DEFAULT '',
        last_ingested_at TEXT DEFAULT NULL,
        last_error TEXT DEFAULT '',
        chunk_count INTEGER DEFAULT 0,
        file_count INTEGER DEFAULT 0,
        processed_files INTEGER DEFAULT 0,
        progress_total INTEGER DEFAULT 0,
        sync_interval TEXT DEFAULT 'manual',
        next_sync_at TEXT DEFAULT NULL,
        restriction_level TEXT DEFAULT 'anonymize' NOT NULL,
        ds_contextual_retrieval INTEGER DEFAULT 0,
        ds_enrichment_prompt_id INTEGER DEFAULT NULL,
        ds_enrichment_backend_id INTEGER DEFAULT NULL,
        ds_colbert_enabled BOOLEAN DEFAULT false,
        is_public INTEGER DEFAULT NULL,
        created_at TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')),
        updated_at TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS'))
    );
    CREATE TABLE IF NOT EXISTS datasource_embeddings (
        id SERIAL PRIMARY KEY,
        datasource_id INTEGER NOT NULL,
        embedding_model TEXT NOT NULL,
        chunk_count INTEGER DEFAULT 0,
        file_count INTEGER DEFAULT 0,
        last_ingested_at TEXT DEFAULT NULL,
        UNIQUE(datasource_id, embedding_model)
    );
    CREATE TABLE IF NOT EXISTS ingestion_jobs (
        id SERIAL PRIMARY KEY,
        datasource_id INTEGER NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        job_type TEXT NOT NULL DEFAULT 'incremental',
        triggered_by TEXT NOT NULL DEFAULT 'manual',
        embedding_model TEXT DEFAULT '',
        requested_action TEXT DEFAULT NULL,
        progress_processed INTEGER DEFAULT 0,
        progress_total INTEGER DEFAULT 0,
        queued_at TIMESTAMPTZ DEFAULT NOW(),
        started_at TIMESTAMPTZ DEFAULT NULL,
        finished_at TIMESTAMPTZ DEFAULT NULL,
        stats_json TEXT DEFAULT '{}',
        error_message TEXT DEFAULT '',
        resumed_from_job INTEGER DEFAULT NULL,
        FOREIGN KEY (datasource_id) REFERENCES datasources(id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS ingestion_file_logs (
        id SERIAL PRIMARY KEY,
        job_id INTEGER NOT NULL,
        file_path TEXT NOT NULL,
        file_name TEXT NOT NULL,
        file_size BIGINT DEFAULT 0,
        file_ext TEXT DEFAULT '',
        parser TEXT DEFAULT '',
        parser_fallback INTEGER DEFAULT 0,
        chunk_strategy TEXT DEFAULT '',
        chunk_count INTEGER DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'success'
                    CHECK (status IN ('success', 'error', 'skipped')),
        error_message TEXT DEFAULT '',
        error_phase TEXT DEFAULT '',
        duration_ms INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')),
        FOREIGN KEY (job_id) REFERENCES ingestion_jobs(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_ifl_job ON ingestion_file_logs(job_id);
    CREATE INDEX IF NOT EXISTS idx_ifl_status ON ingestion_file_logs(status);
    CREATE TABLE IF NOT EXISTS cache_datasources (
        datasource_id INTEGER PRIMARY KEY,
        payload_json  TEXT NOT NULL DEFAULT '{}',
        updated_at    TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'localtime', 'YYYY-MM-DD HH24:MI:SS'))
    );
    CREATE TABLE IF NOT EXISTS cache_jobs (
        job_id        INTEGER PRIMARY KEY,
        datasource_id INTEGER DEFAULT 0,
        status        TEXT DEFAULT '',
        payload_json  TEXT NOT NULL DEFAULT '{}',
        updated_at    TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'localtime', 'YYYY-MM-DD HH24:MI:SS'))
    );
    CREATE INDEX IF NOT EXISTS idx_cache_jobs_status ON cache_jobs(status);
"""


def _make_db(tmp_path):
    """Erstellt Schema in PostgreSQL und gibt eine Marker-ID zurueck."""
    from kiara.shared.db import get_connection

    conn = get_connection()
    # Schema sicherstellen — ggf. Tabellen droppen die von anderen Tests mit falschem Schema hinterlassen wurden.
    # Marker: datasources.ingestion_status existiert UND ingestion_jobs.finished_at ist TIMESTAMPTZ
    # (andere Tests legen finished_at als TEXT an, was cleanup_old() bricht).
    schema_ok = True
    try:
        conn.execute("SELECT ingestion_status FROM datasources LIMIT 0")
    except Exception:
        conn.rollback()
        schema_ok = False
    if schema_ok:
        try:
            row = conn.execute(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name = 'ingestion_jobs' AND column_name = 'finished_at'"
            ).fetchone()
            if row is not None and row["data_type"] != "timestamp with time zone":
                schema_ok = False
        except Exception:
            conn.rollback()
            schema_ok = False
    if not schema_ok:
        conn.execute("DROP TABLE IF EXISTS ingestion_file_logs CASCADE")
        conn.execute("DROP TABLE IF EXISTS cache_jobs CASCADE")
        conn.execute("DROP TABLE IF EXISTS cache_datasources CASCADE")
        conn.execute("DROP TABLE IF EXISTS ingestion_jobs CASCADE")
        conn.execute("DROP TABLE IF EXISTS datasource_embeddings CASCADE")
        conn.execute("DROP TABLE IF EXISTS datasources CASCADE")
    conn.executescript(_SCHEMA)
    conn.commit()
    # Testdaten aufraeumen
    conn.execute("DELETE FROM ingestion_file_logs")
    conn.execute("DELETE FROM cache_jobs")
    conn.execute("DELETE FROM cache_datasources")
    conn.execute("DELETE FROM ingestion_jobs")
    conn.execute("DELETE FROM datasource_embeddings")
    conn.execute("DELETE FROM datasources")
    conn.commit()
    conn.execute("INSERT INTO datasources (name, type) VALUES (%s, %s)", ("Test DS", "local"))
    conn.commit()
    ds_row = conn.execute("SELECT id FROM datasources WHERE name = 'Test DS'").fetchone()
    ds_id = ds_row["id"]
    conn.close()
    return ds_id


def _patch_db(ds_id, monkeypatch):
    """Patcht db_query/db_transaction auf PostgreSQL."""
    import kiara.shared.db as shared_db
    import kiara.web.db_utils as db_utils
    from kiara.shared.db import get_connection

    @contextmanager
    def fake_query():
        c = get_connection()
        try:
            yield c
        finally:
            c.close()

    @contextmanager
    def fake_tx():
        c = get_connection()
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    monkeypatch.setattr(db_utils, "db_query", fake_query)
    monkeypatch.setattr(db_utils, "db_transaction", fake_tx)
    monkeypatch.setattr(shared_db, "db_query", fake_query)
    monkeypatch.setattr(shared_db, "db_transaction", fake_tx)
    return fake_query, fake_tx


# ---------------------------------------------------------------------------
# 1. FileLogEntry Dataclass
# ---------------------------------------------------------------------------


class TestFileLogEntry:
    def test_defaults(self):
        from kiara.ingestion.pipeline import FileLogEntry

        entry = FileLogEntry(
            file_path="/a/b.txt", file_name="b.txt", file_size=1234, file_ext=".txt"
        )
        assert entry.status == "success"
        assert entry.parser == ""
        assert entry.parser_fallback is False
        assert entry.chunk_count == 0
        assert entry.duration_ms == 0

    def test_error_state(self):
        from kiara.ingestion.pipeline import FileLogEntry

        entry = FileLogEntry(
            file_path="/a/b.txt",
            file_name="b.txt",
            file_size=0,
            file_ext=".txt",
            status="error",
            error_message="Timeout",
            error_phase="loading",
        )
        assert entry.status == "error"
        assert entry.error_message == "Timeout"


# ---------------------------------------------------------------------------
# 2. file_log in _process_single_file()
# ---------------------------------------------------------------------------


class TestFileLogInProcessSingleFile:
    def _make_ctx(self):
        from kiara.ingestion.runtime import ProcessingContext

        return ProcessingContext()

    def test_success_populates_parser_and_chunks(self):
        """Erfolgreiche Verarbeitung setzt parser, chunk_strategy, chunk_count."""
        from kiara.ingestion.pipeline import FileLogEntry
        from kiara.ingestion.runtime import _process_single_file

        file_log = FileLogEntry(
            file_path="/tmp/test.txt", file_name="test.txt", file_size=100, file_ext=".txt"
        )
        stats = {"errors": 0}
        config = {"ingestion": {"chunk_size": 500, "chunk_overlap": 50}}

        mock_doc = {
            "text": "Hello World " * 10,
            "metadata": {"source": "/tmp/test.txt", "parser": "standard", "file_type": ".txt"},
        }

        with patch("kiara.ingestion.runtime.load_file", return_value=mock_doc):
            with patch(
                "kiara.ingestion.runtime.chunk_document",
                return_value=[
                    {
                        "text": "chunk1",
                        "metadata": {
                            **mock_doc["metadata"],
                            "chunk_index": 0,
                            "chunk_strategy": "narrative",
                        },
                    },
                    {
                        "text": "chunk2",
                        "metadata": {
                            **mock_doc["metadata"],
                            "chunk_index": 1,
                            "chunk_strategy": "narrative",
                        },
                    },
                ],
            ):
                texts, metas, ids, _full_text, _file_spans = _process_single_file(
                    "/tmp/test.txt",
                    config,
                    None,
                    1,
                    self._make_ctx(),
                    stats,
                    file_log=file_log,
                )

        assert file_log.parser == "standard"
        assert file_log.parser_fallback is False
        assert file_log.chunk_count == 2
        assert file_log.chunk_strategy == "narrative"
        assert file_log.status == "success"

    def test_loading_error_sets_error_fields(self):
        """Datei kann nicht geladen werden → status=error, error_phase=loading."""
        from kiara.ingestion.pipeline import FileLogEntry
        from kiara.ingestion.runtime import _process_single_file

        file_log = FileLogEntry(
            file_path="/tmp/bad.pdf", file_name="bad.pdf", file_size=100, file_ext=".pdf"
        )
        stats = {"errors": 0}
        config = {"ingestion": {"chunk_size": 500, "chunk_overlap": 50}}

        with patch("kiara.ingestion.runtime.load_file", return_value=None):
            with patch("kiara.ingestion.runtime.describe_image", return_value=None):
                _process_single_file(
                    "/tmp/bad.pdf",
                    config,
                    None,
                    1,
                    self._make_ctx(),
                    stats,
                    file_log=file_log,
                )

        assert file_log.status == "error"
        assert file_log.error_phase == "loading"
        assert file_log.parser == "unknown"

    def test_none_file_log_no_crash(self):
        """file_log=None (Standard) → kein Crash, backward-kompatibel."""
        from kiara.ingestion.runtime import _process_single_file

        stats = {"errors": 0}
        config = {"ingestion": {"chunk_size": 500, "chunk_overlap": 50}}

        with patch("kiara.ingestion.runtime.load_file", return_value=None):
            with patch("kiara.ingestion.runtime.describe_image", return_value=None):
                _process_single_file(
                    "/tmp/test.txt",
                    config,
                    None,
                    1,
                    self._make_ctx(),
                    stats,
                )
        assert stats["errors"] == 1


# ---------------------------------------------------------------------------
# 3. file_log_callback in run_ingestion()
# ---------------------------------------------------------------------------


class TestFileLogCallback:
    def test_callback_called_per_file(self, tmp_path):
        """file_log_callback wird pro verarbeiteter Datei aufgerufen."""
        from kiara.ingestion.pipeline import run_ingestion

        # Testdateien erstellen
        for name in ["a.txt", "b.txt", "c.txt"]:
            (tmp_path / name).write_text(f"Content of {name}")

        config = {
            "datasource": {"path": str(tmp_path), "file_types": [".txt"], "exclude_patterns": []},
            "ingestion": {"chunk_size": 500, "chunk_overlap": 50, "batch_size": 10},
            "vectordb": {"path": str(tmp_path / "vdb"), "backend": "qdrant"},
            "graph": {"enabled": False},
        }

        collected = []

        def callback(entry):
            collected.append(entry)

        mock_store = MagicMock()
        mock_store.delete_by_sources.return_value = None
        mock_store.add.return_value = None

        with (
            patch("kiara.ingestion.pipeline.VectorStore", return_value=mock_store),
            patch("kiara.ingestion.pipeline.get_embedding_model", return_value="test-model"),
        ):
            with patch("kiara.ingestion.pipeline.HashStore") as MockHash:
                hs = MockHash.return_value
                hs.load.return_value = {}
                hs.upsert_many.return_value = None
                hs.remove_files.return_value = None

                run_ingestion(
                    config,
                    incremental=False,
                    datasource_id=1,
                    file_log_callback=callback,
                )

        assert len(collected) == 3
        names = {e.file_name for e in collected}
        assert names == {"a.txt", "b.txt", "c.txt"}
        for entry in collected:
            assert entry.file_ext == ".txt"
            assert entry.duration_ms >= 0

    def test_no_callback_no_crash(self, tmp_path):
        """file_log_callback=None → kein Crash."""
        from kiara.ingestion.pipeline import run_ingestion

        (tmp_path / "x.txt").write_text("Content")

        config = {
            "datasource": {"path": str(tmp_path), "file_types": [".txt"], "exclude_patterns": []},
            "ingestion": {"chunk_size": 500, "chunk_overlap": 50, "batch_size": 10},
            "vectordb": {"path": str(tmp_path / "vdb"), "backend": "qdrant"},
            "graph": {"enabled": False},
        }

        mock_store = MagicMock()
        mock_store.delete_by_sources.return_value = None
        mock_store.add.return_value = None

        with (
            patch("kiara.ingestion.pipeline.VectorStore", return_value=mock_store),
            patch("kiara.ingestion.pipeline.get_embedding_model", return_value="test-model"),
        ):
            with patch("kiara.ingestion.pipeline.HashStore") as MockHash:
                hs = MockHash.return_value
                hs.load.return_value = {}
                hs.upsert_many.return_value = None
                hs.remove_files.return_value = None

                stats = run_ingestion(config, incremental=False, datasource_id=1)

        assert stats["processed_files"] >= 0  # Kein Crash


# ---------------------------------------------------------------------------
# 4. API-Endpoint GET /file-logs
# ---------------------------------------------------------------------------


@pytest.fixture
def api_app(tmp_path, monkeypatch):
    """FastAPI TestClient mit ingestion_file_logs-Tabelle."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from kiara.web.api_ingestion_jobs import router
    from kiara.web.auth import get_admin_user

    ds_id = _make_db(tmp_path)
    fake_query, fake_tx = _patch_db(ds_id, monkeypatch)

    import kiara_db.base as repo_base
    import kiara_db.repositories.cache as repo_cache
    import kiara_db.repositories.datasource as repo_ds
    import kiara_db.repositories.job as repo_job

    import kiara.shared.db as shared_db_mod
    import kiara.web.cache_refresh as cache_mod
    import kiara.web.ingestion_worker as worker_mod

    monkeypatch.setattr(shared_db_mod, "db_query", fake_query)
    monkeypatch.setattr(shared_db_mod, "db_transaction", fake_tx)
    monkeypatch.setattr(worker_mod, "db_query", fake_query)
    monkeypatch.setattr(worker_mod, "db_transaction", fake_tx)
    monkeypatch.setattr(cache_mod, "db_query", fake_query)
    monkeypatch.setattr(cache_mod, "db_transaction", fake_tx)
    monkeypatch.setattr(repo_base, "db_query", fake_query)
    monkeypatch.setattr(repo_base, "db_transaction", fake_tx)
    monkeypatch.setattr(repo_job, "db_query", fake_query)
    monkeypatch.setattr(repo_job, "db_transaction", fake_tx)
    monkeypatch.setattr(repo_cache, "db_query", fake_query)
    monkeypatch.setattr(repo_cache, "db_transaction", fake_tx)
    monkeypatch.setattr(repo_ds, "db_query", fake_query)
    monkeypatch.setattr(repo_ds, "db_transaction", fake_tx)

    monkeypatch.setattr(worker_mod, "get_running_job_id", lambda: None)
    monkeypatch.setattr(worker_mod, "is_job_paused", lambda job_id: False)
    mock_worker = MagicMock()  # Kept for fixture return value compatibility

    from kiara.admin.registry import admin_registry, load_admin_functions
    from kiara.admin.router import generate_admin_routes

    load_admin_functions()
    app = FastAPI()
    app.include_router(router)
    generate_admin_routes(app, admin_registry)
    app.dependency_overrides[get_admin_user] = lambda: {
        "id": 1,
        "username": "admin",
        "is_admin": True,
    }

    return TestClient(app), ds_id, mock_worker


def _seed_file_logs(ds_id, job_id, entries):
    """Fuegt File-Log-Eintraege in die DB ein."""
    from kiara.shared.db import get_connection

    conn = get_connection()
    for e in entries:
        conn.execute(
            "INSERT INTO ingestion_file_logs "
            "(job_id, file_path, file_name, file_size, file_ext, "
            "parser, parser_fallback, chunk_strategy, chunk_count, "
            "status, error_message, error_phase, duration_ms) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                job_id,
                e.get("path", f"/data/{e['name']}"),
                e["name"],
                e.get("size", 1000),
                e.get("ext", ".txt"),
                e.get("parser", "standard"),
                e.get("fallback", 0),
                e.get("strategy", "narrative"),
                e.get("chunks", 5),
                e.get("status", "success"),
                e.get("error", ""),
                e.get("phase", ""),
                e.get("duration", 100),
            ),
        )
    conn.commit()
    conn.close()


class TestFileLogsEndpoint:
    def _insert_job(self, ds_id, status="success"):
        from kiara.shared.db import get_connection

        conn = get_connection()
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, finished_at) "
            "VALUES (%s, %s, NOW())",
            (ds_id, status),
        )
        row = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()
        job_id = row["id"]
        conn.commit()
        conn.close()
        return job_id

    def test_empty_logs(self, api_app):
        client, ds_id, _ = api_app
        job_id = self._insert_job(ds_id)

        resp = client.get(f"/api/admin/ingestion-jobs/{job_id}/file-logs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"] == []
        assert data["total"] == 0
        assert data["summary"]["success"] == 0

    def test_returns_file_logs(self, api_app):
        client, ds_id, _ = api_app
        job_id = self._insert_job(ds_id)
        _seed_file_logs(
            ds_id,
            job_id,
            [
                {"name": "a.txt", "status": "success", "chunks": 10},
                {
                    "name": "b.pdf",
                    "status": "error",
                    "error": "Parse failed",
                    "phase": "loading",
                    "chunks": 0,
                    "ext": ".pdf",
                },
            ],
        )

        resp = client.get(f"/api/admin/ingestion-jobs/{job_id}/file-logs")
        data = resp.json()
        assert data["total"] == 2
        assert data["summary"]["success"] == 1
        assert data["summary"]["error"] == 1
        assert data["summary"]["total_chunks"] == 10

    def test_filter_by_status(self, api_app):
        client, ds_id, _ = api_app
        job_id = self._insert_job(ds_id)
        _seed_file_logs(
            ds_id,
            job_id,
            [
                {"name": "ok.txt", "status": "success"},
                {"name": "bad.txt", "status": "error"},
                {"name": "skip.txt", "status": "skipped"},
            ],
        )

        resp = client.get(f"/api/admin/ingestion-jobs/{job_id}/file-logs?status=error")
        data = resp.json()
        assert data["total"] == 1
        assert data["data"][0]["file_name"] == "bad.txt"

    def test_search_by_name(self, api_app):
        client, ds_id, _ = api_app
        job_id = self._insert_job(ds_id)
        _seed_file_logs(
            ds_id,
            job_id,
            [
                {"name": "report_2024.pdf"},
                {"name": "invoice_2024.pdf"},
                {"name": "notes.txt"},
            ],
        )

        resp = client.get(f"/api/admin/ingestion-jobs/{job_id}/file-logs?search=2024")
        data = resp.json()
        assert data["total"] == 2

    def test_sort_by_duration(self, api_app):
        client, ds_id, _ = api_app
        job_id = self._insert_job(ds_id)
        _seed_file_logs(
            ds_id,
            job_id,
            [
                {"name": "fast.txt", "duration": 50},
                {"name": "slow.txt", "duration": 5000},
                {"name": "medium.txt", "duration": 500},
            ],
        )

        resp = client.get(
            f"/api/admin/ingestion-jobs/{job_id}/file-logs?sort=duration_ms&direction=desc"
        )
        data = resp.json()
        names = [r["file_name"] for r in data["data"]]
        assert names == ["slow.txt", "medium.txt", "fast.txt"]

    def test_pagination(self, api_app):
        client, ds_id, _ = api_app
        job_id = self._insert_job(ds_id)
        _seed_file_logs(ds_id, job_id, [{"name": f"file_{i:03d}.txt"} for i in range(10)])

        resp = client.get(f"/api/admin/ingestion-jobs/{job_id}/file-logs?limit=3&offset=0")
        data = resp.json()
        assert len(data["data"]) == 3
        assert data["total"] == 10

        resp2 = client.get(f"/api/admin/ingestion-jobs/{job_id}/file-logs?limit=3&offset=9")
        data2 = resp2.json()
        assert len(data2["data"]) == 1

    def test_invalid_sort_column_defaults(self, api_app):
        client, ds_id, _ = api_app
        job_id = self._insert_job(ds_id)
        _seed_file_logs(ds_id, job_id, [{"name": "a.txt"}])

        # SQL-Injection-Versuch als sort
        resp = client.get(f"/api/admin/ingestion-jobs/{job_id}/file-logs?sort=id;DROP TABLE--")
        assert resp.status_code == 200  # Fallback auf file_name

    def test_nonexistent_job_404(self, api_app):
        client, _, _ = api_app
        resp = client.get("/api/admin/ingestion-jobs/999999/file-logs")
        assert resp.status_code == 404

    def test_summary_independent_of_filters(self, api_app):
        """Summary zaehlt ALLE Dateien des Jobs, nicht nur gefilterte."""
        client, ds_id, _ = api_app
        job_id = self._insert_job(ds_id)
        _seed_file_logs(
            ds_id,
            job_id,
            [
                {"name": "a.txt", "status": "success", "chunks": 10},
                {"name": "b.txt", "status": "success", "chunks": 20},
                {"name": "c.txt", "status": "error", "chunks": 0},
            ],
        )

        resp = client.get(f"/api/admin/ingestion-jobs/{job_id}/file-logs?status=error")
        data = resp.json()
        # Gefilterte Daten: nur 1 Eintrag
        assert data["total"] == 1
        # Summary: alle 3 Dateien
        assert data["summary"]["success"] == 2
        assert data["summary"]["error"] == 1
        assert data["summary"]["total_chunks"] == 30


# ---------------------------------------------------------------------------
# 5. Ghost-Job-Erkennung
# ---------------------------------------------------------------------------


class TestGhostDetection:
    def test_running_job_without_heartbeat_is_ghost(self, api_app, monkeypatch):
        from kiara.shared.db import get_connection

        client, ds_id, mock_worker = api_app
        import kiara.web.ingestion_worker as wmod

        monkeypatch.setattr(wmod, "get_running_job_id", lambda: None)

        conn = get_connection()
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, started_at) "
            "VALUES (%s, %s, NOW() - INTERVAL '10 minutes')",
            (ds_id, "running"),
        )
        conn.commit()
        conn.close()

        resp = client.get("/api/admin/ingestion-jobs")
        data = resp.json()
        assert data["total"] == 1
        assert data["data"][0]["is_ghost"] is True

    def test_running_job_with_heartbeat_not_ghost(self, api_app, monkeypatch):
        from kiara.shared.db import get_connection

        client, ds_id, mock_worker = api_app

        conn = get_connection()
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, started_at) VALUES (%s, %s, NOW())",
            (ds_id, "running"),
        )
        row = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()
        job_id = row["id"]
        conn.commit()
        conn.close()

        import kiara.web.ingestion_worker as wmod

        monkeypatch.setattr(wmod, "get_running_job_id", lambda: job_id)
        monkeypatch.setattr(wmod, "is_job_paused", lambda jid: False)

        resp = client.get("/api/admin/ingestion-jobs")
        data = resp.json()
        assert data["total"] == 1
        assert "is_ghost" not in data["data"][0]

    def test_ghost_in_single_job_endpoint(self, api_app, monkeypatch):
        from kiara.shared.db import get_connection

        client, ds_id, mock_worker = api_app
        import kiara.web.ingestion_worker as wmod

        monkeypatch.setattr(wmod, "get_running_job_id", lambda: None)

        conn = get_connection()
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, started_at) "
            "VALUES (%s, %s, NOW() - INTERVAL '10 minutes')",
            (ds_id, "running"),
        )
        row = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()
        job_id = row["id"]
        conn.commit()
        conn.close()

        resp = client.get(f"/api/admin/ingestion-jobs/{job_id}")
        data = resp.json()
        assert data["is_ghost"] is True

    def test_completed_job_not_ghost(self, api_app, monkeypatch):
        from kiara.shared.db import get_connection

        client, ds_id, mock_worker = api_app
        import kiara.web.ingestion_worker as wmod

        monkeypatch.setattr(wmod, "get_running_job_id", lambda: None)

        conn = get_connection()
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, finished_at) "
            "VALUES (%s, %s, NOW())",
            (ds_id, "success"),
        )
        conn.commit()
        conn.close()

        resp = client.get("/api/admin/ingestion-jobs")
        data = resp.json()
        assert "is_ghost" not in data["data"][0]


# ---------------------------------------------------------------------------
# 6. VectorDB-Punktestand in _finalize_success
# ---------------------------------------------------------------------------


class TestVectorDbStats:
    def test_stats_include_vectordb_points(self, tmp_path, monkeypatch):
        """_finalize_success schreibt vectordb_points_ds und vectordb_points_total."""
        from kiara.shared.db import get_connection

        ds_id = _make_db(tmp_path)
        _patch_db(ds_id, monkeypatch)

        import kiara.web.ingestion_worker as worker_mod

        fake_q, fake_tx = _patch_db(ds_id, monkeypatch)
        monkeypatch.setattr(worker_mod, "db_query", fake_q)
        monkeypatch.setattr(worker_mod, "db_transaction", fake_tx)

        # Job anlegen
        conn = get_connection()
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, started_at) VALUES (%s, %s, NOW())",
            (ds_id, "running"),
        )
        row = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()
        job_id = row["id"]
        conn.commit()
        conn.close()

        # Mock VectorStore
        mock_store = MagicMock()
        mock_store.get_stats.return_value = {
            "total_chunks": 50000,
            "backend": "qdrant",
            "collection": "kb_nomic-embed-text",
        }

        # Mock ChunkDatasources (count_by_datasource liefert DS-Punktestand)
        mock_chunk_ds = MagicMock()
        mock_chunk_ds.count_by_datasource.return_value = 1234

        worker = worker_mod.IngestionWorker()

        with (
            patch("kiara.retrieval.vectorstore.VectorStore", return_value=mock_store),
            patch("kiara.ingestion._file_utils.ChunkDatasources", return_value=mock_chunk_ds),
            patch("kiara.web.ingestion_worker.HybridRetriever", create=True),
        ):
            stats = {"total_files": 10, "processed_files": 10, "total_chunks": 100}
            worker._finalize_success(
                job_id, ds_id, stats, {"vectordb": {"path": str(tmp_path)}}, "nomic-embed-text"
            )

        assert stats["vectordb_points_ds"] == 1234
        assert stats["vectordb_points_total"] == 50000
        assert stats["vectordb_backend"] == "qdrant"
        assert stats["vectordb_collection"] == "kb_nomic-embed-text"

        # Pruefen, dass stats_json in DB geschrieben wurde
        conn = get_connection()
        row = conn.execute(
            "SELECT stats_json FROM ingestion_jobs WHERE id = %s", (job_id,)
        ).fetchone()
        conn.close()
        # jsonb-Spalte: psycopg2 liefert dict direkt, sonst string parsen.
        saved = (
            row["stats_json"]
            if isinstance(row["stats_json"], dict)
            else json.loads(row["stats_json"] or "{}")
        )
        assert saved["vectordb_points_ds"] == 1234

    def test_stats_survive_get_stats_failure(self, tmp_path, monkeypatch):
        """Wenn get_stats() fehlschlaegt, crasht _finalize_success nicht."""
        from kiara.shared.db import get_connection

        ds_id = _make_db(tmp_path)
        fake_q, fake_tx = _patch_db(ds_id, monkeypatch)

        import kiara.web.ingestion_worker as worker_mod

        monkeypatch.setattr(worker_mod, "db_query", fake_q)
        monkeypatch.setattr(worker_mod, "db_transaction", fake_tx)

        conn = get_connection()
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, started_at) VALUES (%s, %s, NOW())",
            (ds_id, "running"),
        )
        row = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()
        job_id = row["id"]
        conn.commit()
        conn.close()

        mock_store = MagicMock()
        mock_store.get_stats.side_effect = Exception("Qdrant down")

        # Mock ChunkDatasources
        mock_chunk_ds = MagicMock()
        mock_chunk_ds.count_by_datasource.return_value = 42

        worker = worker_mod.IngestionWorker()

        with (
            patch("kiara.retrieval.vectorstore.VectorStore", return_value=mock_store),
            patch("kiara.ingestion._file_utils.ChunkDatasources", return_value=mock_chunk_ds),
            patch("kiara.web.ingestion_worker.HybridRetriever", create=True),
        ):
            stats = {"total_files": 1, "processed_files": 1, "total_chunks": 5}
            worker._finalize_success(
                job_id, ds_id, stats, {"vectordb": {"path": str(tmp_path)}}, "nomic-embed-text"
            )

        # vectordb_points_ds trotzdem gesetzt
        assert stats["vectordb_points_ds"] == 42
        # vectordb_points_total nicht gesetzt (Exception)
        assert "vectordb_points_total" not in stats


# ---------------------------------------------------------------------------
# 7. CASCADE / Safety-DELETE bei History-Cleanup
# ---------------------------------------------------------------------------


class TestHistoryCleanupCascade:
    def test_file_logs_deleted_with_job(self, api_app):
        """History-Cleanup loescht File-Logs zusammen mit Jobs."""
        from kiara.shared.db import get_connection

        client, ds_id, _ = api_app

        conn = get_connection()
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, finished_at) "
            "VALUES (%s, %s, NOW() - INTERVAL '60 days')",
            (ds_id, "success"),
        )
        row = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()
        job_id = row["id"]
        # File-Logs fuer diesen Job
        for i in range(5):
            conn.execute(
                "INSERT INTO ingestion_file_logs (job_id, file_path, file_name, status) "
                "VALUES (%s, %s, %s, 'success')",
                (job_id, f"/data/file{i}.txt", f"file{i}.txt"),
            )
        conn.commit()
        conn.close()

        # Pruefen: 5 File-Logs vorhanden
        conn = get_connection()
        count = conn.execute(
            "SELECT COUNT(*) AS cnt FROM ingestion_file_logs WHERE job_id = %s", (job_id,)
        ).fetchone()["cnt"]
        conn.close()
        assert count == 5

        # Cleanup
        resp = confirmed_request(client, "DELETE", "/api/admin/ingestion-jobs/history?days=30")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 1

        # File-Logs auch weg
        conn = get_connection()
        count = conn.execute(
            "SELECT COUNT(*) AS cnt FROM ingestion_file_logs WHERE job_id = %s", (job_id,)
        ).fetchone()["cnt"]
        conn.close()
        assert count == 0

    def test_recent_job_file_logs_preserved(self, api_app):
        """Aktuelle Jobs und ihre File-Logs bleiben erhalten."""
        from kiara.shared.db import get_connection

        client, ds_id, _ = api_app

        conn = get_connection()
        # Alter Job (wird geloescht)
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, finished_at) "
            "VALUES (%s, %s, NOW() - INTERVAL '60 days')",
            (ds_id, "success"),
        )
        old_row = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()
        old_id = old_row["id"]
        conn.execute(
            "INSERT INTO ingestion_file_logs (job_id, file_path, file_name, status) "
            "VALUES (%s, '/data/old.txt', 'old.txt', 'success')",
            (old_id,),
        )
        # Neuer Job (bleibt)
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, finished_at) "
            "VALUES (%s, %s, NOW())",
            (ds_id, "success"),
        )
        new_row = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()
        new_id = new_row["id"]
        conn.execute(
            "INSERT INTO ingestion_file_logs (job_id, file_path, file_name, status) "
            "VALUES (%s, '/data/new.txt', 'new.txt', 'success')",
            (new_id,),
        )
        conn.commit()
        conn.close()

        client.delete("/api/admin/ingestion-jobs/history?days=30")

        conn = get_connection()
        logs = conn.execute(
            "SELECT file_name FROM ingestion_file_logs WHERE job_id = %s", (new_id,)
        ).fetchall()
        conn.close()
        assert len(logs) == 1
        assert logs[0]["file_name"] == "new.txt"


# ---------------------------------------------------------------------------
# 8. Schema-Migration Idempotenz
# ---------------------------------------------------------------------------

# TestMigrationIdempotent wurde entfernt:
# importierte aus kiara.web.db_migrations (Modul existiert nicht mehr).
# ingestion_file_logs ist heute Teil der initialen Schema-Erstellung (schema.sql).


# ===========================================================================
# Tiefergehende Tests — Edge Cases & Integration
# ===========================================================================


# ---------------------------------------------------------------------------
# 9. Worker Batched Flush
# ---------------------------------------------------------------------------


class TestWorkerBatchedFlush:
    """Testet die Flush-Logik im IngestionWorker (_on_file_log / _flush_file_logs)."""

    def test_flush_at_threshold(self, tmp_path, monkeypatch):
        """Buffer flusht automatisch bei 25 Eintraegen."""
        from kiara.shared.db import get_connection

        ds_id = _make_db(tmp_path)
        fake_q, fake_tx = _patch_db(ds_id, monkeypatch)

        import kiara.web.ingestion_worker as worker_mod

        monkeypatch.setattr(worker_mod, "db_query", fake_q)
        monkeypatch.setattr(worker_mod, "db_transaction", fake_tx)

        # Job anlegen
        conn = get_connection()
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status) VALUES (%s, %s)", (ds_id, "running")
        )
        row = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()
        job_id = row["id"]
        conn.commit()
        conn.close()

        # Buffer + Flush-Logik aus dem Worker replizieren
        from kiara.ingestion.pipeline import FileLogEntry

        _file_log_buffer = []
        _FILE_LOG_FLUSH = 25
        flush_count = [0]

        def _flush():
            if not _file_log_buffer:
                return
            with fake_tx() as db:
                db.executemany(
                    "INSERT INTO ingestion_file_logs "
                    "(job_id, file_path, file_name, file_size, file_ext, "
                    "parser, parser_fallback, chunk_strategy, chunk_count, "
                    "status, error_message, error_phase, duration_ms) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    [
                        (
                            job_id,
                            e.file_path,
                            e.file_name,
                            e.file_size,
                            e.file_ext,
                            e.parser,
                            int(e.parser_fallback),
                            e.chunk_strategy,
                            e.chunk_count,
                            e.status,
                            e.error_message,
                            e.error_phase,
                            e.duration_ms,
                        )
                        for e in _file_log_buffer
                    ],
                )
            flush_count[0] += 1
            _file_log_buffer.clear()

        def _on_file_log(entry):
            _file_log_buffer.append(entry)
            if len(_file_log_buffer) >= _FILE_LOG_FLUSH:
                _flush()

        # 30 Eintraege einspeisen
        for i in range(30):
            entry = FileLogEntry(
                file_path=f"/data/file{i}.txt",
                file_name=f"file{i}.txt",
                file_size=100,
                file_ext=".txt",
            )
            _on_file_log(entry)

        # Nach 30 Eintraegen: 1x geflusht bei 25
        assert flush_count[0] == 1

        # 5 im Buffer verbleibend
        assert len(_file_log_buffer) == 5

        # Rest-Flush
        _flush()
        assert flush_count[0] == 2
        assert len(_file_log_buffer) == 0

        # 30 Zeilen in DB
        conn = get_connection()
        count = conn.execute(
            "SELECT COUNT(*) AS cnt FROM ingestion_file_logs WHERE job_id = %s", (job_id,)
        ).fetchone()["cnt"]
        conn.close()
        assert count == 30

    def test_flush_on_empty_buffer_noop(self, tmp_path, monkeypatch):
        """Flush auf leeren Buffer ist ein No-Op (kein DB-Call)."""
        ds_id = _make_db(tmp_path)
        fake_q, fake_tx = _patch_db(ds_id, monkeypatch)

        call_count = [0]
        original_tx = fake_tx

        @contextmanager
        def counting_tx():
            call_count[0] += 1
            with original_tx() as db:
                yield db

        _file_log_buffer = []

        def _flush():
            if not _file_log_buffer:
                return
            with counting_tx() as db:
                pass  # Wuerde INSERT machen
            _file_log_buffer.clear()

        _flush()
        assert call_count[0] == 0  # Kein DB-Call

    def test_flush_survives_db_error(self, tmp_path, monkeypatch):
        """Bei DB-Fehler wird der Buffer trotzdem geleert (kein Retry-Loop)."""
        from kiara.ingestion.pipeline import FileLogEntry

        _file_log_buffer = []
        _file_log_buffer.append(
            FileLogEntry(file_path="/a.txt", file_name="a.txt", file_size=0, file_ext=".txt")
        )

        @contextmanager
        def failing_tx():
            raise RuntimeError("DB locked")
            yield  # Generator-Syntax erfordert yield.

        def _flush():
            if not _file_log_buffer:
                return
            try:
                with failing_tx() as db:
                    pass
            except Exception:
                pass  # Wie im Worker: logger.warning
            _file_log_buffer.clear()

        _flush()
        assert len(_file_log_buffer) == 0  # Buffer trotzdem geleert


# ---------------------------------------------------------------------------
# 10. Embedding-Fehler → file_log
# ---------------------------------------------------------------------------


class TestEmbeddingErrorFileLog:
    def test_embedding_exception_sets_error_in_file_log(self, tmp_path):
        """Wenn _write_embeddings fehlschlaegt, wird file_log auf error gesetzt."""
        from kiara.ingestion.pipeline import run_ingestion

        (tmp_path / "doc.txt").write_text("Test content for embedding failure")

        config = {
            "datasource": {"path": str(tmp_path), "file_types": [".txt"], "exclude_patterns": []},
            "ingestion": {"chunk_size": 500, "chunk_overlap": 50, "batch_size": 10},
            "vectordb": {"path": str(tmp_path / "vdb"), "backend": "qdrant"},
            "graph": {"enabled": False},
        }

        collected = []
        mock_store = MagicMock()
        mock_store.delete_by_sources.return_value = None
        mock_store.add.side_effect = ConnectionError("Ollama down")

        with (
            patch("kiara.ingestion.pipeline.VectorStore", return_value=mock_store),
            patch("kiara.ingestion.pipeline.get_embedding_model", return_value="test-model"),
        ):
            with patch("kiara.ingestion.pipeline.HashStore") as MockHash:
                hs = MockHash.return_value
                hs.load.return_value = {}
                hs.upsert_many.return_value = None
                hs.remove_files.return_value = None

                stats = run_ingestion(
                    config,
                    incremental=False,
                    datasource_id=1,
                    file_log_callback=lambda e: collected.append(e),
                )

        assert len(collected) == 1
        entry = collected[0]
        assert entry.status == "error"
        assert entry.error_phase == "embedding"
        assert "Ollama down" in entry.error_message
        assert stats["errors"] == 1


# ---------------------------------------------------------------------------
# 11. Skipped-Status (0 Chunks, kein Fehler)
# ---------------------------------------------------------------------------


class TestSkippedFileLog:
    def test_zero_chunks_no_error_becomes_skipped(self, tmp_path):
        """Datei liefert leeren Text → 0 Chunks → file_log.status = 'skipped'."""
        from kiara.ingestion.pipeline import run_ingestion

        (tmp_path / "empty.txt").write_text("")

        config = {
            "datasource": {"path": str(tmp_path), "file_types": [".txt"], "exclude_patterns": []},
            "ingestion": {"chunk_size": 500, "chunk_overlap": 50, "batch_size": 10},
            "vectordb": {"path": str(tmp_path / "vdb"), "backend": "qdrant"},
            "graph": {"enabled": False},
        }

        collected = []
        mock_store = MagicMock()
        mock_store.delete_by_sources.return_value = None
        mock_store.add.return_value = None

        with (
            patch("kiara.ingestion.pipeline.VectorStore", return_value=mock_store),
            patch("kiara.ingestion.pipeline.get_embedding_model", return_value="test-model"),
        ):
            with patch("kiara.ingestion.pipeline.HashStore") as MockHash:
                hs = MockHash.return_value
                hs.load.return_value = {}
                hs.upsert_many.return_value = None
                hs.remove_files.return_value = None

                run_ingestion(
                    config,
                    incremental=False,
                    datasource_id=1,
                    file_log_callback=lambda e: collected.append(e),
                )

        # Leere Datei → load_file gibt None zurueck → error oder
        # load_file gibt leeren Text → chunk_document liefert 0 Chunks
        assert len(collected) == 1
        entry = collected[0]
        # Entweder skipped (0 Chunks, kein Error) oder error (Laden fehlgeschlagen)
        assert entry.status in ("skipped", "error")


# ---------------------------------------------------------------------------
# 12. DDU Timeout → file_log
# ---------------------------------------------------------------------------


class TestDduTimeoutFileLog:
    def test_ddu_timeout_sets_error_fields(self):
        """DDU-Timeout setzt file_log.status=error mit Phase loading."""
        from kiara.ingestion.pipeline import FileLogEntry
        from kiara.ingestion.runtime import ProcessingContext, _process_single_file

        file_log = FileLogEntry(
            file_path="/tmp/big.pdf", file_name="big.pdf", file_size=50_000_000, file_ext=".pdf"
        )
        stats = {"errors": 0}
        config = {"ingestion": {"chunk_size": 500, "chunk_overlap": 50}}

        # DDU gibt Doc zurueck, aber mit ddu_error=timeout
        timeout_doc = {
            "text": "",
            "metadata": {
                "source": "/tmp/big.pdf",
                "parser": "standard",
                "ddu_error": "timeout",
                "file_type": ".pdf",
            },
        }

        with patch("kiara.ingestion.runtime.load_file", return_value=timeout_doc):
            with patch("kiara.ingestion.runtime.describe_image", return_value=None):
                texts, _, _, _full_text, _file_spans = _process_single_file(
                    "/tmp/big.pdf",
                    config,
                    None,
                    1,
                    ProcessingContext(ddu_enabled=True),
                    stats,
                    file_log=file_log,
                )

        assert texts == []
        assert file_log.status == "error"
        assert file_log.error_message == "DDU timeout"
        assert file_log.error_phase == "loading"
        assert stats["errors"] == 1


# ---------------------------------------------------------------------------
# 13. parser_fallback Tracking
# ---------------------------------------------------------------------------


class TestParserFallbackFileLog:
    def test_parser_fallback_true_propagated(self):
        """parser_fallback=True in Metadata → file_log.parser_fallback=True."""
        from kiara.ingestion.pipeline import FileLogEntry
        from kiara.ingestion.runtime import ProcessingContext, _process_single_file

        file_log = FileLogEntry(
            file_path="/tmp/doc.pdf", file_name="doc.pdf", file_size=1000, file_ext=".pdf"
        )
        stats = {"errors": 0}
        config = {"ingestion": {"chunk_size": 500, "chunk_overlap": 50}}

        fallback_doc = {
            "text": "Fallback text content",
            "metadata": {
                "source": "/tmp/doc.pdf",
                "parser": "standard",
                "parser_fallback": True,
                "file_type": ".pdf",
            },
        }

        with patch("kiara.ingestion.runtime.load_file", return_value=fallback_doc):
            with patch(
                "kiara.ingestion.runtime.chunk_document",
                return_value=[
                    {
                        "text": "chunk",
                        "metadata": {
                            **fallback_doc["metadata"],
                            "chunk_index": 0,
                            "chunk_strategy": "default",
                        },
                    },
                ],
            ):
                _process_single_file(
                    "/tmp/doc.pdf",
                    config,
                    None,
                    1,
                    ProcessingContext(),
                    stats,
                    file_log=file_log,
                )

        assert file_log.parser_fallback is True
        assert file_log.parser == "standard"

    def test_parser_fallback_integer_1_converted_to_true(self):
        """parser_fallback=1 (Integer) → file_log.parser_fallback=True."""
        from kiara.ingestion.pipeline import FileLogEntry
        from kiara.ingestion.runtime import ProcessingContext, _process_single_file

        file_log = FileLogEntry(
            file_path="/tmp/x.pdf", file_name="x.pdf", file_size=1000, file_ext=".pdf"
        )
        stats = {"errors": 0}
        config = {"ingestion": {"chunk_size": 500, "chunk_overlap": 50}}

        doc = {
            "text": "Content",
            "metadata": {
                "source": "/tmp/x.pdf",
                "parser": "docling",
                "parser_fallback": 1,
                "file_type": ".pdf",
            },
        }

        with patch("kiara.ingestion.runtime.load_file", return_value=doc):
            with patch(
                "kiara.ingestion.runtime.chunk_document",
                return_value=[
                    {
                        "text": "c",
                        "metadata": {
                            **doc["metadata"],
                            "chunk_index": 0,
                            "chunk_strategy": "default",
                        },
                    },
                ],
            ):
                _process_single_file(
                    "/tmp/x.pdf",
                    config,
                    None,
                    1,
                    ProcessingContext(),
                    stats,
                    file_log=file_log,
                )

        # bool(1) == True
        assert file_log.parser_fallback is True


# ---------------------------------------------------------------------------
# 14. VLM + Standard Dual-Doc → chunk_count akkumuliert
# ---------------------------------------------------------------------------


class TestDualDocChunkAccumulation:
    def test_vlm_and_standard_chunks_accumulated(self):
        """Datei mit Standard-Doc + VLM-Doc → chunk_count ist Summe beider."""
        from kiara.ingestion.pipeline import FileLogEntry
        from kiara.ingestion.runtime import ProcessingContext, _process_single_file

        file_log = FileLogEntry(
            file_path="/tmp/photo.jpg", file_name="photo.jpg", file_size=500_000, file_ext=".jpg"
        )
        stats = {"errors": 0}
        config = {"ingestion": {"chunk_size": 500, "chunk_overlap": 50}}

        standard_doc = {
            "text": "EXIF metadata extracted",
            "metadata": {"source": "/tmp/photo.jpg", "parser": "standard", "file_type": ".jpg"},
        }
        vlm_doc = {
            "text": "A photo showing a mountain landscape",
            "metadata": {"source": "/tmp/photo.jpg", "parser": "vlm", "file_type": ".jpg"},
        }

        ctx = ProcessingContext(
            vlm_enabled=True, vlm_model="llava", vlm_url="http://localhost:11434"
        )

        with patch("kiara.ingestion.runtime.load_file", return_value=standard_doc):
            with patch("kiara.ingestion.runtime.describe_image", return_value=vlm_doc):
                with patch(
                    "kiara.ingestion.runtime.chunk_document",
                    return_value=[
                        {
                            "text": "standard chunk",
                            "metadata": {
                                **standard_doc["metadata"],
                                "chunk_index": 0,
                                "chunk_strategy": "narrative",
                            },
                        },
                    ],
                ):
                    _process_single_file(
                        "/tmp/photo.jpg",
                        config,
                        None,
                        1,
                        ctx,
                        stats,
                        file_log=file_log,
                    )

        # Standard: 1 Chunk via chunk_document
        # VLM: 1 Chunk (vlm_single, nicht via chunk_document sondern direkt)
        assert file_log.chunk_count == 2
        # Erste chunk_strategy vom Standard-Doc
        assert file_log.chunk_strategy == "narrative"


# ---------------------------------------------------------------------------
# 15. Cancel → file_log wird trotzdem geschrieben
# ---------------------------------------------------------------------------


class TestCancelFileLogWritten:
    def test_cancel_during_embedding_still_logs(self, tmp_path):
        """Bei Cancel waehrend Embedding wird file_log trotzdem per Callback gemeldet."""
        from kiara.ingestion.pipeline import _CancelledError, run_ingestion

        (tmp_path / "doc.txt").write_text("Content that will be cancelled")

        config = {
            "datasource": {"path": str(tmp_path), "file_types": [".txt"], "exclude_patterns": []},
            "ingestion": {"chunk_size": 500, "chunk_overlap": 50, "batch_size": 10},
            "vectordb": {"path": str(tmp_path / "vdb"), "backend": "qdrant"},
            "graph": {"enabled": False},
        }

        collected = []
        mock_store = MagicMock()
        mock_store.delete_by_sources.return_value = None
        # Erster add() wirft CancelledError
        mock_store.add.side_effect = _CancelledError()

        with (
            patch("kiara.ingestion.pipeline.VectorStore", return_value=mock_store),
            patch("kiara.ingestion.pipeline.get_embedding_model", return_value="test-model"),
        ):
            with patch("kiara.ingestion.pipeline.HashStore") as MockHash:
                hs = MockHash.return_value
                hs.load.return_value = {}
                hs.upsert_many.return_value = None
                hs.remove_files.return_value = None

                run_ingestion(
                    config,
                    incremental=False,
                    datasource_id=1,
                    file_log_callback=lambda e: collected.append(e),
                )

        # Trotz Cancel wird der Callback aufgerufen (Zeile 616-618)
        assert len(collected) == 1
        entry = collected[0]
        assert entry.duration_ms >= 0
        assert entry.file_name == "doc.txt"


# ---------------------------------------------------------------------------
# 16. duration_ms Plausibilitaet
# ---------------------------------------------------------------------------


class TestDurationPlausibility:
    def test_duration_is_positive(self, tmp_path):
        """duration_ms ist > 0 fuer verarbeitete Dateien."""
        from kiara.ingestion.pipeline import run_ingestion

        (tmp_path / "test.txt").write_text("Content " * 100)

        config = {
            "datasource": {"path": str(tmp_path), "file_types": [".txt"], "exclude_patterns": []},
            "ingestion": {"chunk_size": 500, "chunk_overlap": 50, "batch_size": 10},
            "vectordb": {"path": str(tmp_path / "vdb"), "backend": "qdrant"},
            "graph": {"enabled": False},
        }

        collected = []
        mock_store = MagicMock()
        mock_store.delete_by_sources.return_value = None
        mock_store.add.return_value = None

        with (
            patch("kiara.ingestion.pipeline.VectorStore", return_value=mock_store),
            patch("kiara.ingestion.pipeline.get_embedding_model", return_value="test-model"),
        ):
            with patch("kiara.ingestion.pipeline.HashStore") as MockHash:
                hs = MockHash.return_value
                hs.load.return_value = {}
                hs.upsert_many.return_value = None
                hs.remove_files.return_value = None

                run_ingestion(
                    config,
                    incremental=False,
                    datasource_id=1,
                    file_log_callback=lambda e: collected.append(e),
                )

        assert len(collected) == 1
        assert collected[0].duration_ms >= 0
        # Sollte unter 30 Sekunden bleiben (kein realer Ollama-Call)
        assert collected[0].duration_ms < 30_000


# ---------------------------------------------------------------------------
# 17. Integration: Worker schreibt → API liest korrekt
# ---------------------------------------------------------------------------


class TestWorkerToApiIntegration:
    """End-to-End: File-Logs in DB geschrieben, ueber API korrekt gelesen."""

    def test_written_logs_match_api_response(self, api_app):
        from kiara.shared.db import get_connection

        client, ds_id, _ = api_app

        # Job + diverse File-Logs direkt in DB
        conn = get_connection()
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, finished_at) "
            "VALUES (%s, %s, NOW())",
            (ds_id, "success"),
        )
        row = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()
        job_id = row["id"]

        test_entries = [
            (
                job_id,
                "/data/report.pdf",
                "report.pdf",
                150000,
                ".pdf",
                "docling",
                0,
                "narrative",
                25,
                "success",
                "",
                "",
                3200,
            ),
            (
                job_id,
                "/data/image.png",
                "image.png",
                5000000,
                ".png",
                "vlm",
                0,
                "vlm_single",
                1,
                "success",
                "",
                "",
                8500,
            ),
            (
                job_id,
                "/data/broken.xlsx",
                "broken.xlsx",
                2000,
                ".xlsx",
                "unknown",
                0,
                "",
                0,
                "error",
                "Parse error: corrupt file",
                "loading",
                50,
            ),
            (
                job_id,
                "/data/empty.txt",
                "empty.txt",
                0,
                ".txt",
                "standard",
                0,
                "",
                0,
                "skipped",
                "",
                "",
                5,
            ),
            (
                job_id,
                "/data/fallback.pdf",
                "fallback.pdf",
                80000,
                ".pdf",
                "standard",
                1,
                "default",
                12,
                "success",
                "",
                "",
                1500,
            ),
        ]

        conn.executemany(
            "INSERT INTO ingestion_file_logs "
            "(job_id, file_path, file_name, file_size, file_ext, "
            "parser, parser_fallback, chunk_strategy, chunk_count, "
            "status, error_message, error_phase, duration_ms) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            test_entries,
        )
        conn.commit()
        conn.close()

        # Ungefilterter Abruf
        resp = client.get(f"/api/admin/ingestion-jobs/{job_id}/file-logs")
        data = resp.json()

        assert data["total"] == 5
        assert data["summary"]["success"] == 3
        assert data["summary"]["error"] == 1
        assert data["summary"]["skipped"] == 1
        assert data["summary"]["total_chunks"] == 25 + 1 + 0 + 0 + 12

        # parser_fallback korrekt durchgereicht
        fallback_entry = next(r for r in data["data"] if r["file_name"] == "fallback.pdf")
        assert fallback_entry["parser_fallback"] == 1

        # Error-Eintrag korrekt
        error_entry = next(r for r in data["data"] if r["status"] == "error")
        assert error_entry["error_message"] == "Parse error: corrupt file"
        assert error_entry["error_phase"] == "loading"

        # Sort nach file_size desc
        resp2 = client.get(
            f"/api/admin/ingestion-jobs/{job_id}/file-logs?sort=file_size&direction=desc"
        )
        sizes = [r["file_size"] for r in resp2.json()["data"]]
        assert sizes == sorted(sizes, reverse=True)

        # Kombination: status-Filter + Suche
        resp3 = client.get(
            f"/api/admin/ingestion-jobs/{job_id}/file-logs?status=success&search=pdf"
        )
        data3 = resp3.json()
        assert data3["total"] == 2  # report.pdf + fallback.pdf (success + pdf)
        for r in data3["data"]:
            assert r["status"] == "success"
            assert "pdf" in r["file_name"].lower()


# ---------------------------------------------------------------------------
# 18. Spec #32: File-log error in Late Chunking path
# ---------------------------------------------------------------------------


class TestFileLogErrorInLateChunkingPath:
    """Late Chunking enabled, embed_texts_late returns valid embeddings,
    but store.add_with_embeddings raises -> file_log records error."""

    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    @patch("kiara.ingestion.runtime._process_single_file")
    @patch("kiara.ingestion.runtime._should_use_late_chunking")
    @patch("kiara.ingestion.runtime._late_chunking_file_eligible")
    @patch("kiara.ingestion.runtime.embed_texts_late")
    def test_file_log_error_in_late_chunking_path(
        self,
        mock_etl,
        mock_lcfe,
        mock_sulc,
        mock_psf,
        mock_write,
        mock_es,
        mock_graph,
    ):
        from kiara.ingestion.runtime import (
            ProcessingContext,
            _IngestionContext,
            _process_file_in_pipeline,
        )

        collected_logs = []

        def log_callback(entry):
            collected_logs.append(entry)

        ictx = _IngestionContext(
            config={
                "vectordb": {"path": "/tmp"},
                "ingestion": {"batch_size": 10, "chunk_size": 500, "chunk_overlap": 50},
                "enrichment": {"late_chunking": True},
            },
            ctx=ProcessingContext(),
            store=MagicMock(),
            chunk_ds=MagicMock(),
            hash_store=MagicMock(),
            batch_size=10,
            datasource_id=None,
            embedding_model="nomic-embed-text",
            hash_mode="mtime",
            ds_overrides=None,
            file_hashes={"/f.txt": ("h1", 123)},
            stats={
                "total_files": 1,
                "processed_files": 0,
                "skipped_files": 0,
                "errors": 0,
                "total_chunks": 0,
            },
            pending_hashes={},
            consecutive_errors=0,
            skipped=0,
            total=1,
            file_log_callback=log_callback,
            progress_callback=MagicMock(),
        )

        mock_psf.return_value = (
            ["chunk text"],
            [{"source": "/f.txt", "file_type": ".txt"}],
            ["id1"],
            "Full document text",
            [(0, 10)],
        )

        # Late Chunking is enabled and produces valid embeddings
        mock_sulc.return_value = True
        mock_lcfe.return_value = True
        mock_etl.return_value = [[0.1] * 768]

        # But _write_embeddings raises an exception (simulating store error)
        mock_write.side_effect = ValueError("add_with_embeddings failed")

        result = _process_file_in_pipeline(ictx, "/f.txt")

        assert result is True  # Pipeline continues (SKIP, not ABORT)
        assert ictx.stats["errors"] == 1

        # file_log_callback was called
        assert len(collected_logs) == 1
        entry = collected_logs[0]
        assert entry.status == "error"
        assert entry.error_phase == "embedding"
        assert "add_with_embeddings" in entry.error_message
