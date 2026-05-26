"""Tests fuer die Ingestion-Jobs API-Endpoints.

Testet:
- POST /api/admin/ingestion-jobs (Enqueue)
- GET /api/admin/ingestion-jobs (Liste)
- GET /api/admin/ingestion-jobs/{id} (Detail)
- POST .../cancel, .../pause
- DELETE .../history
- Schedule CRUD
"""

import json
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from tests.admin_confirm import confirmed_request

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_INGESTION_SCHEMA = """
    DROP TABLE IF EXISTS ingestion_file_logs CASCADE;
    DROP TABLE IF EXISTS ingestion_jobs CASCADE;
    DROP TABLE IF EXISTS datasource_embeddings CASCADE;
    DROP TABLE IF EXISTS datasources CASCADE;
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
        ds_vlm_enabled BOOLEAN DEFAULT false,
        ds_accept_all_text INTEGER DEFAULT 0,
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
        id          SERIAL PRIMARY KEY,
        job_id      INTEGER NOT NULL,
        file_path   TEXT NOT NULL,
        file_name   TEXT NOT NULL,
        file_size   BIGINT DEFAULT 0,
        file_ext    TEXT DEFAULT '',
        parser      TEXT DEFAULT '',
        parser_fallback INTEGER DEFAULT 0,
        chunk_strategy  TEXT DEFAULT '',
        chunk_count INTEGER DEFAULT 0,
        status      TEXT NOT NULL DEFAULT 'success'
                    CHECK (status IN ('success', 'error', 'skipped')),
        error_message   TEXT DEFAULT '',
        error_phase     TEXT DEFAULT '',
        duration_ms     INTEGER DEFAULT 0,
        created_at  TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'localtime', 'YYYY-MM-DD HH24:MI:SS')),
        FOREIGN KEY (job_id) REFERENCES ingestion_jobs(id) ON DELETE CASCADE
    );
    DROP TABLE IF EXISTS cache_datasources CASCADE;
    DROP TABLE IF EXISTS cache_jobs CASCADE;
    CREATE TABLE cache_datasources (
        datasource_id INTEGER PRIMARY KEY,
        payload_json  TEXT NOT NULL DEFAULT '{}',
        updated_at    TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'localtime', 'YYYY-MM-DD HH24:MI:SS'))
    );
    CREATE TABLE cache_jobs (
        job_id        INTEGER PRIMARY KEY,
        datasource_id INTEGER DEFAULT 0,
        status        TEXT DEFAULT '',
        payload_json  TEXT NOT NULL DEFAULT '{}',
        updated_at    TEXT DEFAULT (to_char(CURRENT_TIMESTAMP AT TIME ZONE 'localtime', 'YYYY-MM-DD HH24:MI:SS'))
    );
    CREATE INDEX IF NOT EXISTS idx_cache_jobs_status ON cache_jobs(status);
"""


@pytest.fixture
def test_app(monkeypatch):
    """FastAPI TestClient mit PostgreSQL-DB."""
    from fastapi import FastAPI

    import kiara.shared.db as shared_db
    import kiara.web.db_utils as db_utils
    from kiara.shared.db import get_connection
    from kiara.web.api_ingestion_jobs import router

    conn = get_connection()
    conn.executescript(_INGESTION_SCHEMA)
    conn.commit()
    # Testdaten aufraeumen
    conn.execute("DELETE FROM ingestion_file_logs")
    conn.execute("DELETE FROM cache_jobs")
    conn.execute("DELETE FROM cache_datasources")
    conn.execute("DELETE FROM ingestion_jobs")
    conn.execute("DELETE FROM datasource_embeddings")
    conn.execute("DELETE FROM datasources")
    conn.commit()
    # Test-Datasource einfuegen
    conn.execute("INSERT INTO datasources (name, type) VALUES (%s, %s)", ("Test DS", "local"))
    conn.commit()
    # DS-ID ermitteln
    ds_row = conn.execute("SELECT id FROM datasources WHERE name = 'Test DS'").fetchone()
    ds_id = ds_row["id"]
    conn.close()

    @contextmanager
    def fake_db_query():
        c = get_connection()
        try:
            yield c
        finally:
            c.close()

    @contextmanager
    def fake_db_transaction():
        c = get_connection()
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    monkeypatch.setattr(db_utils, "db_query", fake_db_query)
    monkeypatch.setattr(db_utils, "db_transaction", fake_db_transaction)
    monkeypatch.setattr(shared_db, "db_query", fake_db_query)
    monkeypatch.setattr(shared_db, "db_transaction", fake_db_transaction)

    # Repository-Module nutzen eigene Imports von db_query/db_transaction
    # Nach kiara-db-Extraktion liegen die Symbole in kiara_db.*
    import kiara_db.base as repo_base
    import kiara_db.repositories.cache as repo_cache
    import kiara_db.repositories.datasource as repo_ds
    import kiara_db.repositories.job as repo_job

    monkeypatch.setattr(repo_base, "db_query", fake_db_query)
    monkeypatch.setattr(repo_base, "db_transaction", fake_db_transaction)
    monkeypatch.setattr(repo_job, "db_query", fake_db_query)
    monkeypatch.setattr(repo_job, "db_transaction", fake_db_transaction)
    monkeypatch.setattr(repo_cache, "db_query", fake_db_query)
    monkeypatch.setattr(repo_cache, "db_transaction", fake_db_transaction)
    monkeypatch.setattr(repo_ds, "db_query", fake_db_query)
    monkeypatch.setattr(repo_ds, "db_transaction", fake_db_transaction)

    from kiara.admin.registry import admin_registry, load_admin_functions
    from kiara.admin.router import generate_admin_routes
    from kiara.web.auth import get_admin_user

    load_admin_functions()

    app = FastAPI()
    app.include_router(router)
    generate_admin_routes(app, admin_registry)
    # Auth-Dependency ueberschreiben
    app.dependency_overrides[get_admin_user] = lambda: {
        "id": 1,
        "username": "admin",
        "is_admin": True,
    }

    # DB-basierte Worker-Funktionen mocken
    import kiara.web.ingestion_worker as worker_mod

    monkeypatch.setattr(worker_mod, "get_running_job_id", lambda: None)
    monkeypatch.setattr(worker_mod, "is_job_paused", lambda job_id: False)
    monkeypatch.setattr(worker_mod, "db_query", fake_db_query)
    monkeypatch.setattr(worker_mod, "db_transaction", fake_db_transaction)

    # Cache-Refresh nutzt eigene Imports von db_query/db_transaction
    import kiara.web.cache_refresh as cache_mod

    monkeypatch.setattr(cache_mod, "db_query", fake_db_query)
    monkeypatch.setattr(cache_mod, "db_transaction", fake_db_transaction)

    return TestClient(app), ds_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestListJobs:
    def test_empty_list(self, test_app):
        client, _ = test_app
        resp = client.get(
            "/api/admin/ingestion-jobs", headers={"X-Requested-With": "XMLHttpRequest"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["data"] == []

    def test_list_with_jobs(self, test_app):
        from kiara.shared.db import get_connection

        client, ds_id = test_app
        conn = get_connection()
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status) VALUES (%s, %s)", (ds_id, "success")
        )
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status) VALUES (%s, %s)", (ds_id, "pending")
        )
        conn.commit()
        conn.close()

        resp = client.get("/api/admin/ingestion-jobs")
        data = resp.json()
        assert data["total"] == 2
        assert len(data["data"]) == 2

    def test_filter_by_status(self, test_app):
        from kiara.shared.db import get_connection

        client, ds_id = test_app
        conn = get_connection()
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status) VALUES (%s, %s)", (ds_id, "success")
        )
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status) VALUES (%s, %s)", (ds_id, "error")
        )
        conn.commit()
        conn.close()

        # Ungefilterte Abfrage triggert Cold-Start und befuellt cache_jobs
        client.get("/api/admin/ingestion-jobs")

        resp = client.get("/api/admin/ingestion-jobs?status=error")
        data = resp.json()
        assert data["total"] == 1
        assert data["data"][0]["status"] == "error"


class TestEnqueueJob:
    def test_enqueue_success(self, test_app):
        client, ds_id = test_app
        resp = client.post(
            "/api/admin/ingestion-jobs",
            json={"datasource_id": ds_id, "embedding_model": "nomic-embed-text"},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data

    def test_enqueue_nonexistent_ds(self, test_app):
        client, _ = test_app
        resp = client.post(
            "/api/admin/ingestion-jobs",
            json={"datasource_id": 999999},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 404

    def test_enqueue_duplicate_rejected(self, test_app):
        from kiara.shared.db import get_connection

        client, ds_id = test_app
        conn = get_connection()
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status) VALUES (%s, %s)", (ds_id, "pending")
        )
        conn.commit()
        conn.close()

        resp = client.post(
            "/api/admin/ingestion-jobs",
            json={"datasource_id": ds_id},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 409


class TestGetJob:
    def test_get_existing_job(self, test_app):
        from kiara.shared.db import get_connection

        client, ds_id = test_app
        conn = get_connection()
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, stats_json) VALUES (%s, %s, %s)",
            (ds_id, "success", json.dumps({"total_files": 10})),
        )
        job_row = conn.execute(
            "SELECT id FROM ingestion_jobs WHERE datasource_id = %s ORDER BY id DESC LIMIT 1",
            (ds_id,),
        ).fetchone()
        job_id = job_row["id"]
        conn.commit()
        conn.close()

        resp = client.get(f"/api/admin/ingestion-jobs/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["stats"]["total_files"] == 10

    def test_get_nonexistent_job(self, test_app):
        client, _ = test_app
        resp = client.get("/api/admin/ingestion-jobs/999999")
        assert resp.status_code == 404


class TestDeleteHistory:
    def test_delete_old_jobs(self, test_app):
        from kiara.shared.db import get_connection

        client, ds_id = test_app
        conn = get_connection()
        # Job von vor 60 Tagen
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, finished_at) "
            "VALUES (%s, %s, NOW() - INTERVAL '60 days')",
            (ds_id, "success"),
        )
        # Job von heute
        conn.execute(
            "INSERT INTO ingestion_jobs (datasource_id, status, finished_at) "
            "VALUES (%s, %s, NOW())",
            (ds_id, "success"),
        )
        conn.commit()
        conn.close()

        resp = confirmed_request(
            client,
            "DELETE",
            "/api/admin/ingestion-jobs/history?days=30",
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] == 1
