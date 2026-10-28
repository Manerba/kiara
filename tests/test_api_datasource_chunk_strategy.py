"""API-Roundtrip-Tests fuer ds_chunk_strategy (Phase 7 + Phase 11.1 Refactoring).

Testet POST, PUT, GET der Datasource-API bezueglich Chunk-Strategie:
  - POST speichert ds_chunk_strategy korrekt
  - PUT aendert ds_chunk_strategy
  - GET liefert ds_chunk_strategy zurueck
  - Ungueltiger Wert wird trotzdem gespeichert (keine Server-Validierung)

Phase 11.1: connector_config JSON-Spalte statt Einzelspalten.
"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kiara.shared.db import get_connection
from kiara.web.api_datasource import router
from kiara.web.auth import get_admin_user
from tests.admin_confirm import confirmed_request

# ---------------------------------------------------------------------------
# DB-Schema (Phase 11.1: connector_config statt 30+ Einzelspalten)
# ---------------------------------------------------------------------------

_FULL_DS_SCHEMA = """
    DROP TABLE IF EXISTS cache_jobs CASCADE;
    DROP TABLE IF EXISTS cache_datasources CASCADE;
    DROP TABLE IF EXISTS ingestion_jobs CASCADE;
    DROP TABLE IF EXISTS datasource_embeddings CASCADE;
    DROP TABLE IF EXISTS datasource_users CASCADE;
    DROP TABLE IF EXISTS datasource_groups CASCADE;
    DROP TABLE IF EXISTS datasources CASCADE;

    CREATE TABLE datasources (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT NOT NULL DEFAULT 'local',
        path TEXT DEFAULT '',
        is_active BOOLEAN DEFAULT true,
        created_at TEXT DEFAULT (NOW()),
        updated_at TEXT DEFAULT (NOW()),
        connector_config TEXT DEFAULT '{}',
        ds_chunk_strategy TEXT DEFAULT 'auto',
        ds_chunk_size INTEGER DEFAULT NULL,
        ds_chunk_overlap INTEGER DEFAULT NULL,
        ds_file_types TEXT DEFAULT NULL,
        ds_exclude_patterns TEXT DEFAULT NULL,
        ds_ddu_enabled BOOLEAN DEFAULT false,
        ds_ddu_ocr_images INTEGER DEFAULT 1,
        ds_vlm_enabled BOOLEAN DEFAULT false,
        ds_accept_all_text INTEGER DEFAULT 0,
        ds_index_attachments INTEGER DEFAULT 1,
        sync_interval TEXT DEFAULT 'global',
        restriction_level TEXT DEFAULT 'anonymize',
        ds_contextual_retrieval INTEGER DEFAULT 0,
        ds_enrichment_prompt_id INTEGER DEFAULT NULL,
        ds_enrichment_backend_id INTEGER DEFAULT NULL,
        ds_colbert_enabled BOOLEAN DEFAULT false
    );

    CREATE TABLE datasource_groups (
        id SERIAL PRIMARY KEY,
        datasource_id INTEGER NOT NULL,
        group_id INTEGER NOT NULL
    );

    CREATE TABLE datasource_users (
        id SERIAL PRIMARY KEY,
        datasource_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL
    );

    CREATE TABLE datasource_embeddings (
        id SERIAL PRIMARY KEY,
        datasource_id INTEGER NOT NULL,
        embedding_model TEXT NOT NULL,
        chunk_count INTEGER DEFAULT 0,
        file_count INTEGER DEFAULT 0,
        last_ingested_at TEXT DEFAULT NULL,
        UNIQUE(datasource_id, embedding_model)
    );

    CREATE TABLE ingestion_jobs (
        id SERIAL PRIMARY KEY,
        datasource_id INTEGER NOT NULL,
        status TEXT DEFAULT 'pending',
        embedding_model TEXT DEFAULT '',
        created_at TEXT DEFAULT (NOW())
    );

    CREATE TABLE cache_datasources (
        datasource_id INTEGER PRIMARY KEY,
        payload_json  TEXT NOT NULL DEFAULT '{}',
        updated_at    TEXT DEFAULT (NOW())
    );

    CREATE TABLE cache_jobs (
        job_id        INTEGER PRIMARY KEY,
        datasource_id INTEGER DEFAULT 0,
        status        TEXT DEFAULT '',
        payload_json  TEXT NOT NULL DEFAULT '{}',
        updated_at    TEXT DEFAULT (NOW())
    );
    CREATE INDEX IF NOT EXISTS idx_cache_jobs_status ON cache_jobs(status);
"""


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """API TestClient mit PostgreSQL-DB und gemockten Abhaengigkeiten."""
    conn = get_connection()
    conn.executescript(_FULL_DS_SCHEMA)
    conn.commit()
    conn.close()

    from kiara.admin.registry import admin_registry, load_admin_functions
    from kiara.admin.router import generate_admin_routes

    load_admin_functions()

    app = FastAPI()
    app.include_router(router)
    generate_admin_routes(app, admin_registry)

    async def mock_admin():
        return {"id": 1, "username": "admin", "is_admin": True}

    app.dependency_overrides[get_admin_user] = mock_admin

    with (
        patch("kiara.shared.crypto.encrypt", side_effect=lambda x: x),
        patch("kiara.admin.services.datasource_service.encrypt", side_effect=lambda x: x),
        patch("kiara.shared.crypto.is_encrypted", return_value=False),
        patch(
            "kiara.config.load_config",
            return_value={"datasource": {"path": "/tmp"}, "chunk_size": 1000, "chunk_overlap": 200},
        ),
        patch(
            "kiara.admin.services.datasource_service.load_config",
            return_value={"ingestion": {"chunk_size": 1000, "chunk_overlap": 200}},
        ),
        patch("kiara.web.ingestion_worker.get_running_job_id", return_value=None),
        patch("kiara.web.ingestion_worker.is_job_paused", return_value=False),
    ):
        with TestClient(app) as c:
            yield c

    # Cleanup + Schema wiederherstellen
    from db_helpers import restore_test_schema

    restore_test_schema()


def _create_ds(client, name="Test DS", chunk_strategy="auto", **extra):
    """Hilfsfunktion: Erstellt eine Datasource via POST."""
    payload = {
        "name": name,
        "type": "local",
        "connector_config": {"path": "/tmp/test"},
        "ds_chunk_strategy": chunk_strategy,
        **extra,
    }
    resp = client.post("/api/admin/datasources", json=payload)
    assert resp.status_code == 200, f"POST failed: {resp.json()}"
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_post_stores_chunk_strategy(client):
    """POST /api/admin/datasources mit ds_chunk_strategy='prose' → DB speichert 'prose'."""
    ds_id = _create_ds(client, chunk_strategy="prose")

    resp = client.get("/api/admin/datasources")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 1
    assert data[0]["id"] == ds_id
    assert data[0]["ds_chunk_strategy"] == "prose"


def test_put_updates_chunk_strategy(client):
    """PUT /api/admin/datasources/{id} aendert ds_chunk_strategy von 'auto' auf 'code'."""
    ds_id = _create_ds(client, chunk_strategy="auto")

    resp = confirmed_request(
        client,
        "PUT",
        f"/api/admin/datasources/{ds_id}",
        json={
            "name": "Test DS",
            "type": "local",
            "connector_config": {"path": "/tmp/test"},
            "ds_chunk_strategy": "code",
        },
    )
    assert resp.status_code == 200

    resp = client.get("/api/admin/datasources")
    data = resp.json()["data"]
    assert data[0]["ds_chunk_strategy"] == "code"


def test_get_returns_chunk_strategy(client):
    """GET /api/admin/datasources liefert ds_chunk_strategy korrekt zurueck."""
    _create_ds(client, name="DS Prose", chunk_strategy="prose")
    _create_ds(client, name="DS Tabular", chunk_strategy="tabular")

    resp = client.get("/api/admin/datasources")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 2

    strategies = {d["name"]: d["ds_chunk_strategy"] for d in data}
    assert strategies["DS Prose"] == "prose"
    assert strategies["DS Tabular"] == "tabular"


def test_invalid_chunk_strategy_accepted(client):
    """Ungueltiger Wert 'invalid' wird ohne Fehler gespeichert (keine Validierung)."""
    ds_id = _create_ds(client, chunk_strategy="invalid")

    resp = client.get("/api/admin/datasources")
    data = resp.json()["data"]
    assert data[0]["ds_chunk_strategy"] == "invalid"


def test_post_stores_connector_config(client):
    """POST speichert connector_config als JSON."""
    payload = {
        "name": "SMB Test",
        "type": "smb",
        "connector_config": {
            "server": "192.168.1.10",
            "share": "Daten",
            "username": "admin",
            "password": "pw",
        },
    }
    resp = client.post("/api/admin/datasources", json=payload)
    assert resp.status_code == 200, f"POST failed: {resp.text}"

    resp = client.get("/api/admin/datasources")
    data = resp.json()["data"]
    ds = data[0]
    # password should be masked in GET response
    assert ds["connector_config"]["server"] == "192.168.1.10"
    assert ds["connector_config"]["password"] == "********"


def test_connector_types_endpoint(client):
    """GET /api/admin/connector-types liefert alle registrierten Konnektoren."""
    resp = client.get("/api/admin/connector-types")
    assert resp.status_code == 200
    types = resp.json()["data"]
    names = [t["name"] for t in types]
    assert "local" in names
    assert "smb" in names
    assert "imap" in names
    assert "exchange_online" in names
    assert "sql" in names
    assert "webdav" in names
    # Jeder Typ hat ein Schema
    for t in types:
        assert "schema" in t
        assert "supports_browse" in t
        assert isinstance(t["schema"], dict)
